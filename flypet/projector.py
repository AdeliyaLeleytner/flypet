"""Runtime use of the trained state->soft-token projector: the LLM describes a probe by reading the
downstream population directly (no labels, no readout table)."""

from __future__ import annotations
import os
from hashlib import sha256
import json
from pathlib import Path
from typing import TYPE_CHECKING
import warnings
import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from .engine import RunResult

ROOT = Path(__file__).resolve().parent.parent
_CAND = [
    ROOT / "data" / "projector_gpu" / "brain_projector_clean_v1.pt",
    ROOT / "data" / "projector_gpu" / "brain_projector_06b_6k.pt",
    ROOT / "data" / "projector_gpu" / "brain_projector_06b.pt",
]
_WARNED_LEGACY = set()


def resolve_checkpoint(pt_path: str | Path | None = None) -> Path:
    """Resolve on each construction so newly promoted checkpoints are noticed."""
    requested = pt_path if pt_path is not None else os.environ.get("FLYPET_READER_CHECKPOINT")
    return (
        Path(requested).expanduser()
        if requested
        else next((c for c in _CAND if c.exists()), _CAND[0])
    )


DEFAULT_PT = resolve_checkpoint()  # retained for existing callers checking availability


def checkpoint_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_reader_checkpoint(pt_path=None):
    """Load trusted local weights and inspect provenance without loading an LLM."""
    path = resolve_checkpoint(pt_path).resolve()
    before = checkpoint_sha256(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint_sha256(path) != before:
        raise RuntimeError(f"Checkpoint changed while being read: {path}")
    feats = np.asarray(ck["feats"], dtype="<i8")
    mu, sd = (np.asarray(ck[key], dtype="<f4") for key in ("mu", "sd"))
    if (
        feats.ndim != 1
        or mu.shape != feats.shape
        or sd.shape != feats.shape
        or not np.all(np.isfinite(mu))
        or not np.all(np.isfinite(sd))
        or np.any(sd <= 0)
    ):
        raise ValueError(f"Invalid checkpoint feature normalization: {path}")
    has_split_manifest = bool(
        ck.get("manifest") and ck.get("split_fingerprint") and ck.get("dataset_sha256")
    )
    if not has_split_manifest and str(path) not in _WARNED_LEGACY:
        warnings.warn(
            f"Legacy reader checkpoint has no split manifest: {path}. "
            "Its evaluation provenance is unverified; bundled historical readers predate the leakage fix. Use a verified clean checkpoint "
            "or set FLYPET_READER_CHECKPOINT.",
            RuntimeWarning,
            stacklevel=2,
        )
        _WARNED_LEGACY.add(str(path))
    preprocessing = {
        "feature_count": len(feats),
        "feats_sha256": sha256(feats.tobytes()).hexdigest(),
        "mu_sha256": sha256(mu.tobytes()).hexdigest(),
        "sd_sha256": sha256(sd.tobytes()).hexdigest(),
        "transform": "log1p(rate), then (x - checkpoint_mu)/checkpoint_sd",
    }
    metadata = {
        "checkpoint_path": str(path),
        "checkpoint_sha256": before,
        "preprocessing": preprocessing,
        "preprocessing_sha256": sha256(
            json.dumps(preprocessing, sort_keys=True).encode()
        ).hexdigest(),
        "model": ck["model"],
        "model_revision": ck.get("model_revision"),
        "dataset_sha256": ck.get("dataset_sha256"),
        "split_fingerprint": ck.get("split_fingerprint"),
        "manifest": ck.get("manifest"),
        "has_split_manifest": has_split_manifest,
    }
    return path, ck, metadata


def _cache_error(path, reason):
    return ValueError(
        f"Cannot reuse RSA feature cache {path}: {reason}. "
        "Keep this historical cache; regenerate probes with scripts/rsa_v2.py "
        "--feats <fresh-path.npz> --out <fresh-result.json>, then give that same "
        "--feats path to scripts/rsa_functional.py."
    )


def load_feature_cache(path, reader_metadata, protocol=None):
    """Refuse unversioned or incompatible normalized features before LLM use."""
    with np.load(path, allow_pickle=False) as cached:
        if "metadata" not in cached.files:
            raise _cache_error(path, "missing provenance metadata")
        metadata = json.loads(str(cached["metadata"].item()))
        if metadata.get("schema_version") != 1:
            raise _cache_error(path, "unsupported metadata schema")
        reader = metadata.get("reader", {})
        for key in ("checkpoint_sha256", "preprocessing_sha256"):
            if reader.get(key) != reader_metadata[key]:
                raise _cache_error(path, f"{key} differs from the selected reader")
        if protocol is not None and metadata.get("protocol") != protocol:
            raise _cache_error(path, "probe protocol differs from this request")
        features, cats = cached["F"], cached["cats"]
        expected = reader_metadata["preprocessing"]["feature_count"]
        if (
            features.ndim != 2
            or features.shape[1] != expected
            or cats.ndim != 1
            or len(features) != len(cats)
            or not np.all(np.isfinite(features))
        ):
            raise _cache_error(path, "feature dimensions or values are invalid")
    return features, [str(cat) for cat in cats], metadata


def save_feature_cache(path, features, cats, reader_metadata, protocol):
    metadata = {"schema_version": 1, "reader": reader_metadata, "protocol": protocol}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            F=np.asarray(features, dtype=np.float32),
            cats=np.asarray(cats, dtype=str),
            metadata=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        )
    return metadata


class Projector(nn.Module):
    def __init__(self, k, tokens, d, hidden=1024):
        super().__init__()
        self.tokens, self.d = tokens, d
        self.net = nn.Sequential(nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, tokens * d))
        self.ln = nn.LayerNorm(d)

    def forward(self, x, emb_std):
        return self.ln(self.net(x).view(x.shape[0], self.tokens, self.d)) * emb_std


class BrainReader:
    """state (rate vector) -> caption text, via the projector and a frozen LLM."""

    def __init__(self, pt_path: str | Path | None = None, device: str | None = None):
        from transformers import AutoTokenizer, AutoModelForCausalLM

        self.checkpoint_path, ck, self.checkpoint_metadata = load_reader_checkpoint(pt_path)
        self.checkpoint_sha256 = self.checkpoint_metadata["checkpoint_sha256"]
        self.feats = np.asarray(ck["feats"])
        self.mu = np.asarray(ck["mu"], dtype=np.float32)
        self.sd = np.asarray(ck["sd"], dtype=np.float32)
        self.tokens = int(ck["tokens"])
        self.model_id = ck["model"]
        self.model_revision = ck.get("model_revision")
        self.dev = torch.device(device or ("mps" if torch.backends.mps.is_available() else "cpu"))
        revision = {"revision": self.model_revision} if self.model_revision else {}
        self.tok = AutoTokenizer.from_pretrained(self.model_id, **revision)
        self.llm = (
            AutoModelForCausalLM.from_pretrained(self.model_id, dtype=torch.float32, **revision)
            .to(self.dev)
            .eval()
        )
        self.llm.requires_grad_(False)
        self.E = self.llm.get_input_embeddings()
        d = self.E.weight.shape[1]
        self.emb_std = float(self.E.weight.std())
        self.proj = Projector(len(self.feats), self.tokens, d).to(self.dev).eval()
        self.proj.load_state_dict(ck["proj"])
        self.pos = {int(i): j for j, i in enumerate(self.feats)}

    def features(self, res: RunResult) -> np.ndarray:
        x = np.zeros(len(self.feats), dtype=np.float32)
        for i, c in res.counts.items():
            j = self.pos.get(int(i))
            if j is not None:
                x[j] = np.log1p(c / res.duration_s)
        return (x - self.mu) / self.sd

    @torch.no_grad()
    def describe(self, res: RunResult, max_new_tokens: int = 256) -> str:
        x = torch.as_tensor(self.features(res)[None], device=self.dev)
        soft = self.proj(x, self.emb_std)
        attn = torch.ones((1, self.tokens), dtype=torch.long, device=self.dev)
        g = self.llm.generate(
            inputs_embeds=soft,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tok.pad_token_id,
            eos_token_id=self.tok.eos_token_id,
        )
        return self.tok.decode(g[0], skip_special_tokens=True).strip()
