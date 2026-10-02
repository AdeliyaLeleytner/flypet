"""Mushroom body as the pet's memory: words become odours, dopamine changes KC->MBON weights, MBONs report valence.

Design rules: signs and spiking dynamics are kept, input
enters through real olfactory receptor neurons (one glomerulus = one ORN type), output is read from identified
neurons (the 96 MBONs), and memory lives in the anatomical KC->MBON synapses, not in a side table.
"""

from __future__ import annotations
import json, hashlib, os, urllib.request
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from . import connectome as C
from .engine import Brain, StimInput, RunResult

EMBED_MODEL = os.environ.get("FLYPET_EMBED_MODEL", "qwen3-embedding:0.6b")
EMBED_INSTRUCT = (
    "Instruct: Represent the concept for grouping similar things by what they are\nQuery: "
)
OLLAMA = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# ----------------------------------------------------------------------------------------------- odours
def embed_texts(texts: list[str]) -> np.ndarray:
    """Semantic embedding via the local Ollama embedding model; falls back to a character hash (no semantics)."""
    try:
        req = urllib.request.Request(
            f"{OLLAMA}/api/embed",
            data=json.dumps(
                {"model": EMBED_MODEL, "input": [EMBED_INSTRUCT + t for t in texts]}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            e = np.asarray(json.load(r)["embeddings"], dtype=np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True)
    except Exception:
        out = []
        for t in texts:
            h = hashlib.sha256(t.lower().encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            v = rng.standard_normal(1024).astype(np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


class OdorEncoder:
    """text -> embedding -> fixed random projection onto glomeruli -> the k most driven glomeruli get ORN input.

    The projection is a fixed seeded interface (like FLM's B), which is unavoidable: words have no smell.
    Semantic similarity of words becomes overlap of glomerular sets, which is what makes generalisation testable.
    """

    def __init__(
        self,
        k_active: int = 3,
        seed: int = 4242,
        dim: int = 1024,
        rate_hz: float = 100.0,
        frac: float = 1.0,
    ):
        ann = C.annotations()
        ct = ann.cell_type.astype(str)
        orn = ann[(ann.cell_class == "olfactory") & (ct != "")]
        self.glomeruli = sorted(orn.cell_type.astype(str).unique())
        self.by_glom = {
            g: [int(x) for x in orn.index[orn.cell_type.astype(str) == g]] for g in self.glomeruli
        }
        rng = np.random.default_rng(seed)
        self.P = rng.standard_normal((dim, len(self.glomeruli))).astype(np.float32) / np.sqrt(dim)
        self.k, self.rate_hz, self.frac = k_active, rate_hz, frac
        self._cache: dict[str, np.ndarray] = {}

    def glomerular_code(self, text: str) -> list[str]:
        if text not in self._cache:
            self._cache[text] = embed_texts([text])[0]
        z = self._cache[text] @ self.P
        top = np.argsort(-z)[: self.k]
        return [self.glomeruli[i] for i in top]

    def stim(self, text: str, seed: int = 0) -> StimInput:
        rng = np.random.default_rng(seed)
        ids = []
        for g in self.glomerular_code(text):
            pool = self.by_glom[g]
            k = max(1, int(round(self.frac * len(pool))))
            ids += list(rng.choice(pool, k, replace=False)) if k < len(pool) else pool
        return StimInput(ids, self.rate_hz, key=f"odor:{text}")

    def similarity(self, a: str, b: str) -> float:
        sa, sb = set(self.glomerular_code(a)), set(self.glomerular_code(b))
        return len(sa & sb) / len(sa | sb)


# ----------------------------------------------------------------------------------------- mushroom body
@dataclass
class Valence:
    approach_hz: float
    avoid_hz: float
    score: float  # (approach - avoid) / (approach + avoid + 1)
    n_mbon_active: int
    top: list = field(default_factory=list)


class MushroomBody:
    """Populations, compartments (DAN->MBON wiring), dopamine-gated KC->MBON depression, valence readout."""

    def __init__(
        self,
        brain: Brain,
        eta: float = 0.6,
        floor: float = 0.05,
        kc_scale_hz: float = 15.0,
        dan_scale_hz: float = 20.0,
        dan_threshold: float = 0.3,
    ):
        self.b = brain
        ann = C.annotations()
        flyid2i = brain.flyid2i
        ct = ann.cell_type.astype(str)
        nt = ann.top_nt.astype(str)
        cc = ann.cell_class.astype(str)
        idx = lambda mask: np.array([flyid2i[int(x)] for x in ann.index[mask]], dtype=np.int64)
        self.KC, self.MBON, self.DAN = idx(cc == "Kenyon_Cell"), idx(cc == "MBON"), idx(cc == "DAN")
        self.PAM = idx((cc == "DAN") & ct.str.startswith("PAM"))
        self.PPL1 = idx((cc == "DAN") & ct.str.startswith("PPL1"))
        self.pam_ids = [int(brain.i2flyid[i]) for i in self.PAM]
        self.ppl1_ids = [int(brain.i2flyid[i]) for i in self.PPL1]
        self.mbon_type = {int(i): ct[brain.i2flyid[i]] for i in self.MBON}
        self.mbon_nt = {int(i): nt[brain.i2flyid[i]] for i in self.MBON}
        # valence sign per MBON (Aso et al. 2014): glutamatergic MBONs drive avoidance, cholinergic drive approach,
        # GABAergic MBON11 (gamma1pedc) drives approach; other GABAergic MBONs are left out of the score.
        self.sign = {}
        for i in self.MBON:
            t, n = self.mbon_type[int(i)], self.mbon_nt[int(i)]
            self.sign[int(i)] = (
                -1.0
                if n == "glutamate"
                else (1.0 if n == "acetylcholine" or t == "MBON11" else 0.0)
            )
        pre, post, w0 = brain.i_pre, brain._i_post, brain.w0
        # KC->MBON synapses
        self.syn_idx = np.flatnonzero(np.isin(pre, self.KC) & np.isin(post, self.MBON))
        self.syn_pre, self.syn_post = pre[self.syn_idx], post[self.syn_idx]
        # compartments: DANs with direct synapses onto each MBON, weighted by synapse count
        m = np.isin(pre, self.DAN) & np.isin(post, self.MBON)
        self.comp: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for mbon in self.MBON:
            sel = m & (post == mbon)
            dans, counts = np.unique(pre[sel], return_counts=False), None
            if sel.any():
                d = pre[sel]
                c = np.abs(w0[sel]).astype(np.float64)
                u, inv = np.unique(d, return_inverse=True)
                wsum = np.bincount(inv, weights=c)
                keep = wsum >= 3
                self.comp[int(mbon)] = (u[keep], wsum[keep] / max(1e-9, wsum[keep].sum()))
            else:
                self.comp[int(mbon)] = (np.array([], dtype=np.int64), np.array([]))
        self.eta, self.floor, self.kc_scale_hz, self.dan_scale_hz, self.dan_threshold = (
            eta,
            floor,
            kc_scale_hz,
            dan_scale_hz,
            dan_threshold,
        )
        # memory = learned factor on top of the base scale (which carries the model corrections)
        self.base_scale = brain.w_scale[self.syn_idx].astype(np.float32).copy()
        self.mem = np.ones(len(self.syn_idx), dtype=np.float32)
        self.log: list[dict] = []

    def _push(self):
        self.b.set_weights(self.syn_idx, self.base_scale * self.mem)

    # -- reinforcement inputs
    def reward(self, rate_hz: float = 60.0) -> StimInput:
        return StimInput(self.pam_ids, rate_hz, key="dopamine:PAM(reward)")

    def punishment(self, rate_hz: float = 60.0) -> StimInput:
        return StimInput(self.ppl1_ids, rate_hz, key="dopamine:PPL1(punishment)")

    # -- plasticity
    def learn(self, res: RunResult, note: str = "") -> dict:
        """Dopamine-gated depression of KC->MBON synapses: coincident KC activity and DAN activity in the
        MBON's compartment weaken the synapse (Hige et al. 2015; Aso & Rubin 2016). Persistent via Brain.w_scale."""
        v = res.rate_vector(self.b.n)
        a = np.minimum(1.0, v[self.syn_pre] / self.kc_scale_hz)  # KC activity per synapse
        dop = np.zeros(self.b.n, dtype=np.float32)
        for mbon, (dans, wts) in self.comp.items():
            if len(dans):
                level = float(np.minimum(1.0, (v[dans] * wts).sum() / self.dan_scale_hz))
                dop[mbon] = level if level >= self.dan_threshold else 0.0  # compartment specificity
        d = dop[self.syn_post]
        change = self.eta * a * d
        touched = change > 0
        if touched.any():
            self.mem[touched] = np.maximum(self.floor, self.mem[touched] * (1.0 - change[touched]))
            self._push()
        entry = {
            "note": note,
            "n_synapses_changed": int(touched.sum()),
            "mean_change": float(change[touched].mean()) if touched.any() else 0.0,
            "kc_active": int((v[self.KC] > 0).sum()),
            "dan_active": int((v[self.DAN] > 0).sum()),
            "mbons_with_dopamine": int((dop[self.MBON] > 0).sum()),
        }
        self.log.append(entry)
        return entry

    # -- readout
    def valence(self, res: RunResult, top_n: int = 6) -> Valence:
        v = res.rate_vector(self.b.n)
        ap = sum(v[i] for i in self.MBON if self.sign[int(i)] > 0)
        av = sum(v[i] for i in self.MBON if self.sign[int(i)] < 0)
        order = sorted(self.MBON, key=lambda i: -v[i])
        top = [
            (self.mbon_type[int(i)], self.mbon_nt[int(i)][:4], round(float(v[i]), 1))
            for i in order[:top_n]
            if v[i] > 0
        ]
        return Valence(
            float(ap),
            float(av),
            float((ap - av) / (ap + av + 1.0)),
            int((v[self.MBON] > 0).sum()),
            top,
        )

    def kc_code(self, res: RunResult) -> set:
        v = res.rate_vector(self.b.n)
        return set(np.flatnonzero(v[self.KC] > 0).tolist())

    # -- persistence
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, syn_idx=self.syn_idx, mem=self.mem)
        Path(str(path) + ".log.json").write_text(json.dumps(self.log, ensure_ascii=False, indent=1))

    def load(self, path: str | Path):
        z = np.load(path)
        assert np.array_equal(z["syn_idx"], self.syn_idx), (
            "synapse index mismatch (different neuron order?)"
        )
        self.mem = z["mem"].astype(np.float32)
        self._push()
        lp = Path(str(path) + ".log.json")
        if lp.exists():
            self.log = json.loads(lp.read_text())

    def reset(self):
        self.mem[:] = 1.0
        self._push()
        self.log = []

    def memory_strength(self) -> float:
        """Fraction of KC->MBON weight removed by learning (0 = naive)."""
        return float(1.0 - self.mem.mean())
