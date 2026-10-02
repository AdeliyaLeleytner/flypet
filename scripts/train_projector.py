"""Neural state -> soft tokens -> frozen LLM with leakage-resistant evaluation.

Use --prepare-only to inspect deduplication, held-out groups and preprocessing
without importing torch/transformers, initializing a simulator, or using network.
"""

from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
import time
from uuid import uuid4

import numpy as np

ROOT = Path(os.environ.get("FLYPET_ROOT", str(Path(__file__).resolve().parents[1])))
sys.path.insert(0, str(ROOT))
from flypet.projector_data import KEYS, load_samples, prepare_samples, write_prepared, file_hash
from flypet.catalog import STIMULI


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=16)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--n_gen", "--n-gen", dest="n_gen", type=int, default=100)
    ap.add_argument("--gen-max-tokens", type=int, default=256)
    ap.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    ap.add_argument("--ckpt", action="store_true", help="gradient checkpointing in frozen LLM")
    ap.add_argument("--data-root", default=os.environ.get("FLYPET_DATA", str(ROOT / "data")))
    ap.add_argument("--out", help="new output directory; existing directories are refused")
    ap.add_argument("--features", choices=("downstream", "all"), default="downstream")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0, help="training RNG seed")
    ap.add_argument(
        "--split-seed", type=int, default=0, help="fixed data split/evaluation RNG seed"
    )
    ap.add_argument("--device", help="torch device; default cuda, then mps, then cpu")
    ap.add_argument("--validation-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.15)
    ap.add_argument("--min-frequency", type=int, default=3)
    ap.add_argument("--prepare-only", action="store_true")
    a = ap.parse_args()
    if min(a.epochs, a.tokens, a.batch, a.gen_max_tokens) < 1 or a.lr <= 0:
        ap.error("epochs, tokens, batch, gen-max-tokens and learning rate must be positive")
    if "/" in a.tag or "\\" in a.tag:
        ap.error("tag cannot contain path separators")
    from flypet import connectome as C

    min_index = C.neuron_order()[1][-1] if a.features == "downstream" else 0
    samples, sources = load_samples(a.data_root)
    prepared = prepare_samples(
        samples,
        min_index=min_index,
        min_frequency=a.min_frequency,
        seed=a.split_seed,
        validation_fraction=a.validation_fraction,
        test_fraction=a.test_fraction,
        n_gen=a.n_gen,
        sources=sources,
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    OUT = (
        Path(a.out)
        if a.out
        else Path(a.data_root) / "projector_runs" / f"{stamp}-{uuid4().hex[:8]}"
    )
    OUT.mkdir(parents=True, exist_ok=False)
    manifest = prepared["manifest"]
    manifest["code_sha256"] = {
        str(path.relative_to(ROOT)): file_hash(path)
        for path in (
            Path(__file__).resolve(),
            ROOT / "flypet/projector_data.py",
            ROOT / "flypet/catalog.py",
            ROOT / "flypet/connectome.py",
        )
    }
    manifest["neuron_index_provenance"] = (
        {str(path.relative_to(ROOT)): file_hash(path) for path in (C.PATH_COMP, C.PATH_ANN)}
        if a.features == "downstream"
        else {}
    )
    summary = write_prepared(prepared, OUT)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print("prepared:", OUT, flush=True)
    if a.prepare_only:
        return
    import torch
    from torch import nn

    dev = torch.device(
        a.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    X, D, feats, mu, sd = (prepared[key] for key in ("X", "D", "feats", "mu", "sd"))
    caps = prepared["captions"]
    train_idx, validation_idx, test_idx = (
        prepared["splits"][name] for name in ("train", "validation", "test")
    )
    sb_idx = prepared["interaction_rows"]

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(a.model)
    DT = getattr(torch, a.dtype)
    llm = AutoModelForCausalLM.from_pretrained(a.model, dtype=DT).to(dev).eval()
    llm.requires_grad_(False)
    if a.ckpt:
        llm.gradient_checkpointing_enable()
        llm.config.use_cache = False
    E = llm.get_input_embeddings()
    d = E.weight.shape[1]
    emb_std = float(E.weight.std())

    class Projector(nn.Module):
        def __init__(self, k, hidden=1024):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(k, hidden), nn.GELU(), nn.Linear(hidden, a.tokens * d)
            )
            self.ln = nn.LayerNorm(d)

        def forward(self, x):
            return (self.ln(self.net(x).view(x.shape[0], a.tokens, d)) * emb_std).to(DT)

    enc = [tok(c + tok.eos_token, add_special_tokens=False)["input_ids"] for c in caps]

    def batch_tensors(ids_list):
        L = max(map(len, ids_list))
        ids = torch.full((len(ids_list), L), tok.pad_token_id, dtype=torch.long)
        lab = torch.full((len(ids_list), L), -100, dtype=torch.long)
        att = torch.zeros((len(ids_list), L), dtype=torch.long)
        for i, s in enumerate(ids_list):
            ids[i, : len(s)] = torch.tensor(s)
            lab[i, : len(s)] = torch.tensor(s)
            att[i, : len(s)] = 1
        return ids.to(dev), lab.to(dev), att.to(dev)

    def forward_loss(proj, feats_np, rows):
        soft = proj(torch.as_tensor(feats_np[rows], device=dev))
        ids, lab, att = batch_tensors([enc[i] for i in rows])
        emb = E(ids)
        inputs = torch.cat([soft, emb], 1)
        labels = torch.cat(
            [torch.full((len(rows), a.tokens), -100, dtype=torch.long, device=dev), lab], 1
        )
        attn = torch.cat([torch.ones((len(rows), a.tokens), dtype=torch.long, device=dev), att], 1)
        return llm(inputs_embeds=inputs, attention_mask=attn, labels=labels).loss

    @torch.no_grad()
    def nll(proj, feats_np, rows):
        tot, cnt_t = 0.0, 0
        for o in range(0, len(rows), a.batch):
            r = rows[o : o + a.batch]
            loss = forward_loss(proj, feats_np, r)
            ntok = sum(len(enc[i]) for i in r)
            tot += float(loss) * ntok
            cnt_t += ntok
        return tot / cnt_t if cnt_t else None

    @torch.no_grad()
    def generate(proj, feats_np, rows):
        outs = []
        for o in range(0, len(rows), a.batch):
            r = rows[o : o + a.batch]
            soft = proj(torch.as_tensor(feats_np[r], device=dev))
            attn = torch.ones((len(r), a.tokens), dtype=torch.long, device=dev)
            g = llm.generate(
                inputs_embeds=soft,
                attention_mask=attn,
                max_new_tokens=a.gen_max_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            outs += [tok.decode(x, skip_special_tokens=True).strip() for x in g]
        return outs

    LABEL2KEY = {STIMULI[k].label_ru: k for k in KEYS}

    def parse(text):
        m = re.search(
            r"Сенсоры: (.*?)\. Запах: (.*?)\. Дофамин: (.*?)\. Тело: хоботок (\S+), побег (\S+), чистка антенн (\S+), ходьба (\S+), поворот (\S+)\.",
            text,
        )
        if not m:
            return None
        sens = (
            set()
            if m.group(1) == "ничего"
            else {LABEL2KEY.get(p.split(" (")[0].strip(), "?") for p in m.group(1).split("), ")}
        )
        return {
            "keys": sens,
            "odor": m.group(2),
            "dop": m.group(3),
            "proboscis": m.group(4),
            "escape": m.group(5),
            "groom": m.group(6),
            "walk": m.group(7),
            "turn": m.group(8),
        }

    def score(texts, rows):
        ok = {
            "parsed": 0,
            "keys_exact": 0,
            "keys_f1": 0.0,
            "odor": 0,
            "dop": 0,
            "proboscis": 0,
            "escape": 0,
            "groom": 0,
            "walk": 0,
            "turn": 0,
        }
        if not len(rows):
            return {key: None for key in ok}
        for t, i in zip(texts, rows):
            p, q = parse(t), parse(caps[i])
            if p is None:
                continue
            ok["parsed"] += 1
            ok["keys_exact"] += p["keys"] == q["keys"]
            tp = len(p["keys"] & q["keys"])
            size = len(p["keys"]) + len(q["keys"])
            ok["keys_f1"] += 2 * tp / size if size else 1.0
            for f in ("odor", "dop", "proboscis", "escape", "groom", "walk", "turn"):
                ok[f] += p[f] == q[f]
        return {k: round(v / len(rows), 6) for k, v in ok.items()}

    def train(feats_np, tag):
        proj = Projector(feats_np.shape[1]).to(dev)
        opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=0.01)
        steps = a.epochs * ((len(train_idx) + a.batch - 1) // a.batch)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        t0 = time.time()
        history = []
        best_nll, best_state, best_epoch = float("inf"), None, None
        for ep in range(a.epochs):
            proj.train()
            p = np.random.permutation(train_idx)
            tot = 0.0
            for o in range(0, len(p), a.batch):
                r = p[o : o + a.batch]
                loss = forward_loss(proj, feats_np, r)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(proj.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss.detach())
            proj.eval()
            val_nll = nll(proj, feats_np, validation_idx)
            if val_nll is None or not np.isfinite(val_nll):
                raise RuntimeError("Validation NLL is missing or nonfinite")
            train_nll = tot / ((len(p) + a.batch - 1) // a.batch)
            history.append(
                {"epoch": ep + 1, "train_batch_mean_nll": train_nll, "validation_nll": val_nll}
            )
            if val_nll < best_nll:
                best_nll, best_epoch = val_nll, ep + 1
                best_state = {
                    key: value.detach().cpu().clone() for key, value in proj.state_dict().items()
                }
            print(
                f"[{tag}] epoch {ep + 1}: train nll {train_nll:.3f} | validation nll {val_nll:.3f} | {(time.time() - t0) / 60:.1f} min",
                flush=True,
            )
        proj.load_state_dict(best_state)
        proj.eval()
        return proj, {
            "history": history,
            "selected_epoch": best_epoch,
            "selection": "minimum validation NLL",
        }

    checkpoint_meta = {
        "tokens": a.tokens,
        "model": a.model,
        "seed": a.seed,
        "split_seed": a.split_seed,
        "model_revision": getattr(llm.config, "_commit_hash", None),
        "dataset_sha256": manifest["dataset_sha256"],
        "split_fingerprint": manifest["split_fingerprint"],
        "manifest": "manifest.json",
    }
    brain, brain_training = train(X, "brain")
    torch.save(
        {"proj": brain.state_dict(), "feats": feats, "mu": mu, "sd": sd, **checkpoint_meta},
        OUT / f"brain_projector{a.tag}.pt",
    )
    direct, direct_training = train(D, "direct")
    torch.save(
        {
            "proj": direct.state_dict(),
            "words": prepared["words"],
            "keys": KEYS,
            "unknown_odor_column": manifest["preprocessing"]["unknown_odor_column"],
            **checkpoint_meta,
        },
        OUT / f"direct_projector{a.tag}.pt",
    )

    # The final held-out tests are first read after both models have been selected.
    class Zero(nn.Module):
        def forward(self, x):
            return torch.zeros((x.shape[0], a.tokens, d), device=dev, dtype=DT)

    shuffle_order = np.random.default_rng(a.split_seed + 2).permutation(test_idx)
    shuffled_source = np.roll(shuffle_order, 1)
    Xshuf = X.copy()
    Xshuf[shuffle_order] = X[shuffled_source]
    report = {
        "data_manifest": "manifest.json",
        "dataset_sha256": manifest["dataset_sha256"],
        "split_fingerprint": manifest["split_fingerprint"],
        "training": {"brain": brain_training, "direct": direct_training},
        "model_revision": getattr(llm.config, "_commit_hash", None),
        "versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": __import__("transformers").__version__,
        },
        "nll_test": {
            "brain": nll(brain, X, test_idx),
            "direct": nll(direct, D, test_idx),
            "brain_shuffled_states": nll(brain, Xshuf, test_idx),
            "zero_soft_prefix": nll(Zero(), X, test_idx),
        },
        "shuffle_test_pairs": [
            {
                "target_row_id": manifest["rows"][int(target)]["id"],
                "source_row_id": manifest["rows"][int(source)]["id"],
            }
            for target, source in zip(shuffle_order, shuffled_source)
        ],
    }
    rows = prepared["generation_rows"]
    gb, gd = generate(brain, X, rows), generate(direct, D, rows)
    report["gen_test"] = {"n": int(len(rows)), "brain": score(gb, rows), "direct": score(gd, rows)}
    print(
        "test NLL:", {k: None if v is None else round(v, 3) for k, v in report["nll_test"].items()}
    )
    print("generation accuracy on test:", report["gen_test"], flush=True)
    predictions = []

    def record_predictions(split, rows, brain_texts, direct_texts):
        for i, bt, dt in zip(rows, brain_texts, direct_texts):
            predictions.append(
                {
                    "split": split,
                    "row_id": manifest["rows"][int(i)]["id"],
                    "truth": caps[int(i)],
                    "brain": bt,
                    "direct": dt,
                }
            )

    record_predictions("test", rows, gb, gd)
    if len(sb_idx):
        gb2, gd2 = generate(brain, X, sb_idx), generate(direct, D, sb_idx)
        record_predictions("interaction", sb_idx, gb2, gd2)

        def prob_dist(texts):
            counts = {}
            for text in texts:
                parsed = parse(text)
                key = parsed["proboscis"] if parsed else "unparsed"
                counts[key] = counts.get(key, 0) + 1
            return counts

        report["interaction_sugar_bitter"] = {
            "n": int(len(sb_idx)),
            "truth_proboscis": prob_dist([caps[i] for i in sb_idx]),
            "brain_proboscis": prob_dist(gb2),
            "direct_proboscis": prob_dist(gd2),
            "brain": score(gb2, sb_idx),
            "direct": score(gd2, sb_idx),
            "brain_nll": nll(brain, X, sb_idx),
            "direct_nll": nll(direct, D, sb_idx),
        }
        print("interaction test:", report["interaction_sugar_bitter"], flush=True)
    report["config"] = vars(a) | {
        "n_train": int(len(train_idx)),
        "n_validation": int(len(validation_idx)),
        "n_test": int(len(test_idx)),
        "n_features": int(len(feats)),
        "device_resolved": str(dev),
    }
    (OUT / f"predictions{a.tag}.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2) + "\n"
    )
    report["artifact_hashes"] = {
        path.name: file_hash(path) for path in OUT.iterdir() if path.is_file()
    }
    (OUT / f"report{a.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
