#!/usr/bin/env python3
"""'Tell the fly a smell': held-key odorant transfer on the fixed PAIRS panel.

Stages: embed (local, Qwen3-4B-Base) -> translate (ridge to the 23 block glomeruli, keys held out or not)
-> simulate (brain, 3 seeds) -> analyze (rank of the key odorant among the 111 block odorants).
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/rosetta_20260925"
PAIRS = [
    ("banana", "isopentyl acetate"),
    ("vinegar", "acetic acid"),
    ("freshly cut grass", "Z3-hexenol"),
    ("rose", "phenethyl alcohol"),
    ("rancid butter", "butyric acid"),
    ("wintergreen", "methyl salicylate"),
    ("bitter almond", "benzaldehyde"),
    ("lavender", "linalool"),
    ("orange peel", "limonene"),
    ("clove", "eugenol"),
    ("mushroom", "1-octen-3-ol"),
    ("pineapple", "ethyl butyrate"),
    ("rotting flesh", "cadaverine"),
    ("pine needles", "alpha-pinene"),
    ("nail polish remover", "acetone"),
    ("butter popcorn", "2,3-butanedione"),
    ("vodka", "ethanol"),
    ("sweaty feet", "isopentanoic acid"),
    ("apple", "hexyl acetate"),
    ("smoky whisky", "4-ethylguaiacol"),
    ("cooked cabbage", "dimethyl sulfide"),
    ("household ammonia", "ammonium hydroxide"),
    ("geranium", "geraniol"),
    ("green leaves", "E2-hexenal"),
    ("sour milk", "lactic acid"),
]
PROMPTS = {"smell": ("{name} smells like", "last"), "name": ("The odorant {name}", "name")}


def embed(texts, template, pool, model_id="Qwen/Qwen3-4B-Base"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, local_files_only=True)
        .to("mps")
        .eval()
    )
    out = []
    with torch.no_grad():
        for w in texts:
            text = template.format(name=w)
            ids = tok(text, return_tensors="pt").to("mps")
            h = model(**ids, output_hidden_states=True).hidden_states[18][0]
            if pool == "last":
                out.append(h[-1].float().cpu().numpy())
            else:
                pre = template.split("{name}")[0].rstrip()
                n_pre = len(tok(pre)["input_ids"]) if pre else 0
                out.append(h[n_pre:].float().mean(0).cpu().numpy())
    return np.stack(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["embed", "translate", "simulate", "analyze"])
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args(argv)
    words = [w for w, _ in PAIRS]
    if a.stage == "embed":
        np.savez(
            DATA / "everyday_embeddings.npz",
            words=np.array(words),
            **{k: embed(words, t, p) for k, (t, p) in PROMPTS.items()},
        )
        print("embedded", len(words), "words with", list(PROMPTS))
        return
    from flypet import rosetta as R
    from flypet.odor import DoorOdor

    if a.stage == "translate":
        from sklearn.linear_model import RidgeCV
        from sklearn.preprocessing import StandardScaler

        door = DoorOdor()
        P = np.load(DATA / "block_predictions.npz")
        gl = [str(g) for g in P["glomeruli"]]
        names = R.door_names(door)
        src = {
            "smell": np.load(DATA / "qwen4b_smells_like.npz"),
            "name": np.load(DATA / "qwen4b_names.npz"),
        }
        W = np.load(DATA / "everyday_embeddings.npz")
        keys_lower = {k.lower() for _, k in PAIRS}
        out = {}
        for prompt, Z in src.items():
            row = {k: i for i, k in enumerate(Z["keys"])}
            usable = [n for n in names if door.name2key[n.lower()] in row]
            X = Z["L18"][[row[door.name2key[n.lower()]] for n in usable]]
            Y = R.glomerular_matrix(door, usable, gl)
            Xw = W[prompt]
            for variant, drop in (("A", set()), ("B", keys_lower)):
                keep = np.array([n.lower() not in drop for n in usable])
                pred = np.zeros((len(words), len(gl)))
                for g in range(len(gl)):
                    m = keep & ~np.isnan(Y[:, g])
                    sc = StandardScaler().fit(X[m])
                    pred[:, g] = (
                        RidgeCV(alphas=np.logspace(0, 5, 11))
                        .fit(sc.transform(X[m]), Y[m, g])
                        .predict(sc.transform(Xw))
                    )
                out[f"{prompt}_{variant}"] = np.clip(pred, 0, 1)
                perm = np.random.default_rng(0).permutation(len(words))
                out[f"{prompt}_{variant}_shuffled"] = np.clip(pred[perm], 0, 1)
        np.savez(DATA / "everyday_inputs.npz", words=np.array(words), glomeruli=np.array(gl), **out)
        print("translated:", list(out))
        return
    if a.stage == "simulate":
        from flypet.experiments import DEFAULT_PROTOCOL, load_protocol

        protocol = load_protocol(DEFAULT_PROTOCOL)
        door = DoorOdor(protocol["odor_rate_max_hz"], protocol["odor_min_response"])
        Z = np.load(DATA / "everyday_inputs.npz")
        gl = [str(g) for g in Z["glomeruli"]]
        jobs = [
            (f"{v}|{w}|{s}", dict(zip(gl, Z[v][i])), s)
            for v in Z.files
            if v not in ("words", "glomeruli")
            for i, w in enumerate(words)
            for s in R.SEEDS
        ]
        sims = R.simulate(
            jobs,
            n_workers=a.workers,
            rate_max=door.rate_max,
            min_response=door.min_response,
            scale=protocol["odor_scale"],
            duration_ms=protocol["duration_ms"],
        )
        R.save_sims(DATA / "everyday_sims.npz", sims, {"n_jobs": len(jobs)})
        print("simulated", len(jobs))
        return
    # analyze
    sys.path.insert(0, str(ROOT / "scripts/rosetta"))
    from analyze_block import _Order, corr_rows

    P = np.load(DATA / "block_predictions.npz")
    block = [str(b) for b in P["block"]]
    Z = np.load(DATA / "everyday_inputs.npz")
    base = R.load_sims(DATA / "block_sims.npz")
    ev = R.load_sims(DATA / "everyday_sims.npz")
    pops = R.populations(_Order())
    key_idx = [block.index(k) for _, k in PAIRS]
    res = {"pairs": PAIRS, "variants": {}}
    for level in ["input", "KC", "LH"]:
        if level == "input":
            A = P["actual"]
        else:
            A = np.log1p(
                np.stack(
                    [
                        np.mean([R.rates(base[f"actual|{b}|{s}"], pops[level]) for s in R.SEEDS], 0)
                        for b in block
                    ]
                )
            )
        live = A.std(0) > 0
        A = A[:, live]
        mu = A.mean(0, keepdims=True)
        for v in [f for f in Z.files if f not in ("words", "glomeruli")]:
            if level == "input":
                Q = Z[v][:, live]
            else:
                Q = np.log1p(
                    np.stack(
                        [
                            np.mean([R.rates(ev[f"{v}|{w}|{s}"], pops[level]) for s in R.SEEDS], 0)
                            for w in words
                        ]
                    )
                )[:, live]
            S = corr_rows(Q - mu, A - mu)
            ranks = [int((S[i] > S[i, k]).sum()) + 1 for i, k in enumerate(key_idx)]
            top3 = [[block[j] for j in np.argsort(-S[i])[:3]] for i in range(len(words))]
            res["variants"].setdefault(v, {})[level] = {
                "ranks": ranks,
                "median_rank": float(np.median(ranks)),
                "top5": float(np.mean(np.array(ranks) <= 5)),
                "top3_neighbours": top3,
            }
    json.dump(res, open(DATA / "everyday_analysis.json", "w"), indent=1)
    print(f"rank of the key odorant among {len(block)} (median, top-5 share); chance 56, 4.5%")
    for v, lv in res["variants"].items():
        print(
            f"  {v:18s} "
            + "  ".join(
                f"{L}: {d['median_rank']:5.1f} {d['top5'] * 100:4.0f}%" for L, d in lv.items()
            )
        )
    main_v = res["variants"]["smell_B"]["KC"]
    print("\nprimary (smell prompt, keys held out, Kenyon cells):")
    for (w, k), r, t3 in zip(PAIRS, main_v["ranks"], main_v["top3_neighbours"]):
        print(f"  {w:20s} key {k:20s} rank {r:3d}   nearest: {', '.join(t3)}")


if __name__ == "__main__":
    main()
