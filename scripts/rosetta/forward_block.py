#!/usr/bin/env python3
"""Words -> fly, on the complete DoOR block: predict each held-out odorant's glomerular input from text or
structure features, then run the actual and predicted inputs through the whole-brain model (3 seeds).

Features are fixed in advance: Qwen3-4B-Base layer 18 for "The odorant {name}" (mean over name tokens) and
"{name} smells like" (last token); Morgan fingerprints; character n-grams of the name; shuffled Qwen rows.
Training uses every DoOR odorant outside the test fold, per glomerulus where it is measured.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import rosetta as R
from flypet.experiments import DEFAULT_PROTOCOL, load_protocol
from flypet.odor import DoorOdor

DATA = ROOT / "data/rosetta_20260925"


def features(door, names):
    keys = [door.name2key[n.lower()] for n in names]
    E = np.load(DATA / "qwen4b_names.npz")
    S = np.load(DATA / "qwen4b_smells_like.npz")
    M = np.load(DATA / "morgan.npz")
    e_row = {k: i for i, k in enumerate(E["keys"])}
    s_row = {k: i for i, k in enumerate(S["keys"])}
    m_row = {k: i for i, k in enumerate(M["keys"])}
    ok = [i for i, k in enumerate(keys) if k in e_row and k in s_row and k in m_row]
    keys = [keys[i] for i in ok]
    names = [names[i] for i in ok]
    F = {
        "qwen_name_L18": E["L18"][[e_row[k] for k in keys]],
        "qwen_smell_L18": S["L18"][[s_row[k] for k in keys]],
        "morgan": M["X"][[m_row[k] for k in keys]].astype(np.float64),
        "ngrams": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        .fit_transform(names)
        .toarray(),
    }
    F["shuffled"] = F["qwen_name_L18"][np.random.default_rng(0).permutation(len(names))]
    return names, F


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--no-sim", action="store_true")
    args = ap.parse_args(argv)
    protocol = load_protocol(DEFAULT_PROTOCOL)
    door = DoorOdor(protocol["odor_rate_max_hz"], protocol["odor_min_response"])
    block, gl = R.complete_block(door, 23)
    names, F = features(door, R.door_names(door))
    block = [b for b in block if b in names]
    Yall = R.glomerular_matrix(door, names, gl)
    bidx = np.array([names.index(b) for b in block])
    Yb = Yall[bidx]
    print(
        f"block: {len(block)} odorants x {len(gl)} glomeruli; training pool {len(names)} odorants",
        flush=True,
    )
    folds = np.zeros(len(block), dtype=int)
    for f, (_, te) in enumerate(KFold(8, shuffle=True, random_state=0).split(block)):
        folds[te] = f
    preds = {}
    for fname, X in F.items():
        P = np.zeros_like(Yb)
        for f in range(8):
            test = bidx[folds == f]
            train_mask = np.ones(len(names), bool)
            train_mask[test] = False
            for g in range(len(gl)):
                m = train_mask & ~np.isnan(Yall[:, g])
                sc = StandardScaler().fit(X[m])
                P[folds == f, g] = (
                    RidgeCV(alphas=np.logspace(0, 5, 11))
                    .fit(sc.transform(X[m]), Yall[m, g])
                    .predict(sc.transform(X[test]))
                )
        preds[fname] = np.clip(P, 0, 1)
        r = [np.corrcoef(Yb[:, g], P[:, g])[0, 1] for g in range(len(gl)) if Yb[:, g].std() > 0]
        print(f"  {fname:16s} input level: per-glomerulus r {np.mean(r):.3f}", flush=True)
    np.savez(
        DATA / "block_predictions.npz",
        block=np.array(block),
        glomeruli=np.array(gl),
        actual=Yb,
        folds=folds,
        **{f"pred_{k}": v for k, v in preds.items()},
    )
    if args.no_sim:
        return 0
    jobs = []
    for i, name in enumerate(block):
        for seed in R.SEEDS:
            jobs.append((f"actual|{name}|{seed}", dict(zip(gl, Yb[i])), seed))
            for fname, P in preds.items():
                jobs.append((f"{fname}|{name}|{seed}", dict(zip(gl, P[i])), seed))
    print(f"simulating {len(jobs)} trials on {args.workers} workers", flush=True)
    t0 = time.time()
    sims = R.simulate(
        jobs,
        n_workers=args.workers,
        rate_max=door.rate_max,
        min_response=door.min_response,
        scale=protocol["odor_scale"],
        duration_ms=protocol["duration_ms"],
        checkpoint=DATA / "block_sims.partial.npz",
    )
    R.save_sims(
        DATA / "block_sims.npz",
        sims,
        {"glomeruli": gl, "seeds": R.SEEDS, "protocol": protocol["protocol_id"]},
    )
    json.dump(
        {
            "block": block,
            "glomeruli": gl,
            "n_jobs": len(jobs),
            "wall_s": round(time.time() - t0, 1),
        },
        open(DATA / "block_sims.json", "w"),
        indent=1,
    )
    print(f"done in {time.time() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
