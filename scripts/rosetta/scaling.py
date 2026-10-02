#!/usr/bin/env python3
"""Does a bigger language model align better with the fly? Qwen3-4B vs Qwen3-32B on the complete DoOR block.

Same protocol as forward_block.py / analyze_block.py, no new simulations:
- forward (words -> fly input): ridge per glomerulus, 8 fixed folds over the 111 block odorants, trained on every
  other DoOR odorant; per-glomerulus r with a 95% bootstrap interval over odorants, and input-level retrieval;
- inverse (fly -> words) from the recorded KC and lateral-horn codes of the real odorants, same folds;
- RSA of each model's name geometry against the KC and LH codes.
Pre-registered comparison: the middle layer (4B layer 18 of 36, 32B layer 32 of 64); other depths reported as secondary.
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rosetta"))
from analyze_block import LEVELS, _Order, corr_rows, ranks_of_truth, rdm, summary, upper
from flypet import rosetta as R
from flypet.odor import DoorOdor

OLD, NEW = ROOT / "data/rosetta_20260925", ROOT / "data/rosetta_scaling_20260926"


def spaces():
    out = {}
    for prompt, f4 in (("name", "qwen4b_names.npz"), ("smell", "qwen4b_smells_like.npz")):
        Z = np.load(OLD / f4)
        for depth, key in zip(("1/4", "1/2", "3/4", "last"), ("L9", "L18", "L27", "L36")):
            out[("4B", prompt, depth)] = (Z["keys"], Z[key])
        p32 = NEW / f"qwen32b_{prompt}.npz"
        if p32.exists():
            Z = np.load(p32)
            for depth, key in zip(("1/4", "1/2", "3/4", "last"), ("L1", "L2", "L3", "L4")):
                out[("32B", prompt, depth)] = (Z["keys"], Z[key])
    return out


def main():
    door = DoorOdor()
    P = np.load(OLD / "block_predictions.npz")
    block, gl, folds = [str(b) for b in P["block"]], [str(g) for g in P["glomeruli"]], P["folds"]
    names = R.door_names(door)
    Yall = R.glomerular_matrix(door, names, gl)
    bkeys = [door.name2key[b.lower()] for b in block]
    sims = R.load_sims(OLD / "block_sims.npz")
    pops = R.populations(_Order())
    fly = {
        L: np.log1p(
            np.stack(
                [
                    np.mean([R.rates(sims[f"actual|{b}|{s}"], pops[L]) for s in R.SEEDS], 0)
                    for b in block
                ]
            )
        )
        for L in ("KC", "LH")
    }
    fly = {L: X[:, X.std(0) > 0] for L, X in fly.items()}
    fly_rdm = {L: rdm(X) for L, X in fly.items()}
    rng = np.random.default_rng(0)
    rows = []
    for (model, prompt, depth), (keys, E) in spaces().items():
        row = {k: i for i, k in enumerate(keys)}
        usable = [n for n in names if door.name2key[n.lower()] in row]
        X = E[[row[door.name2key[n.lower()]] for n in usable]]
        Yu = Yall[[names.index(n) for n in usable]]
        bidx = np.array([usable.index(b) for b in block])
        Pp = np.zeros((len(block), len(gl)))
        for f in range(8):
            test = bidx[folds == f]
            tm = np.ones(len(usable), bool)
            tm[test] = False
            for g in range(len(gl)):
                m = tm & ~np.isnan(Yu[:, g])
                sc = StandardScaler().fit(X[m])
                Pp[folds == f, g] = (
                    RidgeCV(alphas=np.logspace(0, 5, 11))
                    .fit(sc.transform(X[m]), Yu[m, g])
                    .predict(sc.transform(X[test]))
                )
        Pp = np.clip(Pp, 0, 1)
        Yb = P["actual"]
        r = np.nanmean([np.corrcoef(Yb[:, g], Pp[:, g])[0, 1] for g in range(len(gl))])
        bs = [
            np.nanmean([np.corrcoef(Yb[b, g], Pp[b, g])[0, 1] for g in range(len(gl))])
            for b in (rng.choice(len(Yb), len(Yb)) for _ in range(200))
        ]
        mu = Yb.mean(0, keepdims=True)
        fwd = summary(ranks_of_truth(corr_rows(Pp - mu, Yb - mu)), len(block))
        Eb = StandardScaler().fit_transform(E[[row[k] for k in bkeys]])
        inv, rsa = {}, {}
        for L, F in fly.items():
            pred = np.zeros_like(Eb)
            for f in range(8):
                tr, te = folds != f, folds == f
                sc = StandardScaler().fit(F[tr])
                pred[te] = (
                    RidgeCV(alphas=np.logspace(0, 6, 13))
                    .fit(sc.transform(F[tr]), Eb[tr])
                    .predict(sc.transform(F[te]))
                )
            inv[L] = summary(ranks_of_truth(corr_rows(pred, Eb)), len(block))
            rsa[L] = float(spearmanr(upper(fly_rdm[L]), upper(rdm(Eb))).statistic)
        rows.append(
            {
                "model": model,
                "prompt": prompt,
                "depth": depth,
                "r": float(r),
                "r_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "forward_top5": fwd["top5"],
                "forward_median_rank": fwd["median_rank"],
                "inverse_LH_top5": inv["LH"]["top5"],
                "inverse_LH_median_rank": inv["LH"]["median_rank"],
                "inverse_KC_median_rank": inv["KC"]["median_rank"],
                "rsa_KC": rsa["KC"],
                "rsa_LH": rsa["LH"],
            }
        )
        print(
            f"{model:4s} {prompt:5s} {depth:4s}  r {r:.3f} [{rows[-1]['r_ci'][0]:.3f}, {rows[-1]['r_ci'][1]:.3f}]  "
            f"fwd top5 {fwd['top5'] * 100:4.1f}% rank {fwd['median_rank']:4.0f}  inv LH rank {inv['LH']['median_rank']:4.0f} KC rank {inv['KC']['median_rank']:4.0f}  "
            f"RSA KC {rsa['KC']:+.3f} LH {rsa['LH']:+.3f}",
            flush=True,
        )
    NEW.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(NEW / "scaling.json", "w"), indent=1)


if __name__ == "__main__":
    main()
