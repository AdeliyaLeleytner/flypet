#!/usr/bin/env python3
"""Bayes player with the decision matched to the balanced score, on the same measurements.

bayes_player.py answers each odour with its most probable marginal label, which maximises per-odour accuracy. Here the
posterior over the 799 histories is rebuilt from the recorded observations, and the answer is the labelling of the seven
odours (3^7 candidates) with the highest expected balanced score. Writes data/flytalk_20260926/bayes_balanced.json.
"""

from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/flytalk"))
import bayes_player as B

D = ROOT / "data/flytalk_20260926"


def balanced(C, L):
    """C: candidates x 7, L: histories x 7 (codes 0 '+', 1 '-', 2 '0') -> candidates x histories expected score."""
    trained = L != 2  # hist x odour
    eq = C[:, None, :] == L[None, :, :]  # cand x hist x odour
    k = trained.sum(1)  # hist
    cr = (eq & trained[None]).sum(2) / np.maximum(k, 1)[None]
    ur = (eq & ~trained[None]).sum(2) / (7 - k)[None]
    return np.where(k[None] > 0, 0.5 * (cr + ur), ur)


def main():
    P = B.Player(D / "bayes_table.npz")
    L = P.labels
    C = np.array(list(itertools.product(range(3), repeat=len(B.PANEL))))
    S = balanced(C, L)  # 2187 x 799
    rows = json.load(open(D / "bayes_player.json"))
    out = []
    for r in rows:
        lp = P.logprior.copy()
        for t in r["transcript"]:
            if t["action"].startswith("SMELL "):
                o = B.PANEL.index(t["action"][6:].strip())
                x = float(t["observation"].split(":")[1].split("(")[0])
                lp = lp + P.loglik(o, x)
        w = np.exp(lp - lp.max())
        w /= w.sum()
        best = C[int(np.argmax(S @ w))]
        pred = {o: "+-0"[int(best[i])] for i, o in enumerate(B.PANEL)}
        truth = r["truth"]
        tr = [o for o in B.PANEL if truth[o] != "0"]
        un = [o for o in B.PANEL if truth[o] == "0"]
        ur = float(np.mean([pred[o] == "0" for o in un]))
        cr = float(np.mean([pred[o] == truth[o] for o in tr])) if tr else float("nan")
        out.append(
            {
                "player": "bayes_balanced",
                "task": "diagnose",
                "seed": r["seed"],
                "history": r["history"],
                "score": 0.5 * (cr + ur) if tr else ur,
                "exact": all(pred[o] == truth[o] for o in B.PANEL),
                "acc7": float(np.mean([pred[o] == truth[o] for o in B.PANEL])),
                "conditioned_right": cr,
                "untouched_right": ur,
                "pred": pred,
                "truth": truth,
                "n_actions": r["n_actions"],
            }
        )
    json.dump(out, open(D / "bayes_balanced.json", "w"), indent=1)
    print(
        f"bayes_balanced: n={len(out)} score {np.mean([r['score'] for r in out]):.3f} exact {np.mean([r['exact'] for r in out]):.3f} "
        f"trained {np.nanmean([r['conditioned_right'] for r in out]):.3f} untrained {np.mean([r['untouched_right'] for r in out]):.3f}"
    )


if __name__ == "__main__":
    main()
