#!/usr/bin/env python3
"""Flatten evaluation episodes into per-fly rows for the shared FlyTalk comparison table.

Episodes in an eval_step*.json are in the order of the evaluation specs: for each task, seeds 0..n-1.
Usage: rows.py EVAL_JSON PLAYER [--out rows.json]  (merges by player, task and seed)
"""

import argparse, json
from pathlib import Path

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("eval_json", type=Path)
ap.add_argument("player")
ap.add_argument(
    "--out",
    type=Path,
    default=Path(__file__).resolve().parents[2] / "data/flytalk_rl_20260926/rows_qwen32b.json",
)
a = ap.parse_args()
eps = json.load(open(a.eval_json))
seen = {}
rows = []
for e in eps:
    seed = seen.get(e["task"], 0)
    seen[e["task"]] = seed + 1
    r = e["result"]
    rows.append(
        {
            "player": a.player,
            "task": e["task"],
            "seed": seed,
            "score": r["score"],
            "exact": bool(r["exact"]),
            "acc7": r.get("acc7"),
            "conditioned_right": r.get("conditioned_right"),
            "untouched_right": r.get("untouched_right"),
            "n_actions": r["n_actions"],
            "pred": r.get("pred"),
            "truth": r.get("truth"),
            "history": e["history"],
            "actions": e["texts"],
        }
    )
old = json.load(open(a.out)) if a.out.exists() else []
keep = [
    x
    for x in old
    if (x["player"], x["task"], x["seed"])
    not in {(y["player"], y["task"], y["seed"]) for y in rows}
]
json.dump(keep + rows, open(a.out, "w"), indent=1, default=str)
d = [x for x in rows if x["task"] == "diagnose"]
if d:
    import math

    mean = lambda k: (
        sum(
            x[k]
            for x in d
            if x[k] is not None and not (isinstance(x[k], float) and math.isnan(x[k]))
        )
        / max(
            1,
            sum(
                1
                for x in d
                if x[k] is not None and not (isinstance(x[k], float) and math.isnan(x[k]))
            ),
        )
    )
    print(
        f"{a.player}: diagnose n={len(d)} score {mean('score'):.3f} exact {sum(x['exact'] for x in d) / len(d):.3f} "
        f"trained right {mean('conditioned_right'):.3f} untrained right {mean('untouched_right'):.3f} smells {mean('n_actions'):.2f}"
    )
