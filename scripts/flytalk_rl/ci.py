#!/usr/bin/env python3
"""Bootstrap 95% intervals over test flies for each player, and paired differences against a reference player."""

import json, sys
from pathlib import Path
import numpy as np

DIR = Path(__file__).resolve().parents[2] / "data/flytalk_rl_20260926"
rows = [
    r for f in sorted(DIR.glob("rows_*.json")) for r in json.load(open(f))
]  # every player's rows
ref = sys.argv[1] if len(sys.argv) > 1 else "qwen32b_zero"
rng = np.random.default_rng(0)
by = {}
for r in rows:
    if r["task"] == "diagnose":
        by.setdefault(r["player"], {})[r["seed"]] = r


def boot(x):
    x = np.asarray(x, float)
    b = [x[rng.integers(len(x), size=len(x))].mean() for _ in range(5000)]
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


for p, d in by.items():
    seeds = sorted(d)
    m, lo, hi = boot([d[s]["score"] for s in seeds])
    e, elo, ehi = boot([float(d[s]["exact"]) for s in seeds])
    line = f"{p:22s} n={len(seeds)} score {m:.3f} [{lo:.3f}, {hi:.3f}]  exact {e:.3f} [{elo:.3f}, {ehi:.3f}]"
    if p != ref and ref in by:
        common = [s for s in seeds if s in by[ref]]
        dm, dlo, dhi = boot([d[s]["score"] - by[ref][s]["score"] for s in common])
        line += f"  paired vs {ref}: {dm:+.3f} [{dlo:+.3f}, {dhi:+.3f}]"
    print(line)
