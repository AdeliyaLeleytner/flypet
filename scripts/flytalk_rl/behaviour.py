#!/usr/bin/env python3
"""What does a FlyTalk player do differently? Per player on diagnose rows: smell allocation, re-smells, false positives
per odour, and the similar-pair error (the partner of a trained ester labelled as trained)."""

import json, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

DIR = Path(__file__).resolve().parents[2] / "data/flytalk_rl_20260926"
rows = [
    r for f in sorted(DIR.glob("rows_*.json")) for r in json.load(open(f))
]  # every player's rows
players = sys.argv[1:] or sorted({r["player"] for r in rows})
PAIR = {"ethyl acetate": "methyl acetate", "methyl acetate": "ethyl acetate"}
for p in players:
    sel = [r for r in rows if r["player"] == p and r["task"] == "diagnose"]
    if not sel:
        continue
    smells = [
        [a.split(" ", 1)[1].strip() for a in r["actions"] if a.upper().startswith("SMELL ")]
        for r in sel
    ]
    distinct = np.mean([len(set(s)) for s in smells])
    resmell = np.mean([len(s) - len(set(s)) for s in smells])
    fp, fp_n, miss, miss_n = Counter(), Counter(), Counter(), Counter()
    pair_err, pair_n = 0, 0
    for r in sel:
        truth, pred = r["truth"], r["pred"] or {}
        for o, t in truth.items():
            if t == "0":
                fp_n[o] += 1
                fp[o] += pred.get(o, "0") != "0"
            else:
                miss_n[o] += 1
                miss[o] += pred.get(o) != t
        for o, t in truth.items():
            if t != "0" and o in PAIR and truth[PAIR[o]] == "0":
                pair_n += 1
                pair_err += pred.get(PAIR[o], "0") != "0"
    print(
        f"== {p}: n={len(sel)} smells {np.mean([len(s) for s in smells]):.1f}, distinct odours {distinct:.1f}, re-smells {resmell:.1f}"
    )
    print(
        "   false positives on untrained odours: "
        + ", ".join(f"{o} {fp[o]}/{fp_n[o]}" for o in fp_n)
    )
    print(
        "   misses on trained odours:            "
        + ", ".join(f"{o} {miss[o]}/{miss_n[o]}" for o in miss_n)
    )
    print(f"   ester partner labelled trained when only its twin was: {pair_err}/{pair_n}")
    counts = Counter(sum(1 for v in (r["pred"] or {}).values() if v != "0") for r in sel)
    print(
        f"   odours labelled trained per fly: {dict(sorted(counts.items()))}; more than two in {sum(v for k, v in counts.items() if k > 2)}/{len(sel)}"
    )
