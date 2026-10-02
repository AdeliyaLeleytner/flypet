#!/usr/bin/env python3
"""Threshold rule plus the task's stated prior, added after seeing that Claude never names more than two odours.

Re-scores the lookup player's own diagnose episodes (same measurements): mean measured shift per odour against the
untrained fly, label the at most two largest shifts beyond +-0.10, everything else 0. Scoring as in flypet.flytalk.
Writes data/flytalk_20260926/lookup_prior.json (rows in the baselines format, player "lookup_prior").
"""

from __future__ import annotations
import json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data/flytalk_20260926"
PANEL = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
]


def main(naive_path=None):
    rows = [
        r
        for r in json.load(open(D / "baselines.json"))
        if r["player"] == "lookup" and r["task"] == "diagnose"
    ]
    if naive_path:
        naive = json.load(open(naive_path))["naive"]
    else:
        import urllib.request

        naive = json.loads(urllib.request.urlopen("http://127.0.0.1:8800/info", timeout=30).read())[
            "naive"
        ]
    out = []
    for r in rows:
        vals = defaultdict(list)
        for t in r["transcript"]:
            m = re.match(r"SMELL (.+)", t["action"])
            if m:
                vals[m.group(1).strip()].append(t["value"])
        d = {o: float(np.mean(vals[o])) - naive[o] for o in PANEL if vals[o]}
        keep = sorted([o for o in d if abs(d[o]) > 0.10], key=lambda o: -abs(d[o]))[:2]
        pred = {o: ("+" if d[o] > 0 else "-") if o in keep else "0" for o in PANEL}
        truth = r["truth"]
        tr = [o for o in PANEL if truth[o] != "0"]
        un = [o for o in PANEL if truth[o] == "0"]
        ur = float(np.mean([pred[o] == "0" for o in un]))
        cr = float(np.mean([pred[o] == truth[o] for o in tr])) if tr else float("nan")
        out.append(
            {
                "player": "lookup_prior",
                "task": "diagnose",
                "seed": r["seed"],
                "history": r["history"],
                "score": 0.5 * (cr + ur) if tr else ur,
                "exact": all(pred[o] == truth[o] for o in PANEL),
                "acc7": float(np.mean([pred[o] == truth[o] for o in PANEL])),
                "conditioned_right": cr,
                "untouched_right": ur,
                "pred": pred,
                "truth": truth,
                "n_actions": r["n_actions"],
            }
        )
    json.dump(out, open(D / "lookup_prior.json", "w"), indent=1)
    print(
        f"lookup_prior: n={len(out)} score {np.mean([r['score'] for r in out]):.3f} exact {np.mean([r['exact'] for r in out]):.3f}"
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
