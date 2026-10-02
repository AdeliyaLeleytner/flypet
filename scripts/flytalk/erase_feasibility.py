#!/usr/bin/env python3
"""Can a single hidden memory be undone at all? For flies with one trained odour, try m opposite-sign trials
(m = 0..6) or m odour-alone exposures (m = 1..6) on that odour, knowing the history, and score the result.
Uses the FlyTalk server (seeds 500-999 are reserved for this analysis)."""

from __future__ import annotations
import argparse, json, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/flytalk_20260926"


def post(url, path, obj):
    req = urllib.request.Request(
        url + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--n", type=int, default=20)
    a = ap.parse_args(argv)
    info = json.loads(urllib.request.urlopen(a.url + "/info", timeout=60).read())
    singles = []
    for seed in range(500, 1000):
        ep = post(a.url, "/reset", {"task": "erase", "seed": seed})
        h = post(a.url, "/result", {"eid": ep["eid"]})["history"]
        post(a.url, "/step", {"eid": ep["eid"], "action": "FINISH"})
        if len(h) == 1:
            singles.append((seed, h[0]))
        if len(singles) >= a.n:
            break
    plans = []
    for seed, (odour, reinf, n) in singles:
        opp = "PUNISH" if reinf == "reward" else "REWARD"
        for m in range(0, 7):
            plans.append((seed, odour, reinf, n, f"opposite x{m}", [f"TRAIN {odour} + {opp}"] * m))
        for m in range(1, 7):
            plans.append((seed, odour, reinf, n, f"expose x{m}", [f"EXPOSE {odour}"] * m))

    def run(plan):
        seed, odour, reinf, n, label, acts = plan
        ep = post(a.url, "/reset", {"task": "erase", "seed": seed})
        for x in acts:
            post(a.url, "/step", {"eid": ep["eid"], "action": x})
        post(a.url, "/step", {"eid": ep["eid"], "action": "FINISH"})
        res = post(a.url, "/result", {"eid": ep["eid"]})["result"]
        return {
            "seed": seed,
            "odour": odour,
            "reinf": reinf,
            "n": n,
            "strategy": label,
            "dev": res["final"][odour] - info["naive"][odour],
            "restored": res["restored"],
            "kept": res["kept"],
            "score": res["score"],
        }

    with ThreadPoolExecutor(24) as ex:
        rows = list(ex.map(run, plans))
    json.dump(rows, open(OUT / "erase_feasibility.json", "w"), indent=1)
    print(
        f"{len(singles)} single-memory flies; deviation of the trained odour from the untrained fly after each strategy"
    )
    for label in [f"opposite x{m}" for m in range(7)] + [f"expose x{m}" for m in range(1, 7)]:
        sel = [r for r in rows if r["strategy"] == label]
        print(
            f"  {label:12s} |dev| median {np.median([abs(r['dev']) for r in sel]):.3f}  restored {np.mean([r['restored'] for r in sel]):.2f}  "
            f"kept {np.mean([r['kept'] for r in sel]):.2f}  score {np.mean([r['score'] for r in sel]):.3f}"
        )
    best = {}
    for r in rows:
        k = r["seed"]
        if k not in best or r["score"] > best[k]["score"]:
            best[k] = r
    print(
        f"best strategy per fly (knowing the history): score {np.mean([b['score'] for b in best.values()]):.3f}, "
        f"restored {np.mean([b['restored'] for b in best.values()]):.2f}"
    )


if __name__ == "__main__":
    main()
