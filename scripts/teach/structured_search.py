#!/usr/bin/env python3
"""Headroom above the minimal textbook, found on the real fly: for each task, try 1-3 trials per targeted odour
(reward for 'approach more', punishment for 'avoid more', round-robin order, at most the budget), choose on
selection seeds (training 7000+, probe 301), and write the choice for scoring on the test seeds by eval_teachers."""

from __future__ import annotations
import argparse, json, sys, time
from itertools import product
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.teach import FlyPool, score
from flypet.teach_tasks import FastSurrogate, make_tasks

DATA = ROOT / "data/teach_20260925"


def variants(task):
    tg = [(o, "reward" if v > 0 else "punish") for o, v in task.targets.items() if v]
    out = []
    for counts in product((1, 2, 3), repeat=len(tg)):
        if sum(counts) > task.budget:
            continue
        left, seq = list(counts), []
        while any(left):
            for i, a in enumerate(tg):
                if left[i]:
                    seq.append(a)
                    left[i] -= 1
        out.append((counts, seq))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    t0 = time.time()
    sur = FastSurrogate(DATA / "surrogate.pkl")
    tasks = make_tasks(sur.panel, budget=sur.budget)
    fly = FlyPool(a.workers)
    naive_sel = fly.episodes([("naive", [], sur.panel, 7000, (301,))])["naive"]["valence"]
    var = {t.name: variants(t) for t in tasks}
    jobs = [
        (f"{t.name}||{i}", seq, sur.panel, 7000, (301,))
        for t in tasks
        for i, (_, seq) in enumerate(var[t.name])
    ]
    print(f"{len(jobs)} selection episodes", flush=True)
    res = fly.episodes(jobs, log_every=200)
    fly.close()
    picks = {}
    for t in tasks:
        best = max(
            range(len(var[t.name])),
            key=lambda i: score(t, res[f"{t.name}||{i}"]["valence"], naive_sel)[:2],
        )
        counts, seq = var[t.name][best]
        picks[t.name] = {
            "actions": seq,
            "counts": dict(zip([o for o, v in t.targets.items() if v], counts)),
            "selection_success": score(t, res[f"{t.name}||{best}"]["valence"], naive_sel)[0],
        }
    json.dump(picks, open(DATA / "teacher_structured.json", "w"), indent=1)
    print(
        f"done in {time.time() - t0:.0f} s; mean selection success {np.mean([p['selection_success'] for p in picks.values()]):.3f}"
    )


if __name__ == "__main__":
    main()
