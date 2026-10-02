#!/usr/bin/env python3
"""A stronger reference teacher: the surrogate proposes its 16 best protocols per task, the real fly picks one
on selection seeds (training 7000+, probe 301), and the pick is scored on the standard test seeds (5000, 101-103)."""

from __future__ import annotations
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.teach import FlyPool, score
from flypet.teach_tasks import FastSurrogate, beam_topk, make_tasks

DATA = ROOT / "data/teach_20260925"
_S = {}


def _init():
    _S["sur"] = FastSurrogate(DATA / "surrogate.pkl")


def _props(task):
    return task.name, [p for p, _, _ in beam_topk(_S["sur"], task, width=48, k=16)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args(argv)
    t0 = time.time()
    sur = FastSurrogate(DATA / "surrogate.pkl")
    tasks = make_tasks(sur.panel, budget=sur.budget)
    with mp.get_context("fork").Pool(min(a.workers, len(tasks)), initializer=_init) as pool:
        props = dict(pool.imap_unordered(_props, tasks))
    print(f"proposals ready in {time.time() - t0:.0f} s", flush=True)
    fly = FlyPool(a.workers)
    naive_sel = fly.episodes([("naive", [], sur.panel, 7000, (301,))])["naive"]["valence"]
    jobs = [
        (f"{t.name}||{i}", p, sur.panel, 7000, (301,))
        for t in tasks
        for i, p in enumerate(props[t.name])
    ]
    sel = fly.episodes(jobs, log_every=200)
    picks = {}
    for t in tasks:
        scored = [
            (score(t, sel[f"{t.name}||{i}"]["valence"], naive_sel)[:2], i)
            for i in range(len(props[t.name]))
        ]
        (s, r), i = max(scored, key=lambda x: (x[0][0], x[0][1]))
        picks[t.name] = {
            "actions": props[t.name][i],
            "selection_success": s,
            "rank_in_surrogate": i,
        }
    fly.close()
    json.dump(picks, open(DATA / "teacher_oracle_rerank.json", "w"), indent=1)
    print(
        f"selection done in {time.time() - t0:.0f} s; mean selection success {np.mean([p['selection_success'] for p in picks.values()]):.3f}"
    )


if __name__ == "__main__":
    main()
