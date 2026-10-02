#!/usr/bin/env python3
"""How much is there for a teacher to learn? Beam-search oracle on the surrogate vs textbook and random teachers,
all scored on the real whole-brain fly."""

from __future__ import annotations
import argparse, json, multiprocessing as mp, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.teach import FlyPool, score
from flypet.teach_tasks import FastSurrogate, beam_search, make_tasks, random_protocol, textbook

DATA = ROOT / "data/teach_20260925"
_S = {}


def _init():
    _S["sur"] = FastSurrogate(DATA / "surrogate.pkl")


def _search(task):
    return task.name, beam_search(_S["sur"], task, width=48)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--n-random", type=int, default=3)
    a = ap.parse_args(argv)
    t0 = time.time()
    sur = FastSurrogate(DATA / "surrogate.pkl")
    tasks = make_tasks(sur.panel, budget=sur.budget)
    print(f"{len(tasks)} tasks, panel {sur.panel}", flush=True)
    with mp.get_context("fork").Pool(a.workers, initializer=_init) as pool:
        oracle = dict(pool.imap_unordered(_search, tasks))
    print(f"beam search done in {time.time() - t0:.0f} s", flush=True)
    rng = np.random.default_rng(1)
    protos = {}
    for t in tasks:
        protos[(t.name, "oracle")] = oracle[t.name][0]
        protos[(t.name, "textbook")] = textbook(t)
        for i in range(a.n_random):
            protos[(t.name, f"random{i}")] = random_protocol(t, rng)
    pool = FlyPool(a.workers)
    jobs = [
        (f"{k[0]}||{k[1]}", acts, sur.panel, sur.train_seed, sur.eval_seeds)
        for k, acts in protos.items()
    ]
    real = pool.episodes(jobs, log_every=50)
    pool.close()
    rows = []
    for t in tasks:
        for kind in ["oracle", "textbook"] + [f"random{i}" for i in range(a.n_random)]:
            acts = protos[(t.name, kind)]
            s_true, r_true, _ = score(t, real[f"{t.name}||{kind}"]["valence"], sur.naive)
            s_sur, r_sur = sur.evaluate(t, acts)
            rows.append(
                {
                    "task": t.name,
                    "targets": t.targets,
                    "teacher": kind.rstrip("0123456789"),
                    "actions": acts,
                    "success_true": s_true,
                    "reward_true": r_true,
                    "success_sur": s_sur,
                    "reward_sur": r_sur,
                    "valence_true": real[f"{t.name}||{kind}"]["valence"],
                }
            )
    json.dump(
        {
            "panel": sur.panel,
            "naive": sur.naive,
            "rows": rows,
            "wall_s": round(time.time() - t0, 1),
        },
        open(DATA / "oracle_baselines.json", "w"),
        indent=1,
    )
    fam = lambda name: (
        "single" if name.count(" ") == 0 else ("pair" if name.count(" ") == 1 else "triple")
    )
    print(
        f"\n{'teacher':9s} {'family':7s} {'success(real)':>13s} {'reward(real)':>12s} {'solved':>7s} {'success(sur)':>12s}"
    )
    for teacher in ["oracle", "textbook", "random"]:
        for f in ["single", "pair", "triple", "all"]:
            sel = [
                r for r in rows if r["teacher"] == teacher and (f == "all" or fam(r["task"]) == f)
            ]
            print(
                f"{teacher:9s} {f:7s} {np.mean([r['success_true'] for r in sel]):13.3f} {np.mean([r['reward_true'] for r in sel]):12.3f} "
                f"{np.mean([r['success_true'] == 1.0 for r in sel]):7.3f} {np.mean([r['success_sur'] for r in sel]):12.3f}"
            )
    print(f"wall {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
