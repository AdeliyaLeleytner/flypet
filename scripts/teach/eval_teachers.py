#!/usr/bin/env python3
"""Score language-model teachers on the real whole-brain fly, on the same 74 tasks as the oracle and baselines.

Input: JSON files {task name: {"actions": [[odour, reinforcement], ...], ...}}; output: teachers_real.json and a table
that also includes the oracle, textbook and random rows from oracle_baselines.json.
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.teach import FlyPool, score
from flypet.teach_tasks import FastSurrogate, make_tasks

DATA = ROOT / "data/teach_20260925"


def family(name, n_targets):
    return {1: "single", 2: "pair"}.get(n_targets[name], "triple")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("teachers", nargs="+", help="label=path.json")
    ap.add_argument("--workers", type=int, default=96)
    a = ap.parse_args(argv)
    sur = FastSurrogate(DATA / "surrogate.pkl")
    tasks = make_tasks(sur.panel, budget=sur.budget)
    n_targets = {t.name: sum(1 for v in t.targets.values() if v) for t in tasks}
    protos = {}
    for spec in a.teachers:
        label, path = spec.split("=", 1)
        if path == "minimal":
            d = {
                t.name: {
                    "actions": [
                        (o, "reward" if v > 0 else "punish") for o, v in t.targets.items() if v
                    ]
                }
                for t in tasks
            }
        else:
            d = json.load(open(path))
        for t in tasks:
            protos[(t.name, label)] = [tuple(x) for x in d.get(t.name, {}).get("actions", [])]
    pool = FlyPool(a.workers)
    real = pool.episodes(
        [
            (f"{k[0]}||{k[1]}", acts, sur.panel, sur.train_seed, sur.eval_seeds)
            for k, acts in protos.items()
        ],
        log_every=100,
    )
    pool.close()
    rows = []
    for (name, label), acts in protos.items():
        t = next(x for x in tasks if x.name == name)
        s, r, _ = score(t, real[f"{name}||{label}"]["valence"], sur.naive)
        rows.append(
            {
                "task": name,
                "teacher": label,
                "actions": acts,
                "n_trials": len(acts),
                "success_true": s,
                "reward_true": r,
                "success_sur": sur.evaluate(t, acts)[0],
                "valence_true": real[f"{name}||{label}"]["valence"],
            }
        )
    base = json.load(open(DATA / "oracle_baselines.json"))["rows"]
    prev = DATA / "teachers_real.json"
    if prev.exists():  # keep teachers scored in earlier runs
        base = base + [
            r for r in json.load(open(prev)) if r["teacher"] not in {k[1] for k in protos}
        ]
    names = {t.name for t in tasks}
    seen = set()
    for r in base:
        key = (r["task"], r["teacher"], json.dumps(r["actions"]))
        if r["task"] in names and key not in seen:
            seen.add(key)
            rows.append(r | {"n_trials": len(r["actions"])})
    json.dump(rows, open(DATA / "teachers_real.json", "w"), indent=1)
    print(
        f"{'teacher':16s} {'family':7s} {'n':>3s} {'success':>8s} {'solved':>7s} {'trials':>6s} {'sur':>6s}"
    )
    for label in sorted(
        {r["teacher"] for r in rows},
        key=lambda x: (
            ["oracle", "textbook", "random"].index(x)
            if x in ("oracle", "textbook", "random")
            else -1
        ),
    ):
        for f in ["single", "pair", "triple", "all"]:
            sel = [
                r
                for r in rows
                if r["teacher"] == label and (f == "all" or family(r["task"], n_targets) == f)
            ]
            print(
                f"{label:16s} {f:7s} {len(sel):3d} {np.mean([r['success_true'] for r in sel]):8.3f} "
                f"{np.mean([r['success_true'] == 1.0 for r in sel]):7.3f} {np.mean([r['n_trials'] for r in sel]):6.1f} "
                f"{np.mean([r['success_sur'] for r in sel]):6.3f}"
            )


if __name__ == "__main__":
    main()
