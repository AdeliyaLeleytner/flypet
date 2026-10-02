#!/usr/bin/env python3
"""Scripted FlyTalk players on the same episodes (seeds) the language models get.

- lookup: smell every odour once, re-smell the most ambiguous ones with the remaining budget, then
  diagnose: label by the mean shift from the reference (+/-0.10) / erase: counter-condition each shifted
  odour once (reward if shifted down, punishment if shifted up), strongest shift first.
- nothing: diagnose answers all 0 / erase finishes at once.
- random: random actions, then a random answer / finish.
- known_history (erase only, cheating reference): reverses the hidden conditioning trial for trial.
"""

from __future__ import annotations
import argparse, json, pickle, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.flytalk import Env
from flypet.teach import FlyPool

OUT = ROOT / "data/flytalk_20260926"


def smell_all(env, ep, rounds_budget):
    shifts = {o: [] for o in env.panel}
    for o in env.panel:
        env.step(ep.eid, f"SMELL {o}")
        shifts[o].append(ep.transcript[-1]["value"] - env.naive[o])
    while ep.steps_left() > rounds_budget:
        o = min(env.panel, key=lambda x: abs(abs(np.mean(shifts[x])) - 0.10))
        env.step(ep.eid, f"SMELL {o}")
        shifts[o].append(ep.transcript[-1]["value"] - env.naive[o])
    return {o: float(np.mean(v)) for o, v in shifts.items()}


def play(env, player, task, seed):
    ep = env.reset(task, seed)
    rng = np.random.default_rng(seed + 999)
    if player == "nothing":
        env.step(
            ep.eid,
            "ANSWER\n" + "\n".join(f"{o}: 0" for o in env.panel)
            if task == "diagnose"
            else "FINISH",
        )
    elif player == "random":
        acts = (
            [f"SMELL {o}" for o in env.panel]
            + [f"TRAIN {o} + {r}" for o in env.panel for r in ("REWARD", "PUNISH")]
            + [f"EXPOSE {o}" for o in env.panel]
        )
        for _ in range(env.budget):
            env.step(ep.eid, acts[rng.integers(len(acts))])
            if ep.done:
                break
        if not ep.done:
            env.step(
                ep.eid,
                "ANSWER\n" + "\n".join(f"{o}: {rng.choice(['+', '-', '0'])}" for o in env.panel)
                if task == "diagnose"
                else "FINISH",
            )
    elif player == "known_history":
        for o, r, n in ep.history:
            for _ in range(n):
                env.step(ep.eid, f"TRAIN {o} + {'PUNISH' if r == 'reward' else 'REWARD'}")
        env.step(ep.eid, "FINISH")
    elif player == "lookup":
        if task == "diagnose":
            s = smell_all(env, ep, 0)
            env.step(
                ep.eid,
                "ANSWER\n"
                + "\n".join(
                    f"{o}: {'+' if v > 0.10 else '-' if v < -0.10 else '0'}" for o, v in s.items()
                ),
            )
        else:
            s = smell_all(env, ep, 3)
            for o in sorted(env.panel, key=lambda x: -abs(s[x])):
                if ep.steps_left() <= 0 or abs(s[o]) <= 0.10:
                    break
                env.step(ep.eid, f"TRAIN {o} + {'REWARD' if s[o] < 0 else 'PUNISH'}")
            if not ep.done:
                env.step(ep.eid, "FINISH")
    return {
        "player": player,
        "task": task,
        "seed": seed,
        "history": ep.history,
        **ep.result,
        "transcript": ep.transcript,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    d = pickle.load(open(ROOT / "data/teach_20260925/surrogate.pkl", "rb"))
    env = Env(FlyPool(a.workers), d["panel"], None)
    jobs = [
        (p, t, s)
        for t in ("diagnose", "erase")
        for p in ("lookup", "nothing", "random") + (("known_history",) if t == "erase" else ())
        for s in range(a.n)
    ]
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda j: play(env, *j), jobs))
    json.dump(rows, open(OUT / "baselines.json", "w"), indent=1, default=str)
    for t in ("diagnose", "erase"):
        print(f"== {t} (n={a.n} episodes each)")
        for p in ("lookup", "nothing", "random", "known_history"):
            sel = [r for r in rows if r["task"] == t and r["player"] == p]
            if not sel:
                continue
            extra = (
                f"conditioned right {np.nanmean([r['conditioned_right'] for r in sel]):.3f} untouched right {np.mean([r['untouched_right'] for r in sel]):.3f}"
                if t == "diagnose"
                else f"restored {np.nanmean([r['restored'] for r in sel]):.3f} kept {np.mean([r['kept'] for r in sel]):.3f}"
            )
            print(
                f"  {p:14s} score {np.mean([r['score'] for r in sel]):.3f}  exact {np.mean([r['exact'] for r in sel]):.3f}  "
                f"acc7 {np.mean([r['acc7'] for r in sel]):.3f}  {extra}  actions {np.mean([r['n_actions'] for r in sel]):.1f}"
            )


if __name__ == "__main__":
    main()
