#!/usr/bin/env python3
"""One table for every FlyTalk diagnose player on the shared test flies (seeds 0-59), with bootstrap 95% intervals
over flies and paired differences against the lookup player on the seeds both played.

Inputs (whatever exists): data/flytalk_20260926/baselines.json, claude_<model>.json, bayes_player.json,
data/flytalk_rl_20260926/*.json (rows with "task", "seed", "score", "exact", ... and optionally "player").
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D1, D2 = ROOT / "data/flytalk_20260926", ROOT / "data/flytalk_rl_20260926"


def load():
    players = {}
    if (D1 / "baselines.json").exists():
        for r in json.load(open(D1 / "baselines.json")):
            players.setdefault(r["player"], []).append(r)
    for p in sorted(D1.glob("claude_*.json")):
        players[p.stem] = json.load(open(p))
    if (D1 / "lookup_prior.json").exists():
        players["lookup_prior"] = json.load(open(D1 / "lookup_prior.json"))
    if (D1 / "bayes_balanced.json").exists():
        players["bayes_balanced"] = json.load(open(D1 / "bayes_balanced.json"))
    if (D1 / "bayes_player.json").exists():
        players["bayes"] = [
            r | {"task": "diagnose"} for r in json.load(open(D1 / "bayes_player.json"))
        ]
    for p in sorted(D2.glob("*.json")) if D2.exists() else []:
        try:
            rows = json.load(open(p))
        except Exception:
            continue
        if (
            isinstance(rows, list)
            and rows
            and isinstance(rows[0], dict)
            and "seed" in rows[0]
            and "score" in rows[0]
        ):
            for r in rows:
                players.setdefault(r.get("player", p.stem), []).append(r)
    return players


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main(task="diagnose"):
    players = load()
    ref = {r["seed"]: r for r in players.get("lookup", []) if r["task"] == task}
    print(
        f"{task}: score [95% CI over flies], exact, trained right, untrained right, n; paired difference vs lookup"
    )
    for name, rows in sorted(
        players.items(),
        key=lambda kv: -np.mean([r["score"] for r in kv[1] if r.get("task", task) == task] or [0]),
    ):
        sel = [r for r in rows if r.get("task", task) == task and 0 <= r["seed"] < 60]
        if not sel:
            continue
        m, lo, hi = boot([r["score"] for r in sel])
        common = [r for r in sel if r["seed"] in ref]
        diff = (
            boot([r["score"] - ref[r["seed"]]["score"] for r in common])
            if common and name != "lookup"
            else None
        )
        tr = np.nanmean([r.get("conditioned_right", np.nan) for r in sel])
        ur = np.nanmean([r.get("untouched_right", np.nan) for r in sel])
        d = (
            f"  vs lookup {diff[0]:+.3f} [{diff[1]:+.3f}, {diff[2]:+.3f}] (n={len(common)})"
            if diff
            else ""
        )
        print(
            f"  {name:22s} {m:.3f} [{lo:.3f}, {hi:.3f}]  exact {np.mean([r['exact'] for r in sel]):.2f}  "
            f"trained {tr:.2f}  untrained {ur:.2f}  n={len(sel)}{d}"
        )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "diagnose")
