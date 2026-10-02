#!/usr/bin/env python3
"""Bounded training-only dose/noise check before continuous-policy fitting."""

import argparse, json, multiprocessing, time, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet import latent_agent_env as environment
from flypet.neural_records import write_json, file_hash
from scripts.latent.collect_calibration import private_hashes


def run(job):
    task = environment.task_spec(job["task"])
    task["seed"] = job["seed"]
    env = environment.WORKER
    t = time.monotonic()
    initial = env.reset(task)
    after = env.step(job["action"])
    probes = [after["measured"]]
    for _ in range(2):
        probes.append(env.observation(env.advance(env.probe, False))["measured"])
    ap = float(np.mean([m["approach_hz"] for m in probes]))
    av = float(np.mean([m["avoid_hz"] for m in probes]))
    return {
        **job,
        "initial": initial["measured"],
        "probes": probes,
        "pooled_valence": (ap - av) / (ap + av + 1),
        "mean_probe_valence": float(np.mean([m["valence"] for m in probes])),
        "pooled_approach_hz": ap,
        "pooled_avoid_hz": av,
        "wall_s": time.monotonic() - t,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocessing", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    protected = private_hashes()
    start = time.monotonic()
    jobs = []
    for task in (0, 1):
        actions = [(1.0, float(dose), 0.0) for dose in (0, 2, 4, 8, 16, 30, 60)] + [
            (1.0, 0.0, float(dose)) for dose in (2, 4, 8, 16, 30, 60)
        ]
        actions += (
            [(float(scale), 12.0, 0.0) for scale in (0, 0.25, 0.5, 1, 1.5)]
            + [(float(scale), 0.0, 12.0) for scale in (0, 0.25, 0.5, 1, 1.5)]
            + [(0.0, 0.0, 0.0)]
        )
        for seed_index in range(3):
            seed = environment.task_spec(task)["seed"] + seed_index * 10000000
            for action in actions:
                jobs.append({"task": task, "seed": seed, "action": list(action)})
    write_json(
        a.out / "plan.json",
        {
            "jobs": jobs,
            "scope": "two training recipes only; shared seeds across actions; three consecutive non-learning probes",
            "source_sha256": {
                n: file_hash(n)
                for n in ["flypet/latent_agent_env.py", "scripts/latent/calibrate_agent.py"]
            },
        },
    )
    rows = []
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=environment.initialize,
        initargs=(str(a.preprocessing),),
    ) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            if len(rows) % 12 == 0:
                print(
                    json.dumps(
                        {
                            "completed": len(rows),
                            "total": len(jobs),
                            "wall_s": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
    summary = []
    for task in (0, 1):
        for action in sorted({tuple(r["action"]) for r in rows if r["task"] == task}):
            group = [r for r in rows if r["task"] == task and tuple(r["action"]) == action]
            summary.append(
                {
                    "task": task,
                    "action": action,
                    "first_mean": float(np.mean([r["probes"][0]["valence"] for r in group])),
                    "first_std": float(np.std([r["probes"][0]["valence"] for r in group], ddof=1)),
                    "pooled_mean": float(np.mean([r["pooled_valence"] for r in group])),
                    "pooled_std": float(np.std([r["pooled_valence"] for r in group], ddof=1)),
                }
            )
    if protected != private_hashes():
        raise RuntimeError("Personal memory changed")
    write_json(a.out / "episodes.json", rows)
    write_json(
        a.out / "report.json",
        {
            "status": "complete",
            "rows": summary,
            "episodes": len(rows),
            "wall_s": time.monotonic() - start,
            "pet_unchanged": True,
        },
    )


if __name__ == "__main__":
    main()
