#!/usr/bin/env python3
"""Collect greedy simulator demonstrations on training/validation tasks only."""

import argparse, json, multiprocessing, time, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet.latent_agent_env import task_spec, initialize, teacher_episode, TEACHER_ACTIONS
from flypet.neural_records import write_json, file_hash
from scripts.latent.collect_calibration import private_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocessing", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tasks", type=int, default=40)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.tasks <= 40 or not 1 <= a.workers <= 16:
        raise ValueError("Pilot task/worker bound exceeded")
    a.out.mkdir(parents=True, exist_ok=False)
    before = private_hashes()
    start = time.monotonic()
    tasks = [task_spec(i) for i in range(a.tasks)]
    write_json(
        a.out / "plan.json",
        {
            "tasks": tasks,
            "teacher_actions": TEACHER_ACTIONS.tolist(),
            "method": "greedy full-simulator branching, mean of 3 future-noise samples conditional on the same observed state",
            "test_tasks_not_simulated": True,
            "preprocessing_sha256": file_hash(a.preprocessing),
        },
    )
    rows = []
    with ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(str(a.preprocessing),),
    ) as pool:
        for result in pool.map(teacher_episode, tasks):
            task = result["task"]
            dest = a.out / f"task-{task['id']:03d}"
            dest.mkdir()
            arrays = {}
            metadata = []
            for turn, example in enumerate(result["examples"]):
                obs = example.pop("observation")
                for group in ("features", "reference"):
                    for name in ("fine", "population"):
                        arrays[f"{turn}_{group}_{name}"] = obs[group][name]
                metadata.append(
                    {
                        **example,
                        "measured": obs["measured"],
                        "reward_readout": obs["reward_readout"],
                        "turn": turn,
                    }
                )
            np.savez_compressed(dest / "features.npz", **arrays)
            row = {
                "task": task,
                "examples": metadata,
                "final": result["final"],
                "final_reward": result["final_reward"],
            }
            write_json(dest / "episode.json", row)
            rows.append(row)
            print(
                json.dumps(
                    {"completed": len(rows), "tasks": a.tasks, "wall_s": time.monotonic() - start}
                ),
                flush=True,
            )
    if before != private_hashes():
        raise RuntimeError("Personal memory changed")
    write_json(
        a.out / "manifest.json",
        {
            "status": "complete",
            "tasks": len(rows),
            "examples": 2 * len(rows),
            "wall_s": time.monotonic() - start,
            "source_sha256": {
                n: file_hash(n)
                for n in ["flypet/latent_agent_env.py", "scripts/latent/collect_agent_teacher.py"]
            },
            "files": {
                str(p.relative_to(a.out)): file_hash(p) for p in a.out.rglob("*") if p.is_file()
            },
            "pet_unchanged": True,
        },
    )


if __name__ == "__main__":
    main()
