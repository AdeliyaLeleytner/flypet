#!/usr/bin/env python3
"""Full-brain deterministic replay and evaluator-metadata isolation check."""

import argparse, json, multiprocessing, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet import latent_agent_env as env
from flypet.neural_records import array_hash, write_json


def check(_):
    task = env.task_spec(0)
    actions = [[0.5, 4, 0], [0.5, 0, 4]]
    first = env.WORKER.replay(task, actions)
    second = env.WORKER.replay({**task, "id": 1234, "target": -0.31}, actions)
    hashes = lambda r: {k: array_hash(r["features"][k]) for k in ("fine", "population")}
    return {
        "first": hashes(first),
        "second": hashes(second),
        "neural_equal": hashes(first) == hashes(second),
        "measured_equal": first["measured"] == second["measured"],
        "readout_equal": first["reward_readout"] == second["reward_readout"],
        "first_reward": float(first["reward"]),
        "second_reward": float(second["reward"]),
        "target_change_affects_reward": bool(first["reward"] != second["reward"]),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preprocessing", required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=2,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=env.initialize,
        initargs=(a.preprocessing,),
    ) as pool:
        rows = list(pool.map(check, [0, 1]))
    result = {
        "same_process_replay": all(
            r["neural_equal"] and r["measured_equal"] and r["readout_equal"] for r in rows
        ),
        "cross_process_replay": rows[0]["first"] == rows[1]["first"],
        "goal_metadata_only_changes_reward": all(r["target_change_affects_reward"] for r in rows),
        "rows": rows,
        "wall_s": time.monotonic() - start,
    }
    result["ok"] = all(
        result[k]
        for k in (
            "same_process_replay",
            "cross_process_replay",
            "goal_metadata_only_changes_reward",
        )
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.out, result)
    print(json.dumps(result))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
