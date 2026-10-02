#!/usr/bin/env python3
"""Fresh continuous mixtures/intensities on the same declared neural ports.

Complements the fixed calibration families. The split is fixed from recipe
groups before simulation. All eight base odors are familiar; this evaluates new
mixtures and continuous exposures, not unseen chemical generalization.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import brian2 as b
import numpy as np
from flypet.neural_runtime import NeuralRuntime
from flypet.neural_records import file_hash, write_json, array_hash
from scripts.latent import collect_calibration as base

WORKER = None


def init(path):
    global WORKER
    brain, memory, door, ports, _ = base.build()
    runtime = NeuralRuntime(brain, seed=1, input_ids=ports, memory=memory)
    profiles = {name: runtime.drive_from_inputs(door.stim(name)) for name in base.ODORS}
    WORKER = (Path(path), runtime, profiles)
    atexit.register(runtime.close)


def recipe(index):
    # Two stochastic repeats of each full exposure recipe remain in one split.
    group = index // 2
    rng = np.random.default_rng(940000 + group)
    odors = rng.choice(base.ODORS, 3, replace=False).tolist()
    bucket = group % 8
    partition = "validation" if bucket == 6 else "development_test" if bucket == 7 else "train"
    return {
        "episode": index,
        "family": f"mixture-{group:04d}",
        "split": partition,
        "seed": 950000 + index * 811,
        "odors": odors,
        "mix_fraction": float(rng.uniform(0.15, 0.85)),
        "query_scale": float(rng.uniform(0.35, 1.5)),
        "second_scale": float(rng.uniform(0.35, 1.5)),
        "training_scale": float(rng.uniform(0.4, 1.4)),
        "reward_a_hz": float(rng.choice([0.0, 20.0, 40.0, 60.0, 90.0])),
        "punish_b_hz": float(rng.choice([0.0, 20.0, 40.0, 60.0, 90.0])),
        "history_kind": str(rng.choice(["naive", "reward", "punish", "opposed"])),
        "role": "new_mixtures_and_intensities_of_familiar_odorants",
    }


def run(spec):
    path, runtime, profiles = WORKER
    runtime.reset(seed=spec["seed"], reset_memory=True)
    a, bb, c = [profiles[name] for name in spec["odors"]]
    mix = spec["mix_fraction"] * a + (1 - spec["mix_fraction"]) * bb
    qa = np.clip(spec["query_scale"] * mix, 0, 300)
    qb = np.clip(spec["second_scale"] * c, 0, 300)
    kind = spec["history_kind"]
    learn = [False] * 6
    ta = np.zeros_like(qa)
    tb = np.zeros_like(qa)
    if kind != "naive":
        ta = np.clip(spec["training_scale"] * mix, 0, 300)
        reinforcement = (
            runtime.memory.punishment(spec["punish_b_hz"])
            if kind == "punish"
            else runtime.memory.reward(spec["reward_a_hz"])
        )
        ta += runtime.drive_from_inputs([reinforcement])
        learn[1] = True
    if kind == "opposed":
        tb = np.clip(spec["training_scale"] * c, 0, 300) + runtime.drive_from_inputs(
            [runtime.memory.punishment(spec["punish_b_hz"])]
        )
        learn[2] = True
    drives = np.stack([qa, ta, tb, np.zeros_like(qa), qa, qb])
    durations = np.full(6, 250.0)
    neural, weights, metrics, walls = base.trajectory(runtime, drives, durations, learn)
    dest = path / "episodes" / f"{spec['episode']:04d}"
    dest.mkdir(parents=True)
    np.savez_compressed(dest / "neural.npz", **neural)
    np.savez_compressed(dest / "memory.npz", **weights)
    np.savez_compressed(
        dest / "inputs.npz",
        drive_hz=drives,
        duration_ms=durations,
        learning=np.array(learn, dtype=bool),
    )
    write_json(
        dest / "episode.json",
        {
            **spec,
            "metrics": metrics,
            "wall_s": walls,
            "initial_memory": "naive",
            "dt_ms": runtime.dt_ms,
            "neural_array_sha256": {k: array_hash(v) for k, v in neural.items()},
        },
    )
    return walls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episodes", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if not 1 <= a.workers <= 4 or not 1 <= a.episodes <= 2048:
        raise ValueError("Local run exceeds bounds")
    a.output.mkdir(parents=True, exist_ok=False)
    before = base.private_hashes()
    start = time.monotonic()
    recipes = [recipe(i) for i in range(a.episodes)]
    write_json(a.output / "plan.json", {"schema_version": 1, "scope": __doc__, "episodes": recipes})
    original = ROOT / "data/latent_runtime_20260927_v2"
    for name in [
        "root_ids.npy",
        "input_root_ids.npy",
        "input_model_indices.npy",
        "memory_synapse_indices.npy",
    ]:
        shutil.copy2(original / name, a.output / name)
    sources = {
        name: file_hash(ROOT / name) for name in base.SOURCES + ["scripts/latent/collect_mixed.py"]
    }
    for name in sources:
        dest = a.output / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    walls = []
    with ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init,
        initargs=(str(a.output),),
    ) as pool:
        for completed, times in enumerate(pool.map(run, recipes), 1):
            walls.extend(times)
            if completed % 16 == 0 or completed == a.episodes:
                print(
                    json.dumps(
                        {
                            "completed": completed,
                            "total": a.episodes,
                            "elapsed_s": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
    after = base.private_hashes()
    if before != after:
        raise RuntimeError("Personal pet files changed")
    parent = json.loads((original / "manifest.json").read_text())
    files = {str(p.relative_to(a.output)): file_hash(p) for p in a.output.rglob("*") if p.is_file()}
    report = {
        "status": "complete",
        "schema_version": 1,
        "episodes": a.episodes,
        "phases_per_episode": 6,
        "scope": "new mixtures and intensities; familiar eight odorants; pre-fixed recipe-group split",
        "source_sha256": sources,
        "input_source_sha256": parent["input_source_sha256"],
        "runtime_parameters": parent["runtime_parameters"],
        "memory_parameters": parent["memory_parameters"],
        "environment": parent["environment"],
        "files": files,
        "pet_unchanged": True,
        "total_wall_s": time.monotonic() - start,
        "phase_median_wall_s": float(np.median(walls)),
        "workers": a.workers,
        "new_gpu_spend_usd": 0,
        "neural_includes_writable_input_ports": True,
    }
    report["environment"]["codegen_target"] = b.prefs.codegen.target
    write_json(a.output / "manifest.json", report)
    verification = base.verify(a.output)
    write_json(a.output / "verification.json", verification)
    if not verification["ok"]:
        raise RuntimeError(str(verification))
    print(
        json.dumps(
            {
                "status": "complete",
                "episodes": a.episodes,
                "verified_files": verification["files_checked"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
