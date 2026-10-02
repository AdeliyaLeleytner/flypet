#!/usr/bin/env python3
"""Collect a fresh panel; read its outcomes only after locking a reader checkpoint."""

import argparse, json, multiprocessing, sys, time, shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet.neural_records import write_json, file_hash
from scripts.latent import collect_mixed as mixed
from scripts.latent import collect_calibration as base


def run(spec):
    folder, runtime, profiles = mixed.WORKER
    runtime.reset(seed=spec["seed"], reset_memory=True)
    a, b, c = [profiles[n] for n in spec["odors"]]
    mixture = spec["mix_fraction"] * a + (1 - spec["mix_fraction"]) * b
    first = np.clip(spec["query_scale"] * mixture, 0, 300)
    second = np.clip(spec["second_scale"] * c, 0, 300)
    kind = spec["history_kind"]
    condition = np.zeros_like(first)
    other = np.zeros_like(first)
    learn = [False] * 6
    if kind != "naive":
        condition = np.clip(spec["training_scale"] * mixture, 0, 300)
        reinforcement = (
            runtime.memory.punishment(spec["punish_b_hz"])
            if kind == "punish"
            else runtime.memory.reward(spec["reward_a_hz"])
        )
        condition += runtime.drive_from_inputs([reinforcement])
        learn[1] = True
    if kind == "opposed":
        other = np.clip(spec["training_scale"] * c, 0, 300) + runtime.drive_from_inputs(
            [runtime.memory.punishment(spec["punish_b_hz"])]
        )
        learn[2] = True
    drives = np.stack([first, condition, other, np.zeros_like(first), first, second])
    durations = np.array(
        [
            250,
            250 * spec["training_repeats"],
            250 * spec["training_repeats"] if kind == "opposed" else 250,
            250,
            250,
            250,
        ],
        dtype=float,
    )
    neural, memory, metrics, walls = base.trajectory(runtime, drives, durations, learn)
    dest = folder / "episodes" / f"{spec['episode']:04d}"
    dest.mkdir(parents=True)
    np.savez_compressed(dest / "neural.npz", **neural)
    np.savez_compressed(dest / "memory.npz", **memory)
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
            "dt_ms": runtime.dt_ms,
            "initial_memory": "naive",
        },
    )
    return spec["episode"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--offset", type=int, default=10000)
    a = p.parse_args()
    if not 10000 <= a.offset <= 100000:
        raise ValueError("Panel offset outside the bounded range")
    a.out.mkdir(parents=True, exist_ok=False)
    protected = base.private_hashes()
    start = time.monotonic()
    recipes = []
    for i in range(64):
        spec = mixed.recipe(a.offset + i)
        family = i // 2
        rng = np.random.default_rng(983000 + a.offset + family)
        spec.update(
            episode=i,
            family=f"sealed-{family:03d}",
            split="sealed_final",
            history_kind=["naive", "reward", "punish", "opposed"][family % 4],
            training_repeats=1 + 2 * ((family // 4) % 2),
            reward_a_hz=float(rng.uniform(15, 75)),
            punish_b_hz=float(rng.uniform(15, 75)),
        )
        recipes.append(spec)
    write_json(
        a.out / "plan.json",
        {
            "status": "sealed_until_checkpoint_selection",
            "recipes": recipes,
            "role": "fresh recipe groups and noise seeds; familiar chemicals; one or three learning windows",
        },
    )
    for name in [
        "root_ids.npy",
        "input_root_ids.npy",
        "input_model_indices.npy",
        "memory_synapse_indices.npy",
    ]:
        shutil.copy2(Path("data/latent_runtime_20260927_v2") / name, a.out / name)
    sources = base.SOURCES + [
        "scripts/latent/collect_mixed.py",
        "scripts/latent/collect_sealed_reader_eval.py",
    ]
    for name in sources:
        dest = a.out / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(name, dest)
    with ProcessPoolExecutor(
        max_workers=4,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=mixed.init,
        initargs=(str(a.out),),
    ) as pool:
        for count, _ in enumerate(pool.map(run, recipes), 1):
            if count % 8 == 0:
                print(
                    json.dumps(
                        {"completed": count, "episodes": 64, "wall_s": time.monotonic() - start}
                    ),
                    flush=True,
                )
    if protected != base.private_hashes():
        raise RuntimeError("Personal memory changed")
    write_json(
        a.out / "manifest.json",
        {
            "status": "sealed_complete",
            "episodes": 64,
            "recipe_groups": 32,
            "seeds_per_recipe": 2,
            "scope": "fresh familiar-chemical mixtures and one/three learning windows; not unseen chemicals",
            "files": {
                str(p.relative_to(a.out)): file_hash(p) for p in a.out.rglob("*") if p.is_file()
            },
            "source_sha256": {name: file_hash(name) for name in sources},
            "pet_unchanged": True,
            "wall_s": time.monotonic() - start,
        },
    )


if __name__ == "__main__":
    main()
