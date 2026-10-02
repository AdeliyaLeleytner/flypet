#!/usr/bin/env python3
"""Collect continuing full-brain episodes without LLM calls or personal memory.

    python scripts/latent/collect_calibration.py --output data/latent_calibration_v1 --episodes 256
    python scripts/latent/collect_calibration.py --verify data/latent_calibration_v1
    python scripts/latent/collect_calibration.py --replay data/latent_calibration_v1 --episode 0

This is calibration data, not a held-out benchmark. The neural arrays contain no
stimulus names, answer labels or history text. Drives, weights and teacher-side
metadata are separate files. Only a COMPLETE manifest is usable for training.
"""

from __future__ import annotations

import argparse
import atexit
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import json
import multiprocessing
import os
import platform
from pathlib import Path
import resource
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import brian2 as b
import numpy as np

from flypet import connectome, corrections
from flypet.engine import Brain
from flypet.mb import MushroomBody
from flypet.neural_runtime import NeuralRuntime
from flypet.neural_records import array_hash, file_hash, spike_arrays, write_json, compare_arrays
from flypet.odor import DoorOdor

ODORS = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
    "geosmin",
]
KINDS = ["naive", "reward_a", "punish_a", "opposed"]
SOURCES = [
    "flypet/engine.py",
    "flypet/neural_runtime.py",
    "flypet/neural_records.py",
    "flypet/mb.py",
    "flypet/odor.py",
    "flypet/connectome.py",
    "flypet/corrections.py",
    "scripts/latent/collect_calibration.py",
]
PRIVATE = ["data/memory.npz", "data/memory.npz.log.json", "data/pet_state.json"]


def private_hashes():
    return {name: file_hash(ROOT / name) if (ROOT / name).exists() else None for name in PRIVATE}


def build():
    start = time.perf_counter()
    brain = corrections.apply(Brain())
    memory = MushroomBody(brain)
    door = DoorOdor()
    inputs = [door.stim(name) for name in ODORS]
    ports = sorted(
        {int(i) for groups in inputs for group in groups for i in group.ids}
        | set(memory.pam_ids)
        | set(memory.ppl1_ids)
    )
    return brain, memory, door, ports, time.perf_counter() - start


def trajectory(runtime, drives, durations, learning):
    steps, weights, metrics, elapsed = [], [], [], []
    for drive, duration, learns in zip(drives, durations, learning, strict=True):
        runtime.set_learning(bool(learns))
        step = runtime.advance(drive, float(duration))
        steps.append(step)
        weights.append(runtime.memory.mem.copy())
        metrics.append(
            {
                "spikes": sum(step.result.counts.values()),
                "active_neurons": len(step.result.counts),
                "valence": runtime.memory.valence(step.result).score,
                "memory_strength": runtime.memory.memory_strength(),
                "learning_events": step.learning_events,
            }
        )
        elapsed.append(step.result.wall_s)
    indices, ticks = spike_arrays([s.result for s in steps], runtime.dt_ms)
    neural = {
        "spike_neuron_index": indices,
        "spike_tick": ticks,
        "phase_end_tick": np.cumsum(
            np.rint(np.asarray(durations) / runtime.dt_ms).astype(np.int64)
        ),
    }
    for key in ("voltage_mv", "synaptic_mv", "refractory_remaining_ms"):
        neural[key] = np.stack([s.observation[key] for s in steps])
    return neural, {"factors_after_phase": np.stack(weights)}, metrics, elapsed


def check_runtime(runtime, door):
    inputs = door.stim("ethyl acetate") + [runtime.memory.reward()]
    drive = runtime.drive_from_inputs(inputs)
    comparisons = []
    runtime.reset(seed=1207, reset_memory=True)
    runtime.set_learning(True)
    full = runtime.advance(drive, 750)
    expected = {
        "spike_neuron_index": spike_arrays([full.result], runtime.dt_ms)[0],
        "spike_tick": spike_arrays([full.result], runtime.dt_ms)[1],
        **{
            k: full.observation[k] for k in ("voltage_mv", "synaptic_mv", "refractory_remaining_ms")
        },
        "memory": runtime.memory.mem.copy(),
    }
    old_limit = runtime.monitor_limit
    for parts, limit in [([250, 250, 250], old_limit), ([73.2, 176.8, 25, 475], 1)]:
        runtime.reset(seed=1207, reset_memory=True)
        runtime.monitor_limit = limit
        steps = [runtime.advance(drive, ms_) for ms_ in parts]
        idx, ticks = spike_arrays([s.result for s in steps], runtime.dt_ms)
        actual = {
            "spike_neuron_index": idx,
            "spike_tick": ticks,
            **{
                k: steps[-1].observation[k]
                for k in ("voltage_mv", "synaptic_mv", "refractory_remaining_ms")
            },
            "memory": runtime.memory.mem.copy(),
        }
        comparisons.append(
            {"partition_ms": parts, "monitor_limit": limit, **compare_arrays(expected, actual)}
        )
    runtime.monitor_limit = old_limit
    if not all(x["ok"] for x in comparisons):
        raise RuntimeError("Full-brain partition check failed: " + json.dumps(comparisons))
    profile = []
    for duration in (10, 25, 50, 100, 250):
        runtime.reset(seed=1701, reset_memory=True)
        runtime.set_learning(False)
        values = [runtime.advance(drive, duration).result.wall_s for _ in range(3)]
        profile.append(
            {
                "chunk_ms": duration,
                "wall_s": values,
                "median_wall_s": float(np.median(values)),
                "wall_seconds_per_model_second": float(np.median(values) / (duration / 1000)),
            }
        )
    return {
        "ok": True,
        "scope": "one full-brain conditioned input, fixed PCG64 noise, same backend",
        "reference_spikes": len(expected["spike_tick"]),
        "partitions": comparisons,
        "profile": profile,
    }


def recipes(count):
    for index in range(count):
        family, replicate = index % 32, index // 32
        pair, kind = family // 4, KINDS[family % 4]
        yield {
            "episode": index,
            "family": f"pair{pair:02d}-{kind}",
            "replicate": replicate,
            "seed": 271000 + index * 997,
            "odor_a": ODORS[pair],
            "odor_b": ODORS[(pair + 1) % 8],
            "kind": kind,
            "role": "calibration_only_not_confirmatory_test",
        }


def make_inputs(runtime, door, recipe):
    a, b_ = door.stim(recipe["odor_a"]), door.stim(recipe["odor_b"])
    kind = recipe["kind"]
    if kind == "naive":
        train_a, train_b, learns = [], [], [False] * 6
    elif kind == "reward_a":
        train_a, train_b, learns = (
            a + [runtime.memory.reward()],
            [],
            [False, True, False, False, False, False],
        )
    elif kind == "punish_a":
        train_a, train_b, learns = (
            a + [runtime.memory.punishment()],
            [],
            [False, True, False, False, False, False],
        )
    else:
        train_a, train_b, learns = (
            a + [runtime.memory.reward()],
            b_ + [runtime.memory.punishment()],
            [False, True, True, False, False, False],
        )
    drives = np.stack([runtime.drive_from_inputs(x) for x in (a, train_a, train_b, [], a, b_)])
    return drives, np.full(6, 250.0), np.array(learns, dtype=bool)


def verify(path):
    manifest = json.loads((path / "manifest.json").read_text())
    errors = []
    if manifest.get("status") != "complete":
        errors.append("manifest is not complete")
    for name, digest in manifest.get("files", {}).items():
        target = (path / name).resolve()
        if (
            not target.is_relative_to(path.resolve())
            or not target.is_file()
            or file_hash(target) != digest
        ):
            errors.append(name)
    if len(list((path / "episodes").glob("*/episode.json"))) != manifest["episodes"]:
        errors.append("episode count")
    n_neurons = len(np.load(path / "root_ids.npy", allow_pickle=False))
    n_ports = len(np.load(path / "input_root_ids.npy", allow_pickle=False))
    n_synapses = len(np.load(path / "memory_synapse_indices.npy", allow_pickle=False))
    for index in range(manifest["episodes"]):
        dest = path / "episodes" / f"{index:04d}"
        try:
            with np.load(dest / "neural.npz", allow_pickle=False) as z:
                if set(z.files) != {
                    "spike_neuron_index",
                    "spike_tick",
                    "phase_end_tick",
                    "voltage_mv",
                    "synaptic_mv",
                    "refractory_remaining_ms",
                }:
                    raise ValueError("unexpected neural fields")
                for field in ("voltage_mv", "synaptic_mv", "refractory_remaining_ms"):
                    if z[field].shape != (6, n_neurons) or not np.isfinite(z[field]).all():
                        raise ValueError("invalid neural shape or values")
                ids, ticks = z["spike_neuron_index"], z["spike_tick"]
                ends = z["phase_end_tick"]
                if ends.shape != (6,) or not np.all(np.diff(ends) > 0):
                    raise ValueError("invalid phase boundaries")
                if (
                    ids.shape != ticks.shape
                    or np.any(ids < 0)
                    or np.any(ids >= n_neurons)
                    or np.any(ticks < 0)
                    or np.any(ticks >= ends[-1])
                    or np.any(np.diff(ticks) < 0)
                ):
                    raise ValueError("invalid sparse spikes")
            with np.load(dest / "memory.npz", allow_pickle=False) as z:
                values = z["factors_after_phase"]
                if (
                    values.shape != (6, n_synapses)
                    or not np.isfinite(values).all()
                    or np.any(values < 0.049999)
                    or np.any(values > 1)
                ):
                    raise ValueError("invalid memory")
            with np.load(dest / "inputs.npz", allow_pickle=False) as z:
                if (
                    z["drive_hz"].shape != (6, n_ports)
                    or not np.isfinite(z["drive_hz"]).all()
                    or np.any(z["drive_hz"] < 0)
                    or np.any(z["drive_hz"] > 300)
                ):
                    raise ValueError("invalid drives")
        except (OSError, KeyError, ValueError) as exc:
            errors.append(f"episode {index}: {exc}")
    return {"ok": not errors, "files_checked": len(manifest.get("files", {})), "errors": errors}


def save_episode(path, runtime, door, recipe):
    runtime.reset(seed=recipe["seed"], reset_memory=True)
    drives, durations, learning = make_inputs(runtime, door, recipe)
    neural, weights, metrics, times = trajectory(runtime, drives, durations, learning)
    dest = path / "episodes" / f"{recipe['episode']:04d}"
    dest.mkdir(parents=True)
    np.savez_compressed(dest / "neural.npz", **neural)
    np.savez_compressed(dest / "memory.npz", **weights)
    np.savez_compressed(
        dest / "inputs.npz", drive_hz=drives, duration_ms=durations, learning=learning
    )
    write_json(
        dest / "episode.json",
        {
            **recipe,
            "metrics": metrics,
            "wall_s": times,
            "initial_memory": "naive",
            "dt_ms": runtime.dt_ms,
            "neural_array_sha256": {k: array_hash(v) for k, v in neural.items()},
        },
    )
    return {
        "times": times,
        "worker_pid": os.getpid(),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
    }


_WORKER = None


def init_worker(path):
    global _WORKER
    brain, memory, door, ports, _ = build()
    runtime = NeuralRuntime(brain, seed=1, input_ids=ports, memory=memory)
    _WORKER = (Path(path), runtime, door)
    atexit.register(runtime.close)


def worker_episode(recipe):
    path, runtime, door = _WORKER
    return save_episode(path, runtime, door, recipe)


def collect(path, count, workers=1):
    if not 1 <= count <= 256:
        raise ValueError("Calibration collection is bounded to 1..256 episodes")
    if not 1 <= workers <= 4:
        raise ValueError("Local collection is bounded to 1..4 workers")
    if path.exists():
        raise FileExistsError("Use a new output directory")
    path.mkdir(parents=True)
    protected = private_hashes()
    specs = list(recipes(count))
    sources = {name: file_hash(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        dest = path / "source_snapshot" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    write_json(
        path / "plan.json",
        {
            "schema_version": 1,
            "episodes": specs,
            "role": "calibration_only",
            "source_sha256": sources,
            "phases": [
                "baseline_a",
                "conditioning_a",
                "conditioning_b",
                "rest",
                "probe_a",
                "probe_b",
            ],
        },
    )
    start = time.perf_counter()
    brain, memory, door, ports, build_s = build()
    np.save(path / "root_ids.npy", connectome.neuron_order()[0])
    np.save(path / "memory_synapse_indices.npy", memory.syn_idx)
    walls, worker_rss = [], {}

    def progress(record, completed):
        walls.extend(record["times"])
        worker_rss[str(record["worker_pid"])] = record["peak_rss_bytes"]
        if completed % 8 == 0 or completed == count:
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "total": count,
                        "elapsed_s": round(time.perf_counter() - start, 1),
                    }
                ),
                flush=True,
            )

    with NeuralRuntime(brain, seed=1, input_ids=ports, memory=memory) as runtime:
        np.save(path / "input_root_ids.npy", runtime.root_ids)
        np.save(path / "input_model_indices.npy", runtime.ports)
        checks = check_runtime(runtime, door)
        write_json(path / "runtime_checks.json", checks)
        print(
            json.dumps(
                {
                    "runtime_checks": "passed",
                    "neurons": brain.n,
                    "input_ports": len(runtime.ports),
                    "median_250ms_wall_s": checks["profile"][-1]["median_wall_s"],
                }
            ),
            flush=True,
        )
        if workers == 1:
            for completed, recipe in enumerate(specs, 1):
                progress(save_episode(path, runtime, door, recipe), completed)
    if workers > 1:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=init_worker,
            initargs=(str(path),),
        ) as pool:
            for completed, record in enumerate(pool.map(worker_episode, specs), 1):
                progress(record, completed)
    unchanged = protected == private_hashes()
    if not unchanged:
        raise RuntimeError("Protected pet files changed during collection")
    input_sources = {
        str(p.relative_to(ROOT)): file_hash(p)
        for p in [
            connectome.PATH_CON,
            connectome.PATH_COMP,
            connectome.PATH_ANN,
            ROOT / "data/door/door_response_matrix.csv",
            ROOT / "data/door/door_mappings.csv",
            ROOT / "data/door/odor.csv",
        ]
    }
    receipt = {
        "status": "complete",
        "schema_version": 1,
        "episodes": count,
        "phases_per_episode": 6,
        "model_seconds": count * 1.5,
        "build_s": build_s,
        "total_wall_s": time.perf_counter() - start,
        "trial_median_wall_s": float(np.median(walls)),
        "trial_p95_wall_s": float(np.percentile(walls, 95)),
        "peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        "workers": workers,
        "worker_peak_rss_bytes": worker_rss,
        "compute_spend_usd": 0,
        "execution": "local_cpu",
        "pet_unchanged": unchanged,
        "protected_sha256": protected,
        "source_sha256": sources,
        "input_source_sha256": input_sources,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "brian2": b.__version__,
            "numpy": np.__version__,
            "codegen_target": b.prefs.codegen.target,
            "actual_codegen": type(brain.neu.state_updater.codeobj).__name__,
            "dt_ms": float(brain.neu.clock.dt / b.ms),
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rng": "private PCG64, Bernoulli per dt, time-major fixed input-port order",
        "runtime_parameters": {
            "learning_window_ms": 250.0,
            "max_rate_hz": 300.0,
            "max_step_ms": 1000.0,
            "input_refractory_ms": 0.0,
            "refractory_policy": "fixed on the entire declared input mask for the episode",
        },
        "memory_parameters": {
            key: float(getattr(memory, key))
            for key in ("eta", "floor", "kc_scale_hz", "dan_scale_hz", "dan_threshold")
        },
        "neural_includes_writable_input_ports": True,
        "reader_mask_file": "input_model_indices.npy",
        "input_root_ids_sha256": array_hash(np.load(path / "input_root_ids.npy")),
        "root_ids_sha256": array_hash(np.load(path / "root_ids.npy")),
        "features_are_restart_checkpoints": False,
    }
    files = sorted(p for p in path.rglob("*") if p.is_file())
    receipt["files"] = {str(p.relative_to(path)): file_hash(p) for p in files}
    receipt["artifact_bytes"] = sum(p.stat().st_size for p in files)
    write_json(path / "manifest.json", receipt)
    verification = verify(path)
    write_json(path / "verification.json", verification)
    if not verification["ok"]:
        raise RuntimeError(str(verification))
    print(
        json.dumps(
            {
                k: receipt[k]
                for k in (
                    "status",
                    "episodes",
                    "total_wall_s",
                    "trial_median_wall_s",
                    "trial_p95_wall_s",
                    "artifact_bytes",
                    "pet_unchanged",
                )
            }
        ),
        flush=True,
    )


def replay(path, episode):
    checks = verify(path)
    if not checks["ok"]:
        raise RuntimeError(str(checks))
    manifest = json.loads((path / "manifest.json").read_text())
    expected_environment = manifest["environment"]
    current_environment = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "brian2": b.__version__,
        "codegen_target": b.prefs.codegen.target,
        "dt_ms": float(b.defaultclock.dt / b.ms),
    }
    for key, value in current_environment.items():
        if expected_environment[key] != value:
            raise RuntimeError(
                f"Replay environment differs for {key}: {value} != {expected_environment[key]}"
            )
    for name, digest in {**manifest["source_sha256"], **manifest["input_source_sha256"]}.items():
        if file_hash(ROOT / name) != digest:
            raise RuntimeError("Replay source differs: " + name)
    protected = private_hashes()
    dest = path / "episodes" / f"{episode:04d}"
    spec = json.loads((dest / "episode.json").read_text())
    with np.load(dest / "inputs.npz", allow_pickle=False) as z:
        drives, durations, learning = z["drive_hz"], z["duration_ms"], z["learning"]
    brain, memory, door, _, _ = build()
    ports = np.load(path / "input_root_ids.npy").tolist()
    with NeuralRuntime(brain, seed=spec["seed"], input_ids=ports, memory=memory) as runtime:
        if not np.array_equal(runtime.root_ids, np.load(path / "input_root_ids.npy")):
            raise RuntimeError("Replay input-port order differs")
        neural, weights, _, _ = trajectory(runtime, drives, durations, learning)
        if type(brain.neu.state_updater.codeobj).__name__ != expected_environment["actual_codegen"]:
            raise RuntimeError("Replay actual codegen backend differs")
    with np.load(dest / "neural.npz", allow_pickle=False) as z:
        neural_check = compare_arrays(dict(z), neural)
    with np.load(dest / "memory.npz", allow_pickle=False) as z:
        memory_check = compare_arrays(dict(z), weights)
    receipt = {
        "episode": episode,
        "neural": neural_check,
        "memory": memory_check,
        "pet_unchanged": protected == private_hashes(),
    }
    receipt["ok"] = neural_check["ok"] and memory_check["ok"] and receipt["pet_unchanged"]
    write_json(path / f"replay-{episode:04d}.json", receipt)
    print(json.dumps(receipt), flush=True)
    if not receipt["ok"]:
        raise RuntimeError("Replay differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--verify", type=Path)
    group.add_argument("--replay", type=Path)
    parser.add_argument("--episodes", type=int, default=256)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--episode", type=int, default=0)
    args = parser.parse_args()
    if args.verify:
        result = verify(args.verify)
        print(json.dumps(result))
        if not result["ok"]:
            raise SystemExit(1)
    elif args.replay:
        replay(args.replay, args.episode)
    else:
        collect(args.output, args.episodes, args.workers)


if __name__ == "__main__":
    main()
