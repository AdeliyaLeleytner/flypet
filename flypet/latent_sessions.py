"""Pinned brain workers for live latent interactions; independent visitor memory."""

from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import secrets
import time
import numpy as np

STATE = None


def _initialize(bundle, seed):
    global STATE
    from .engine import Brain
    from . import corrections
    from .mb import MushroomBody
    from .neural_runtime import NeuralRuntime

    folder = Path(bundle)
    with np.load(folder / "writer_preprocessing.npz", allow_pickle=False) as z:
        ports = z["input_root_ids"]
    with np.load(folder / "reader_preprocessing.npz", allow_pickle=False) as z:
        pre = {k: z[k] for k in z.files}
    brain = corrections.apply(Brain())
    memory = MushroomBody(brain)
    selected = np.array([brain.i2flyid[int(i)] for i in pre["neuron_indices"]])
    if not np.array_equal(selected, pre["neuron_root_ids"]):
        raise ValueError("Reader anatomical order differs")
    runtime = NeuralRuntime(brain, input_ids=ports.tolist(), seed=seed, memory=memory)
    if not np.array_equal(runtime.root_ids, ports):
        raise ValueError("Writer input order differs")
    STATE = {
        "runtime": runtime,
        "memory": memory,
        "pre": pre,
        "reference": None,
        "current": None,
        "telemetry": None,
        "steps": 0,
    }


def _features(step):
    from .latent_observation import from_step
    from .latent_reader_data import read_phase
    from .neural_records import spike_arrays

    pre = STATE["pre"]
    runtime = STATE["runtime"]
    if "population_ids" in pre:
        return from_step(step, pre, runtime.dt_ms)
    ids, ticks = spike_arrays([step.result], runtime.dt_ms)
    arrays = {
        "spike_neuron_index": ids,
        "spike_tick": ticks,
        "phase_end_tick": np.array([round(step.result.duration_s * 1000 / runtime.dt_ms)]),
    }
    for key in ["voltage_mv", "synaptic_mv", "refractory_remaining_ms"]:
        arrays[key] = step.observation[key][None]
    fine = read_phase(arrays, 0, pre["neuron_indices"], dt_ms=runtime.dt_ms)
    return {
        "fine": np.clip((fine - pre["mean"]) / pre["std"], -10, 10).astype(np.float32),
        "spike_bins": np.bincount(
            np.minimum(4, ticks * 5 // arrays["phase_end_tick"][0]), minlength=5
        ),
    }


def _step(drive, learning):
    runtime = STATE["runtime"]
    runtime.set_learning(learning)
    step = runtime.advance(drive, 250)
    features = _features(step)
    if STATE["reference"] is None:
        STATE["reference"] = features
    STATE["current"] = features
    STATE["steps"] += 1
    val = STATE["memory"].valence(step.result)
    telemetry = {
        "approach_hz": val.approach_hz,
        "avoid_hz": val.avoid_hz,
        "valence": val.score,
        "active_neurons": len(step.result.counts),
        "spikes": sum(step.result.counts.values()),
        "memory_strength": STATE["memory"].memory_strength(),
        "episode_time_ms": runtime.tick * runtime.dt_ms,
        "simulation_wall_s": step.result.wall_s,
        "seed": runtime.seed,
        "steps": STATE["steps"],
        "spike_bins": features.get("all_spike_bins", features["spike_bins"]).tolist(),
    }
    if "population_rates_hz" in features:
        telemetry["population_rates_hz"] = features["population_rates_hz"].tolist()
        telemetry["downstream_spike_bins"] = features["spike_bins"].tolist()
    STATE["telemetry"] = telemetry
    return {"current": features, "reference": STATE["reference"], "telemetry": telemetry}


def _observe():
    if STATE["current"] is None:
        return _step(np.zeros(len(STATE["runtime"].ports)), False)
    return {key: STATE[key] for key in ["current", "reference", "telemetry"]}


def _pin_reference():
    _observe()
    STATE["reference"] = STATE["current"]
    return _observe()


class BrainSession:
    def __init__(self, bundle, seed=None):
        self.token = secrets.token_urlsafe(24)
        self.seed = secrets.randbits(32) if seed is None else int(seed)
        self.touched = time.monotonic()
        self.history = []
        self.has_observation = False
        self.pool = ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
            initargs=(str(Path(bundle).resolve()), self.seed),
        )

    def step(self, drive, learning=True):
        self.touched = time.monotonic()
        result = self.pool.submit(
            _step, np.asarray(drive, dtype=np.float64), bool(learning)
        ).result(timeout=120)
        self.has_observation = True
        return result

    def observe(self):
        self.touched = time.monotonic()
        result = self.pool.submit(_observe).result(timeout=120)
        self.has_observation = True
        return result

    def pin_reference(self):
        self.touched = time.monotonic()
        result = self.pool.submit(_pin_reference).result(timeout=120)
        self.has_observation = True
        return result

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
