"""Numpy-only live feature extraction, shared with the dialogue dataset schema."""

from __future__ import annotations
import numpy as np
from .latent_reader_data import read_phase
from .neural_records import spike_arrays


def from_arrays(neural, phase, preprocessing, dt_ms=0.1):
    indices = preprocessing["neuron_indices"]
    fine = read_phase(neural, phase, indices, dt_ms=dt_ms)
    groups = preprocessing["population_ids"]
    sizes = preprocessing["population_sizes"]
    if len(neural["voltage_mv"][phase]) != len(groups):
        raise ValueError("Brain size differs from reader")
    eligible = np.ones(len(groups), bool)
    eligible[preprocessing["input_model_indices"]] = False
    start = int(neural["phase_end_tick"][phase - 1]) if phase else 0
    end = int(neural["phase_end_tick"][phase])
    duration = (end - start) * dt_ms / 1000
    keep = (neural["spike_tick"] >= start) & (neural["spike_tick"] < end)
    ids = neural["spike_neuron_index"][keep]
    ticks = neural["spike_tick"][keep] - start
    allowed = eligible[ids]
    bins = np.minimum(4, ticks[allowed] * 5 // (end - start))
    class_ids = groups[ids[allowed]]
    counts = np.bincount(class_ids * 5 + bins, minlength=len(sizes) * 5).reshape(len(sizes), 5)
    population = np.zeros((len(sizes), 9), np.float32)
    population[:, :5] = np.log1p(counts / np.maximum(sizes[:, None], 1) / (duration / 5))
    population[:, 5] = np.log1p(counts.sum(axis=1) / np.maximum(sizes, 1) / duration)
    for column, raw in [
        (6, (neural["voltage_mv"][phase] + 52) / 7),
        (7, neural["synaptic_mv"][phase] / 20),
        (8, neural["refractory_remaining_ms"][phase] / 2.2),
    ]:
        pooled = np.bincount(
            groups[eligible], weights=raw[eligible], minlength=len(sizes)
        ) / np.maximum(sizes, 1)
        population[:, column] = np.sign(pooled) * np.log1p(abs(pooled)) if column == 7 else pooled
    x = np.clip((fine - preprocessing["mean"]) / preprocessing["std"], -20, 20).astype(np.float32)
    px = np.clip(
        (population - preprocessing["population_mean"]) / preprocessing["population_std"], -20, 20
    ).astype(np.float32)
    return {
        "fine": x,
        "population": px,
        "spike_bins": counts.sum(axis=0),
        "all_spike_bins": np.bincount(np.minimum(4, ticks * 5 // (end - start)), minlength=5),
        "population_rates_hz": counts.sum(axis=1) / np.maximum(sizes, 1) / duration,
    }


def from_step(step, preprocessing, dt_ms=0.1):
    ids, ticks = spike_arrays([step.result], dt_ms)
    arrays = {
        "spike_neuron_index": ids,
        "spike_tick": ticks,
        "phase_end_tick": np.array([round(step.result.duration_s * 1000 / dt_ms)]),
    }
    for key in ["voltage_mv", "synaptic_mv", "refractory_remaining_ms"]:
        arrays[key] = step.observation[key][None]
    return from_arrays(arrays, 0, preprocessing, dt_ms)
