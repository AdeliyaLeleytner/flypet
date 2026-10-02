#!/usr/bin/env python3
"""Replay predicted writer drives in the full brain against reference and equal-energy random drives."""

import argparse, json, multiprocessing, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet import connectome as C
from flypet.neural_runtime import NeuralRuntime
from flypet.neural_records import write_json, file_hash
from scripts.latent.collect_calibration import build, private_hashes

STATE = None


def init(basis_path):
    global STATE
    brain, memory, door, ports, _ = build()
    runtime = NeuralRuntime(brain, input_ids=ports, seed=1, memory=memory)
    with np.load(basis_path, allow_pickle=False) as z:
        basis = z["basis"]
        roots = z["input_root_ids"]
    if not np.array_equal(roots, runtime.root_ids):
        raise ValueError("Writer and runtime port order differ")
    ann = C.annotations().reindex(C.neuron_order()[0])
    classes = ann.cell_class.astype(str).to_numpy()
    populations = {
        name: np.flatnonzero(classes == label)
        for name, label in [("PN", "ALPN"), ("KC", "Kenyon_Cell"), ("MBON", "MBON")]
    }
    STATE = runtime, basis, populations


def run(job):
    runtime, basis, populations = STATE
    target = np.array(job["target"], np.float32)
    predicted = np.array(job["predicted"], np.float32)
    probe = target.copy()
    probe[-2:] = 0
    probe = basis @ probe
    reference = basis @ target
    prediction = basis @ predicted
    random = np.random.default_rng(100000 + job["case"]).permutation(reference)
    arms = {}
    for name, drive in [
        ("reference", reference),
        ("predicted", prediction),
        ("random_equal_energy", random),
    ]:
        runtime.reset(seed=job["seed"], reset_memory=True)
        runtime.set_learning(False)
        before = runtime.advance(probe, 250)
        before_val = runtime.memory.valence(before.result).score
        runtime.set_learning(True)
        training = runtime.advance(drive, 250)
        runtime.set_learning(False)
        runtime.advance(np.zeros_like(drive), 250)
        after = runtime.advance(probe, 250)
        after_val = runtime.memory.valence(after.result).score
        rates = training.result.rate_vector(runtime.brain.n)
        arms[name] = {
            "before_valence": before_val,
            "after_valence": after_val,
            "delta": after_val - before_val,
            "memory_strength": runtime.memory.memory_strength(),
            "input_sum_hz": float(drive.sum()),
            "input_max_hz": float(drive.max()),
            "spikes": sum(training.result.counts.values()),
            "population_rates": {p: rates[idx].tolist() for p, idx in populations.items()},
        }
    return {**job, "arms": arms}


def cosine(a, b):
    a, b = np.asarray(a), np.asarray(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 0 else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cases", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--calibrated-predictions", type=Path)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    protected = private_hashes()
    start = time.monotonic()
    report = json.loads((a.fit / "report.json").read_text())
    choice = min(["mlp", "ridge"], key=lambda k: report["metrics"]["validation"][k]["port_mae_hz"])
    source = a.calibrated_predictions or a.fit / "predictions.json"
    predictions = json.loads(source.read_text())["transfer_test"]
    prediction_key = "rates_hz" if a.calibrated_predictions else choice + "_rates_hz"
    if a.calibrated_predictions:
        choice += " with calibrated dopamine head"
    rng = np.random.default_rng(918)
    indices = np.sort(
        rng.choice(len(predictions["ids"]), min(a.cases, len(predictions["ids"])), replace=False)
    )
    jobs = [
        {
            "case": int(i),
            "example_index": predictions["ids"][i],
            "seed": seed,
            "target": predictions["targets_hz"][i],
            "predicted": predictions[prediction_key][i],
        }
        for i in indices
        for seed in [11, 23]
    ]
    write_json(
        a.out / "plan.json",
        {
            "choice_by_validation": choice,
            "jobs": jobs,
            "source_predictions_sha256": file_hash(source),
            "scope": "held writer targets, familiar chemicals; same seeded input uniforms across arms",
        },
    )
    rows = []
    with ProcessPoolExecutor(
        max_workers=a.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=init,
        initargs=(str(a.fit / "preprocessing.npz"),),
    ) as pool:
        for row in pool.map(run, jobs):
            rows.append(row)
            if len(rows) % 8 == 0:
                print(json.dumps({"completed": len(rows), "total": len(jobs)}), flush=True)
    summary = {}
    for name in ["predicted", "random_equal_energy"]:
        population = {}
        for pop in ["PN", "KC", "MBON"]:
            vals = [
                cosine(
                    r["arms"]["reference"]["population_rates"][pop],
                    r["arms"][name]["population_rates"][pop],
                )
                for r in rows
            ]
            good = [v for v in vals if v is not None]
            population[pop] = {
                "mean_cosine_nonzero": float(np.mean(good)) if good else None,
                "nonzero_pairs": len(good),
                "total": len(vals),
            }
        summary[name] = {
            "mean_abs_delta_error": float(
                np.mean(
                    [abs(r["arms"][name]["delta"] - r["arms"]["reference"]["delta"]) for r in rows]
                )
            ),
            "population_similarity": population,
        }
    unchanged = protected == private_hashes()
    write_json(a.out / "episodes.json", rows)
    write_json(
        a.out / "report.json",
        {
            "status": "complete",
            "writer_choice": choice,
            "cases": len(indices),
            "seeds_per_case": 2,
            "metrics": summary,
            "pet_unchanged": unchanged,
            "wall_s": time.monotonic() - start,
        },
    )
    if not unchanged:
        raise RuntimeError("Personal state changed")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
