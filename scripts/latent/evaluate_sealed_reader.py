#!/usr/bin/env python3
"""Evaluate one validation-selected reader on the sealed, freshly collected panel."""

import argparse, json, re, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet.latent_inference import LatentModels
from flypet.latent_observation import from_arrays
from flypet.latent_reader_data import parse_reply, score
from flypet.neural_records import file_hash, write_json
from flypet import connectome as C
from scripts.latent.train_reader_repair import parse_change
from flypet.latent_questions import (
    HELD_SUMMARY,
    HELD_CHANGE_QUALITATIVE,
    SUMMARY as CANONICAL_SUMMARY,
)
from flypet.response_semantics import stated_direction

AUDIT = "Report the current approach_hz, avoid_hz and valence as JSON. Valence is (approach_hz-avoid_hz)/(approach_hz+avoid_hz+1), positive when approach exceeds avoidance. Use one decimal for rates and three for valence."
SUMMARY = HELD_SUMMARY
CHANGE = "How did the response change relative to the reference observation?"


def direction(text):
    return stated_direction(text, "current")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "bundle", "selection", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--max-seconds", type=int, default=2000)
    a = p.parse_args()
    lock = json.loads(a.selection.read_text())
    bundle_hash = file_hash(a.bundle / "manifest.json")
    if (
        lock.get("bundle_manifest_sha256") != bundle_hash
        or lock.get("selection_data") != "validation"
    ):
        raise ValueError("A matching pre-evaluation selection lock is required")
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = json.loads((a.data / "manifest.json").read_text())
    if manifest["status"] != "sealed_complete":
        raise ValueError("Sealed panel is incomplete")
    for name, digest in manifest["files"].items():
        if file_hash(a.data / name) != digest:
            raise ValueError("Sealed artifact hash mismatch")
    models = LatentModels(a.bundle)
    pre = models.reader_preprocessing
    roots = np.load(a.data / "root_ids.npy")
    order = C.neuron_order()[0]
    if not np.array_equal(roots, order):
        raise ValueError("Connectome order differs")
    annotations = C.annotations().reindex(order)
    cell = annotations.cell_class.astype(str).to_numpy()
    nt = annotations.top_nt.astype(str).to_numpy()
    types = annotations.cell_type.astype(str).to_numpy()
    approach = np.flatnonzero(
        (cell == "MBON") & (nt != "glutamate") & ((nt == "acetylcholine") | (types == "MBON11"))
    )
    avoid = np.flatnonzero((cell == "MBON") & (nt == "glutamate"))
    observations = []
    for folder in sorted((a.data / "episodes").iterdir()):
        episode = json.loads((folder / "episode.json").read_text())
        with np.load(folder / "neural.npz", allow_pickle=False) as z:
            reference = from_arrays(z, 0, pre, episode["dt_ms"])
            for phase in (0, 4, 5):
                begin = int(z["phase_end_tick"][phase - 1]) if phase else 0
                end = int(z["phase_end_tick"][phase])
                duration = (end - begin) * episode["dt_ms"] / 1000
                mask = (z["spike_tick"] >= begin) & (z["spike_tick"] < end)
                counts = np.bincount(z["spike_neuron_index"][mask], minlength=len(order))
                ap = float(counts[approach].sum() / duration)
                av = float(counts[avoid].sum() / duration)
                target = [ap, av, (ap - av) / (ap + av + 1)]
                if abs(target[2] - episode["metrics"][phase]["valence"]) > 1e-6:
                    raise ValueError("Raw readout differs from recorded simulator result")
                observations.append(
                    {
                        "episode": episode["episode"],
                        "family": episode["family"],
                        "seed": episode["seed"],
                        "phase": phase,
                        "reference": reference,
                        "current": from_arrays(z, phase, pre, episode["dt_ms"]),
                        "target": target,
                    }
                )
    write_json(
        a.out / "evaluation_lock.json",
        {
            "selection": lock,
            "dataset_manifest_sha256": file_hash(a.data / "manifest.json"),
            "questions": {
                "audit": AUDIT,
                "summary": SUMMARY,
                "summary_canonical": CANONICAL_SUMMARY,
                "change": CHANGE,
                "change_qualitative": HELD_CHANGE_QUALITATIVE,
            },
            "observations": len(observations),
            "donor_rule": "rotate complete reference/current pairs by one two-seed recipe group (six observations)",
        },
    )
    targets = np.array([r["target"] for r in observations])
    donors = np.roll(np.arange(len(observations)), 6)
    predictions = {}

    def generate(kind, indices, question, donor=False):
        out = []
        for number, index in enumerate(indices, 1):
            if time.monotonic() - start > a.max_seconds:
                raise TimeoutError("Fresh evaluation deadline")
            item = observations[int(donors[index]) if donor else index]
            text = models.read(
                item["current"],
                item["reference"],
                question=question,
                mode="comparison" if kind.startswith("change") else "current",
            )
            out.append(text)
            if number % 32 == 0:
                print(
                    json.dumps(
                        {
                            "stage": kind,
                            "completed": number,
                            "total": len(indices),
                            "wall_s": time.monotonic() - start,
                        }
                    ),
                    flush=True,
                )
        return out

    all_ids = list(range(len(observations)))
    for mode in ("neural", "donor"):
        texts = generate(mode, all_ids, AUDIT, mode == "donor")
        predictions[mode] = texts
    predictions["summary"] = generate("summary", all_ids, SUMMARY)
    changed = [i for i, r in enumerate(observations) if r["phase"] == 4]
    predictions["change"] = generate("change", changed, CHANGE)
    predictions["change_qualitative"] = generate(
        "change_qualitative", changed, HELD_CHANGE_QUALITATIVE
    )
    predictions["summary_canonical"] = generate("summary_canonical", changed, CANONICAL_SUMMARY)
    numeric = score(predictions["neural"], targets)
    donor_score = score(predictions["donor"], targets)
    numeric_errors = np.array(
        [
            abs(parse_reply(text)[2] - target[2]) if parse_reply(text) is not None else 2.0
            for text, target in zip(predictions["neural"], targets)
        ]
    )
    donor_errors = np.array(
        [
            abs(parse_reply(text)[2] - target[2]) if parse_reply(text) is not None else 2.0
            for text, target in zip(predictions["donor"], targets)
        ]
    )
    actual_sign = np.sign(targets[:, 2]).astype(int)
    guessed = [direction(text) for text in predictions["summary"]]
    by_sign = {
        str(s): {
            "n": int((actual_sign == s).sum()),
            "accuracy": float(np.mean([p == s for p, t in zip(guessed, actual_sign) if t == s])),
        }
        for s in (-1, 0, 1)
        if (actual_sign == s).any()
    }
    balanced = float(np.mean([r["accuracy"] for r in by_sign.values()]))
    change_errors = []
    deltas = []
    change_valid = 0
    consistent = 0
    for index, text in zip(changed, predictions["change"]):
        baseline = targets[index - 1, 2]  # states are stored as phase 0, 4, 5 per episode
        delta = targets[index, 2] - baseline
        deltas.append(delta)
        values = parse_change(text)
        change_errors.append(abs(values[1] - values[0] - delta) if values is not None else 4.0)
        if values is not None:
            change_valid += 1
            sign = np.sign(values[1] - values[0])
            consistent += int(
                (sign > 0 and "toward approach" in text)
                or (sign < 0 and "toward avoidance" in text)
                or (sign == 0 and ("no resolved change" in text or "not at all" in text))
            )
    change_mae = float(np.mean(change_errors))
    zero_change = float(np.mean(abs(np.array(deltas))))
    qualitative_correct = []
    for index, text in zip(changed, predictions["change_qualitative"]):
        target_sign = (
            np.sign(round(float(targets[index, 2] - targets[index - 1, 2]), 2))
            if models.reader_config.get("separate_context")
            else np.sign(
                round(float(targets[index, 2]), 2) - round(float(targets[index - 1, 2]), 2)
            )
        )
        prediction = stated_direction(text, "comparison")
        qualitative_correct.append(prediction == target_sign)
    numbers_fraction = float(
        np.mean([bool(re.search(r"\d", text)) for text in predictions["summary"]])
    )
    families = sorted({r["family"] for r in observations})
    family_diffs = []
    for family in families:
        mask = np.array([r["family"] == family for r in observations])
        family_diffs.append(float((numeric_errors[mask] - donor_errors[mask]).mean()))
    resample = (
        np.random.default_rng(939)
        .choice(family_diffs, size=(4000, len(families)), replace=True)
        .mean(1)
    )
    gates = {
        "valid_numeric": numeric["valid_fraction"] >= 0.98,
        "current_mae": float(numeric_errors.mean()) <= 0.15,
        "donor_margin": float(numeric_errors.mean()) <= 0.8 * float(donor_errors.mean()),
        "summary_balanced_accuracy": balanced >= 0.8,
        "change_mae": change_mae <= 0.15,
        "change_over_zero": change_mae <= 0.8 * zero_change,
        "qualitative_counts_omitted": numbers_fraction <= 0.05,
        "qualitative_change_accuracy": float(np.mean(qualitative_correct)) >= 0.8,
    }
    report = {
        "status": "complete",
        "scope": manifest["scope"],
        "recipe_groups": len(families),
        "numeric": numeric,
        "donor": donor_score,
        "current_mae_with_invalid_penalty": float(numeric_errors.mean()),
        "donor_mae_with_invalid_penalty": float(donor_errors.mean()),
        "paired_current_minus_donor_95": np.quantile(resample, [0.025, 0.975]).tolist(),
        "summary_by_sign": by_sign,
        "summary_balanced_accuracy": balanced,
        "summary_majority_accuracy": float(
            max(np.mean(actual_sign == s) for s in np.unique(actual_sign))
        ),
        "summary_numeric_fraction": numbers_fraction,
        "qualitative_change_accuracy": float(np.mean(qualitative_correct)),
        "change": {
            "n": len(changed),
            "valid": change_valid,
            "directionally_consistent": consistent,
            "mae_with_invalid_penalty": change_mae,
            "zero_change_mae": zero_change,
        },
        "gates": gates,
        "panel_gate_pass": all(gates.values()),
        "twins_gate": "separate live check still required",
        "wall_s": time.monotonic() - start,
    }
    records = [
        {k: v for k, v in row.items() if k not in ("reference", "current")} for row in observations
    ]
    write_json(a.out / "observations.json", records)
    write_json(a.out / "predictions.json", predictions)
    write_json(a.out / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
