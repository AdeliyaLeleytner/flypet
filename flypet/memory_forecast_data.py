"""Audited prospective inputs and scoring for frozen-language memory forecasts.

Only two fixed diagnostic responses and a requested chemical profile enter the
neural interface. The requested response is used exclusively to form the label.
This module neither imports nor runs the simulator or a language model.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

from flypet.memory_reader_data import digest, encode_history, json_digest, load_profiles, swap_pairs

DIAGNOSTICS = ("methyl acetate", "1-hexanol")
FIELD = "delta_valence"


def target_values(row):
    value = row["target"][FIELD]
    if type(value) not in (int, float) or not np.isfinite(value) or not -2 <= value <= 2:
        raise ValueError("delta_valence must be a finite number in [-2, 2]")
    return {FIELD: float(value)}


def target_text(row):
    return '{"delta_valence":%.3f}' % target_values(row)[FIELD]


def parse_output(text):
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        value = json.loads(match.group())
        if set(value) != {FIELD}:
            return None
        return target_values({"target": value})
    except (TypeError, ValueError, KeyError):
        return None


def score_outputs(texts, rows):
    if len(texts) != len(rows):
        raise ValueError("Each evaluated row must have exactly one generated response")
    parsed = [parse_output(text) for text in texts]
    valid = [i for i, value in enumerate(parsed) if value is not None]
    errors = [parsed[i][FIELD] - target_values(rows[i])[FIELD] for i in valid]
    groups = defaultdict(dict)
    for i, row in enumerate(rows):
        key = (
            row["split_group_id"],
            row["history_seed"],
            row["pairing_trials"],
            row["probe_compound"],
            row["probe_seed"],
        )
        if row["assignment"] in groups[key]:
            raise ValueError("Duplicate assignment in matched contrast group")
        groups[key][row["assignment"]] = i
    contrasts, truths, predicted, n_pairs = [], [], [], 0
    for group in groups.values():
        if set(group) != {0, 1}:
            continue
        n_pairs += 1
        i, j = group[0], group[1]
        if parsed[i] is None or parsed[j] is None:
            continue
        truth = target_values(rows[j])[FIELD] - target_values(rows[i])[FIELD]
        pred = parsed[j][FIELD] - parsed[i][FIELD]
        truths.append(abs(truth))
        predicted.append(abs(pred))
        contrasts.append(abs(pred - truth))
    return {
        "n": len(rows),
        "parsed": len(valid),
        "parse_rate": len(valid) / len(rows) if rows else None,
        "mae_valid": {FIELD: float(np.abs(errors).mean()) if errors else None},
        "rmse_valid": {FIELD: float(np.sqrt(np.square(errors).mean())) if errors else None},
        "no_change_mae": float(np.mean([abs(target_values(r)[FIELD]) for r in rows]))
        if rows
        else None,
        "matched_assignment_contrast": {
            "paired_histories": n_pairs,
            "valid_pairs": len(contrasts),
            "coverage": len(contrasts) / n_pairs if n_pairs else None,
            "mae_valid": float(np.mean(contrasts)) if contrasts else None,
            "mean_absolute_truth_contrast_valid": float(np.mean(truths)) if truths else None,
            "mean_absolute_predicted_contrast_valid": float(np.mean(predicted))
            if predicted
            else None,
        },
    }


def split_indices(rows):
    labels = np.asarray([r["split"] for r in rows])
    splits = {s: np.flatnonzero(labels == s) for s in sorted(set(labels))}
    if any(s not in splits or not len(splits[s]) for s in ("train", "validation")):
        raise ValueError("Nonempty train and validation splits required")
    families = defaultdict(set)
    for r in rows:
        if r["probe_compound"] in DIAGNOSTICS:
            raise ValueError("Diagnostic probe cannot be a forecast target")
        if r["split"] != "query_chemical_test":
            families[r["split_group_id"]].add(r["split"])
        if r["split"] in ("train", "validation") and (
            r["probe_compound"] == "acetic acid" or "acetic acid" in r["training_compounds"]
        ):
            raise ValueError("Excluded chemical entered fitting or validation")
        target_values(r)
    if any(len(s) > 1 for s in families.values()):
        raise ValueError("Conditioning family crosses splits")
    return splits


def standardize(matrix, training):
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Feature matrix must be finite and two-dimensional")
    mu = matrix[training].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = matrix[training].std(axis=0, dtype=np.float64)
    sd = np.where(sd < 1e-8, 1.0, sd).astype(np.float32)
    return (matrix - mu) / sd, mu, sd


def prepare_dataset(rows, neural_raw, history_raw, manifest=None):
    if len(neural_raw) != len(rows) or len(history_raw) != len(rows):
        raise ValueError("Feature rows must match example rows")
    splits = split_indices(rows)
    neural, nmu, nsd = standardize(neural_raw, splits["train"])
    history, hmu, hsd = standardize(history_raw, splits["train"])
    report = deepcopy(manifest or {})
    report.update(
        {
            "counts": {k: len(v) for k, v in splits.items()},
            "split_fingerprint": json_digest(
                {k: [rows[i]["example_id"] for i in v] for k, v in splits.items()}
            ),
            "preprocessing": {
                "fit_split": "train",
                "neural_features": neural.shape[1],
                "history_features": history.shape[1],
                "history_seeds_as_input": False,
                "normalization": "training mean/std; constant columns use scale one",
                "target_response_excluded": True,
            },
        }
    )
    return {
        "neural": neural,
        "history": history,
        "null": np.zeros_like(neural),
        "neural_mu": nmu,
        "neural_sd": nsd,
        "history_mu": hmu,
        "history_sd": hsd,
        "splits": splits,
        "captions": [target_text(r) for r in rows],
        "manifest": report,
    }


def prospective_rows(source_rows, profiles):
    by_history = defaultdict(dict)
    for r in source_rows:
        if r["probe_compound"] in by_history[r["history_id"]]:
            raise ValueError("Duplicate history/chemical probe")
        by_history[r["history_id"]][r["probe_compound"]] = r
    rows, features, contexts = [], [], []
    for source in source_rows:
        if source["probe_compound"] in DIAGNOSTICS:
            continue
        diagnostics = [by_history[source["history_id"]][d] for d in DIAGNOSTICS]
        for diagnostic in diagnostics:
            if diagnostic["checkpoint_hash"] != source["checkpoint_hash"]:
                raise ValueError("Diagnostic checkpoint differs from requested checkpoint")
            if (
                diagnostic["example_id"] == source["example_id"]
                or diagnostic["probe_seed"] == source["probe_seed"]
            ):
                raise ValueError("Target probe or its RNG leaked into diagnostics")
        neural = np.concatenate(
            [np.log1p(r["mbon_rates_hz"]) for r in diagnostics]
            + [profiles[source["probe_compound"]]]
        )
        if not np.isfinite(neural).all() or any(
            np.any(np.asarray(r["mbon_rates_hz"]) < 0) for r in diagnostics
        ):
            raise ValueError("Invalid diagnostic rates")
        row = deepcopy(source)
        row["observed_response_target"] = row.pop("target")
        row["target"] = {
            FIELD: float(
                source["target"]["valence"] - source["naive_reference"]["valence"]["score"]
            )
        }
        row["diagnostic_example_ids"] = [r["example_id"] for r in diagnostics]
        rows.append(row)
        features.append(neural)
        contexts.append({"target": row["example_id"], "sources": row["diagnostic_example_ids"]})
    split_indices(rows)
    return rows, np.asarray(features, dtype=np.float32), contexts


def load_dataset(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, expected in manifest["derived_files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Derived dataset hash mismatch: {name}")
    rows = [
        json.loads(line)
        for line in (directory / "examples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    ids = [r["example_id"] for r in rows]
    with np.load(directory / "features.npz", allow_pickle=False) as z:
        if len(set(ids)) != len(ids) or ids != list(z["example_ids"]):
            raise ValueError("Derived feature/example IDs differ")
        neural, history = z["neural_raw"].copy(), z["history_raw"].copy()
    hashes = {
        name: digest(directory / name)
        for name in ("examples.jsonl", "features.npz", "manifest.json", "prospective_context.json")
    }
    return rows, neural, history, manifest, hashes


def build_dataset(source_directory, analysis_directory, out):
    source_directory, analysis_directory, out = map(
        Path, (source_directory, analysis_directory, out)
    )
    source = [
        json.loads(line)
        for line in (source_directory / "examples.jsonl").read_text().splitlines()
        if line.strip()
    ]
    protocol = json.loads((source_directory / "protocol.json").read_text())
    for r in source:
        r.setdefault("probe_rate_hz", protocol["odor_rate_max_hz"])
        r.setdefault("probe_duration_ms", r["duration_ms"])
        r.setdefault("probe_odor_scale", protocol.get("odor_scale", 1.0))
    profiles, profile_metadata = load_profiles(source_directory / "input_profiles.json")
    rows, neural, contexts = prospective_rows(source, profiles)
    with np.load(analysis_directory / "prospective_features.npz", allow_pickle=False) as z:
        if list(z["example_ids"]) != [r["example_id"] for r in rows] or not np.array_equal(
            z["features"], neural
        ):
            raise ValueError("Forecast inputs differ from audited CPU prospective task")
    expected = json.loads((analysis_directory / "prospective_context.json").read_text())
    if expected["rows"] != contexts or list(expected["fixed_diagnostic_chemicals"]) != list(
        DIAGNOSTICS
    ):
        raise ValueError("Forecast provenance differs from audited CPU task")
    with np.load(analysis_directory / "prospective_delta_predictions.npz", allow_pickle=False) as z:
        if list(z["example_ids"]) != [r["example_id"] for r in rows] or not np.array_equal(
            z["targets"][:, 2], [r["target"][FIELD] for r in rows]
        ):
            raise ValueError("Forecast labels differ from audited CPU delta task")
    training = [r for r in rows if r["split"] == "train"]
    vocab = sorted(
        {
            c
            for r in training
            for c in [r["probe_compound"]] + [e["chemical"] for e in r["history"]]
            if c
        }
    )
    max_events = max(len(r["history"]) for r in training)
    history = encode_history(rows, vocab, max_events, profiles)
    out.mkdir(parents=True, exist_ok=False)
    (out / "examples.jsonl").write_text(
        "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
    )
    np.savez_compressed(
        out / "features.npz",
        example_ids=np.asarray([r["example_id"] for r in rows]),
        neural_raw=neural,
        history_raw=history,
    )
    (out / "prospective_context.json").write_text(json.dumps(expected, indent=2) + "\n")
    (out / "expected_predictions.json").write_text(
        json.dumps(
            [
                {
                    "example_id": r["example_id"],
                    "split": r["split"],
                    "history_split": r["history_split"],
                    "truth": target_values(r),
                    "diagnostic_example_ids": r["diagnostic_example_ids"],
                }
                for r in rows
            ],
            indent=2,
        )
        + "\n"
    )
    manifest = {
        "schema_version": 1,
        "task": "prospective_delta_valence",
        "target_fields": [FIELD],
        "target_bounds": [-2, 2],
        "target_rounding_decimals": 3,
        "target_semantics": "Requested-probe model-derived MBON valence minus matched naive valence",
        "diagnostic_chemicals": list(DIAGNOSTICS),
        "target_response_excluded": True,
        "feature_order": expected["feature_order"],
        "history_vocabulary": vocab,
        "max_history_events": max_events,
        "chemical_profile": profile_metadata,
        "split_unit": "conditioning chemical pair across assignments/doses/seeds",
        "query_chemical_test_shares_histories": True,
        "cpu_raw_features_exact_match": True,
        "cpu_delta_targets_exact_match": True,
        "sources": {
            **{
                f"source/{n}": digest(source_directory / n)
                for n in (
                    "examples.jsonl",
                    "protocol.json",
                    "plan.json",
                    "feature_schema.json",
                    "input_profiles.json",
                )
            },
            **{
                f"analysis/{n}": digest(analysis_directory / n)
                for n in (
                    "prospective_features.npz",
                    "prospective_context.json",
                    "prospective_delta_predictions.npz",
                )
            },
        },
        "code_sha256": {
            "flypet/memory_forecast_data.py": digest(__file__),
            "flypet/memory_reader_data.py": digest(
                Path(__file__).with_name("memory_reader_data.py")
            ),
        },
        "derived_files": {
            n: digest(out / n)
            for n in (
                "examples.jsonl",
                "features.npz",
                "prospective_context.json",
                "expected_predictions.json",
            )
        },
    }
    manifest = prepare_dataset(rows, neural, history, manifest)["manifest"]
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "out": str(out),
                "counts": manifest["counts"],
                "split_fingerprint": manifest["split_fingerprint"],
                "preprocessing": manifest["preprocessing"],
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    build_dataset(args.source, args.analysis, args.out)
