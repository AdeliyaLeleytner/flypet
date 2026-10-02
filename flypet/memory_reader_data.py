"""CPU-only preparation and scoring for the memory-state language experiment.

The dataset owns the experimental split. This module validates that complete
history templates stay together, then fits every learned preprocessing quantity
on training rows only. It never imports a simulator, torch or transformers.
"""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re

import numpy as np

FIELDS = ("approach_hz", "avoid_hz", "valence")
DAN_MODES = ("none", "reward", "punish")


def digest(path):
    h = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_digest(value):
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def target_values(row):
    values = {key: float(row["target"][key]) for key in FIELDS}
    if not all(np.isfinite(list(values.values()))) or min(values[k] for k in FIELDS[:2]) < 0:
        raise ValueError("Targets must contain finite valence and nonnegative readout rates")
    if not -1 <= values["valence"] <= 1:
        raise ValueError("Valence must be between -1 and 1")
    return values


def target_text(row):
    v = target_values(row)
    return '{"approach_hz":%.1f,"avoid_hz":%.1f,"valence":%.3f}' % (
        v["approach_hz"],
        v["avoid_hz"],
        v["valence"],
    )


def parse_output(text):
    """Parse generated JSON; never evaluate model-generated code."""
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if set(obj) != set(FIELDS) or any(type(obj[k]) not in (int, float) for k in FIELDS):
            return None
        return target_values({"target": obj})
    except (TypeError, ValueError, KeyError):
        return None


def direction(value, threshold=0.05):
    return 1 if value > threshold else -1 if value < -threshold else 0


def score_outputs(texts, rows):
    if len(texts) != len(rows):
        raise ValueError("Each evaluated row must have exactly one generated response")
    parsed = [parse_output(text) for text in texts]
    valid = [i for i, value in enumerate(parsed) if value is not None]
    n = len(rows)
    result = {
        "n": n,
        "parsed": len(valid),
        "parse_rate": len(valid) / n if n else None,
        "direction_threshold": 0.05,
        "direction_accuracy_all": sum(
            direction(parsed[i]["valence"]) == direction(target_values(rows[i])["valence"])
            for i in valid
        )
        / n
        if n
        else None,
    }
    result["truth_direction_counts"] = {
        str(k): sum(direction(target_values(row)["valence"]) == k for row in rows)
        for k in (-1, 0, 1)
    }
    result["predicted_direction_counts_valid"] = {
        str(k): sum(direction(parsed[i]["valence"]) == k for i in valid) for k in (-1, 0, 1)
    }
    result["mae_valid"] = {
        k: float(np.mean([abs(parsed[i][k] - target_values(rows[i])[k]) for i in valid]))
        if valid
        else None
        for k in FIELDS
    }
    # Missing generations are errors for accuracy; MAE is explicitly conditional
    # on parsing and always accompanied by coverage.
    result["self_consistency_mae_valid"] = (
        float(
            np.mean(
                [
                    abs(
                        parsed[i]["valence"]
                        - (parsed[i]["approach_hz"] - parsed[i]["avoid_hz"])
                        / (parsed[i]["approach_hz"] + parsed[i]["avoid_hz"] + 1)
                    )
                    for i in valid
                ]
            )
        )
        if valid
        else None
    )
    return result


def load_dataset(directory):
    directory = Path(directory)
    rows_path, features_path = directory / "examples.jsonl", directory / "features.npz"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    ids = [str(row["example_id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate example_id")
    with np.load(features_path, allow_pickle=False) as z:
        feature_ids = list(map(str, z["example_ids"]))
        neuron_indices = z["neuron_indices"].astype(np.int64)
        rates = z["rates"].astype(np.float32)
        mbon_indices = z["mbon_indices"].astype(np.int64) if "mbon_indices" in z else None
        mbon_rates = z["mbon_rates"].astype(np.float32) if "mbon_rates" in z else None
    if len(set(feature_ids)) != len(feature_ids) or set(ids) != set(feature_ids):
        raise ValueError("Feature rows and examples must have the same unique IDs")
    if rates.shape != (len(ids), len(neuron_indices)):
        raise ValueError("rates must have shape (examples, neurons)")
    if len(set(neuron_indices)) != len(neuron_indices) or np.any(neuron_indices < 0):
        raise ValueError("Neuron indices must be unique nonnegative integers")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Rates must be finite and nonnegative")
    positions = {name: i for i, name in enumerate(feature_ids)}
    reorder = [positions[name] for name in ids]
    rates = rates[reorder]
    if (mbon_indices is None) != (mbon_rates is None):
        raise ValueError("MBON rates and neuron indices must be provided together")
    if mbon_indices is not None:
        neuron_positions = {int(index): i for i, index in enumerate(neuron_indices)}
        if not all(int(index) in neuron_positions for index in mbon_indices):
            raise ValueError("MBON identity absent from downstream feature schema")
        mbon_rates = mbon_rates[reorder]
        neural_mbon = rates[:, [neuron_positions[int(index)] for index in mbon_indices]]
        row_mbon = np.asarray([row["mbon_rates_hz"] for row in rows], dtype=np.float32)
        if not np.array_equal(neural_mbon, mbon_rates) or not np.array_equal(row_mbon, mbon_rates):
            raise ValueError("MBON rates/order disagree between joined rows and downstream matrix")
    protocol_path = directory / "protocol.json"
    protocol = json.loads(protocol_path.read_text()) if protocol_path.exists() else {}
    for row in rows:
        target_values(row)
        if "probe_rate_hz" not in row:
            if "odor_rate_max_hz" not in protocol:
                raise ValueError("Probe rate must be explicit in row or dataset protocol")
            row["probe_rate_hz"] = float(protocol["odor_rate_max_hz"])
        row.setdefault("probe_duration_ms", row.get("duration_ms", protocol.get("duration_ms")))
        if row["probe_duration_ms"] is None:
            raise ValueError("Probe duration is missing")
        row.setdefault("probe_odor_scale", protocol.get("odor_scale", 1.0))
    sources = {"examples.jsonl": digest(rows_path), "features.npz": digest(features_path)}
    if protocol_path.exists():
        sources["protocol.json"] = digest(protocol_path)
    return rows, rates, neuron_indices, sources


def load_profiles(path):
    """Represent missing receptor observations with an explicit presence mask."""
    source = json.loads(Path(path).read_text())
    if "compounds" not in source:
        return source, {
            "representation": "caller supplied fixed profile",
            "source_sha256": digest(path),
        }
    glomeruli = source["glomeruli"]
    vectors = {}
    for chemical, detail in source["compounds"].items():
        values = [detail["glomerular_profile"].get(name) for name in glomeruli]
        vectors[chemical] = [0.0 if v is None else float(v) for v in values] + [
            float(v is not None) for v in values
        ]
    return vectors, {
        "representation": "glomerular values followed by explicit observed mask",
        "glomeruli": glomeruli,
        "source_sha256": digest(path),
    }


def compound(row):
    return str(row.get("probe_compound", row.get("probe_chemical", "")))


def event_compound(event):
    return str(event.get("chemical", event.get("compound", "")) or "")


def encode_history(rows, vocab, max_events, chemical_profiles=None):
    """Apply a fixed history schema without fitting anything on queried rows."""
    positions = {name: i for i, name in enumerate(vocab)}
    if any(len(row["history"]) > max_events for row in rows):
        raise ValueError(
            "Evaluation history exceeds train maximum; use an explicit fixed sequence schema"
        )
    profiles = chemical_profiles or {}
    profile_size = len(next(iter(profiles.values()))) if profiles else 0
    if profiles and any(len(value) != profile_size for value in profiles.values()):
        raise ValueError("Chemical profiles must share the same fixed receptor schema")
    # Unknown chemical, presence, rate, duration, odor scale, reinforcement rate,
    # dopamine mode, then fixed DoOR profile. Seeds are provenance only.
    event_width = len(vocab) + 1 + 1 + 4 + len(DAN_MODES) + profile_size
    history = np.zeros((len(rows), (max_events + 1) * event_width), dtype=np.float32)
    for i, row in enumerate(rows):
        events = [
            {
                "chemical": compound(row),
                "rate_hz": row["probe_rate_hz"],
                "duration_ms": row.get("probe_duration_ms", row.get("duration_ms", 0)),
                "odor_scale": row.get("probe_odor_scale", 1.0),
                "dan_mode": "none",
            }
        ] + row["history"]
        for j, event in enumerate(events):
            start = j * event_width
            name = event_compound(event)
            history[i, start + positions.get(name, len(vocab))] = 1
            numeric = start + len(vocab) + 1
            history[i, numeric] = 1
            history[i, numeric + 1] = float(event.get("rate_hz", 0))
            history[i, numeric + 2] = float(event.get("duration_ms", 0))
            history[i, numeric + 3] = float(event.get("odor_scale", 1))
            history[i, numeric + 4] = float(event.get("reinforcement_rate_hz", 0))
            mode = event.get("dan_mode", "none") or "none"
            if mode not in DAN_MODES:
                raise ValueError(f"Unknown dopamine mode {mode}")
            history[i, numeric + 5 + DAN_MODES.index(mode)] = 1
            if profiles:
                if name and name not in profiles:
                    raise ValueError(f"Missing chemical receptor profile: {name}")
                history[i, numeric + 8 : numeric + 8 + profile_size] = profiles.get(
                    name, [0] * profile_size
                )
    return history


def prepare_dataset(
    rows, rates, neuron_indices, *, split_field="split", min_frequency=1, chemical_profiles=None
):
    if min_frequency < 1:
        raise ValueError("min_frequency must be positive")
    labels = [str(row[split_field]) for row in rows]
    splits = {label: np.flatnonzero(np.asarray(labels) == label) for label in sorted(set(labels))}
    if any(name not in splits or not len(splits[name]) for name in ("train", "validation")):
        raise ValueError("Nonempty train and validation splits are required")
    groups = defaultdict(set)
    for row, label in zip(rows, labels):
        # Unseen-query rows intentionally revisit the same learned animal. They
        # never enter fitting/selection and are not an independent-history test.
        if label != "query_chemical_test":
            groups[str(row["split_group_id"])].add(label)
    overlap = {group: sorted(parts) for group, parts in groups.items() if len(parts) > 1}
    if overlap:
        raise ValueError(f"History template crosses splits: {overlap}")
    tr = splits["train"]
    keep = np.flatnonzero(np.count_nonzero(rates[tr] > 0, axis=0) >= min_frequency)
    if not len(keep):
        raise ValueError("No neuron features present in training data")
    log = np.log1p(rates[:, keep])
    mu = log[tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    sd = (log[tr].std(axis=0, dtype=np.float64) + 0.001).astype(np.float32)
    neural = (log - mu) / sd

    # Every ordered exposure is included. Vocabulary and maximum represented
    # sequence length come from training; longer histories fail explicitly.
    vocab = sorted(
        {
            c
            for i in tr
            for c in ([compound(rows[i])] + [event_compound(e) for e in rows[i]["history"]])
            if c
        }
    )
    max_events = max(len(rows[i]["history"]) for i in tr)
    profiles = chemical_profiles or {}
    profile_size = len(next(iter(profiles.values()))) if profiles else 0
    history = encode_history(rows, vocab, max_events, profiles)
    hmu = history[tr].mean(axis=0, dtype=np.float64).astype(np.float32)
    hstd = history[tr].std(axis=0, dtype=np.float64).astype(np.float32)
    hsd = np.where(hstd < 1e-8, 1.0, hstd + 0.001).astype(np.float32)
    history = (history - hmu) / hsd
    manifest = {
        "schema_version": 1,
        "split_field": split_field,
        "split_unit": "complete history template across seeds",
        "counts": {name: len(indices) for name, indices in splits.items()},
        "split_fingerprint": json_digest(
            {name: [rows[i]["example_id"] for i in ids] for name, ids in splits.items()}
        ),
        "preprocessing": {
            "fit_split": "train",
            "min_frequency": min_frequency,
            "neural_features": len(keep),
            "history_vocabulary": vocab,
            "max_history_events": max_events,
            "profile_size": profile_size,
            "unknown_chemical_column": len(vocab),
            "history_seeds_as_input": False,
            "probe_rate_source": "explicit row or hashed dataset protocol",
            "history_event_numeric_fields": [
                "rate_hz",
                "duration_ms",
                "odor_scale",
                "reinforcement_rate_hz",
            ],
        },
        "target_fields": list(FIELDS),
        "target_rounding": [1, 1, 3],
        "target_semantics": "MBON rate readouts and derived valence, not actual movement",
        "query_chemical_test_shares_histories": True,
    }
    state_ids = [sha256(np.asarray(rate, dtype="<f4").tobytes()).hexdigest() for rate in rates]
    split_states = {name: {state_ids[int(i)] for i in ids} for name, ids in splits.items()}
    names = sorted(splits)
    manifest["audit"] = {
        "raw_neural_state_overlap": {
            f"{left}:{right}": len(split_states[left] & split_states[right])
            for j, left in enumerate(names)
            for right in names[j + 1 :]
        },
        "feature_coverage": {
            name: {
                "fraction_rate_mass_retained": float(
                    rates[ids][:, keep].sum(dtype=np.float64) / rates[ids].sum(dtype=np.float64)
                )
                if rates[ids].sum()
                else 1.0,
                "fraction_active_entries_retained": float(
                    np.count_nonzero(rates[ids][:, keep]) / np.count_nonzero(rates[ids])
                )
                if np.count_nonzero(rates[ids])
                else 1.0,
            }
            for name, ids in splits.items()
        },
    }
    return {
        "neural": neural,
        "history": history,
        "null": np.zeros_like(neural),
        "splits": splits,
        "captions": [target_text(row) for row in rows],
        "manifest": manifest,
        "neuron_indices": neuron_indices[keep],
        "neural_mu": mu,
        "neural_sd": sd,
        "history_mu": hmu,
        "history_sd": hsd,
        "history_vocabulary": vocab,
    }


def swap_pairs(rows, selected):
    """Match opposite reinforcement assignments without looking at target values.

    The key is the fixed conditioning pair, dose, history RNG seed and probe.
    No output-derived distance is used to select the donor.
    """
    groups = defaultdict(list)
    for i in selected:
        row = rows[int(i)]
        key = (
            tuple(row["training_compounds"]),
            row["pairing_trials"],
            row["history_seed"],
            compound(row),
            row.get("probe_seed"),
            row.get("probe_duration_ms", row.get("duration_ms")),
            row["probe_rate_hz"],
        )
        groups[key].append(int(i))
    pairs = []
    for members in groups.values():
        for target in members:
            candidates = [
                source
                for source in members
                if source != target
                and rows[source]["history_id"] != rows[target]["history_id"]
                and rows[source]["assignment"] != rows[target]["assignment"]
            ]
            if len(candidates) > 1:
                raise ValueError(
                    f"Ambiguous prespecified swap counterpart for {rows[target]['example_id']}"
                )
            if candidates:
                pairs.append((target, candidates[0]))
    return pairs
