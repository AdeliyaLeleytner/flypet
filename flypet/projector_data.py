"""Reproducible, CPU-only data preparation for the neural-state projector.

No model or simulator is imported. The catalog imports only NumPy/pandas and reads
connectome files only when the caller explicitly requests the sensory boundary.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
import json
from pathlib import Path

import numpy as np

from .catalog import STIMULI

DATASETS = ("state_dataset", "state_dataset_b", "state_dataset_c", "state_pairs")
KEYS = [k for k in STIMULI if k != "sugar_all_labellar"]
SIDES = ["both", "left", "right"]
SIDE_RU = {"both": "обе стороны", "left": "слева", "right": "справа"}
SPLITS = ("train", "validation", "test", "interaction")


def json_hash(value):
    return sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def file_hash(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_state(indices, rates, min_index=0):
    """Sort sparse entries, discard zeros and normalize byte order for hashing.

    Repeated neuron indices are invalid, rather than silently taking the last
    entry (which would make identity depend on storage order).
    """
    indices = np.asarray(indices)
    rates = np.asarray(rates, dtype="<f4")
    if indices.ndim != 1 or rates.ndim != 1 or len(indices) != len(rates):
        raise ValueError("Sparse indices and rates must be equal-length vectors")
    if not np.issubdtype(indices.dtype, np.integer) or np.any(indices < 0):
        raise ValueError("Neuron indices must be nonnegative integers")
    if not np.all(np.isfinite(rates)) or np.any(rates < 0):
        raise ValueError("Neural rates must be finite and nonnegative")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate neuron index in sparse state")
    keep = (indices >= min_index) & (rates != 0)
    order = np.argsort(indices[keep])
    return np.asarray(indices[keep][order], dtype="<i8"), rates[keep][order]


def state_hash(indices, rates):
    digest = sha256()
    digest.update(np.asarray([len(indices)], dtype="<i8").tobytes())
    digest.update(indices.tobytes())
    digest.update(rates.tobytes())
    return digest.hexdigest()


def lvl(value):
    return "нет" if value < 0.05 else "слабо" if value < 0.4 else "сильно"


def caption(label):
    parts = []
    # Stimulus order is not a target variable: canonicalize it too.
    for stimulus in sorted(label["stimuli"], key=lambda s: (s["key"], s["side"], s["rate_hz"])):
        ratio = stimulus["rate_hz"] / STIMULI[stimulus["key"]].default_rate_hz
        strength = "слабо" if ratio < 0.8 else "обычно" if ratio <= 1.2 else "сильно"
        parts.append(
            f"{STIMULI[stimulus['key']].label_ru} ({SIDE_RU[stimulus['side']]}, {strength})"
        )
    b = label["behaviour"]
    walk = "назад" if b["walk_backward"] > 0.05 else "вперёд" if b["walk_forward"] > 0.05 else "нет"
    turn = (
        "влево"
        if b["turn_left"] > max(0.05, b["turn_right"])
        else "вправо"
        if b["turn_right"] > 0.05
        else "нет"
    )
    dop = {"reward": "награда", "punish": "наказание"}.get(label["dopamine"], "нет")
    sens = ", ".join(parts) if parts else "ничего"
    return (
        f"Сенсоры: {sens}. Запах: {label['odor'] or 'нет'}. Дофамин: {dop}. "
        f"Тело: хоботок {lvl(b['proboscis'])}, побег {lvl(b['escape'])}, чистка антенн {lvl(b['grooming'])}, ходьба {walk}, поворот {turn}."
    )


@dataclass
class Sample:
    indices: np.ndarray
    rates: np.ndarray
    label: dict
    source: dict


def load_samples(data_root, datasets=DATASETS):
    """Read every shard in the requested snapshot, refusing missing datasets."""
    data_root = Path(data_root)
    samples, sources = [], []
    for dataset in datasets:
        shards = sorted((data_root / dataset).glob("shard_*.npz"))
        if not shards:
            raise ValueError(f"No data shards in {data_root / dataset}")
        for path in shards:
            before = file_hash(path)
            with np.load(path, allow_pickle=False) as z:
                n, offsets = int(z["n"]), z["offsets"]
                if (
                    len(offsets) != n + 1
                    or offsets[0] != 0
                    or np.any(np.diff(offsets) < 0)
                    or offsets[-1] != len(z["idx"])
                    or len(z["idx"]) != len(z["rate"])
                    or len(z["labels"]) != n
                ):
                    raise ValueError(f"Malformed shard: {path}")
                relative = str(path.relative_to(data_root))
                for row in range(n):
                    start, stop = offsets[row : row + 2]
                    samples.append(
                        Sample(
                            z["idx"][start:stop].copy(),
                            z["rate"][start:stop].copy(),
                            json.loads(z["labels"][row]),
                            {"shard": relative, "row": row},
                        )
                    )
            if before != file_hash(path):
                raise RuntimeError(f"Shard changed while being read: {path}")
            sources.append(
                {"path": relative, "sha256": before, "rows": n, "bytes": path.stat().st_size}
            )
    return samples, sources


def generation_rows(test_rows, n_gen, seed):
    if n_gen < 0:
        raise ValueError("n_gen must be nonnegative")
    return np.random.default_rng(seed).permutation(test_rows)[:n_gen]


def overlap_counts(signatures, split_rows):
    sets = {name: {signatures[int(i)] for i in rows} for name, rows in split_rows.items()}
    return {
        f"{left}:{right}": len(sets[left] & sets[right]) for left, right in combinations(SPLITS, 2)
    }


def prepare_samples(
    samples,
    *,
    min_index=0,
    min_frequency=3,
    seed=0,
    validation_fraction=0.15,
    test_fraction=0.15,
    n_gen=100,
    sources=None,
):
    """Deduplicate, split state groups, then fit preprocessing using train only.

    Deduplication identity is the model-visible sparse state plus canonical text
    target. Different targets for the same state remain separate, but are always
    assigned together. Any group containing sugar+bitter goes to interaction;
    only its actual sugar+bitter rows enter the interaction metric.
    """
    if min_index < 0 or min_frequency < 1:
        raise ValueError("min_index must be >= 0 and min_frequency >= 1")
    if not (
        0 < validation_fraction < 1
        and 0 < test_fraction < 1
        and validation_fraction + test_fraction < 1
    ):
        raise ValueError("Validation/test fractions must be positive and sum to less than one")
    unique = {}
    for sample in samples:
        ix, rate = canonical_state(sample.indices, sample.rates, min_index)
        state_id, target = state_hash(ix, rate), caption(sample.label)
        row_id = json_hash([state_id, target])
        sb = {"sugar", "bitter"} <= {s["key"] for s in sample.label["stimuli"]}
        if row_id not in unique:
            unique[row_id] = {
                "id": row_id,
                "state_id": state_id,
                "target": target,
                "indices": ix,
                "rates": rate,
                "label": sample.label,
                "sugar_bitter": sb,
                "sources": [],
            }
        unique[row_id]["sources"].append(sample.source)
    rows = [unique[key] for key in sorted(unique)]
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["state_id"]].append(i)
        row["sources"].sort(key=lambda source: json.dumps(source, sort_keys=True))
    interaction_groups = {
        key for key, members in groups.items() if any(rows[i]["sugar_bitter"] for i in members)
    }
    regular = sorted(set(groups) - interaction_groups)
    if len(regular) < 3:
        raise ValueError(
            "Need at least three non-interaction state groups for train/validation/test"
        )
    perm = np.random.default_rng(seed).permutation(len(regular))
    n_test = max(1, int(len(regular) * test_fraction))
    n_val = max(1, int(len(regular) * validation_fraction))
    if n_test + n_val >= len(regular):
        raise ValueError("Split fractions leave no training state groups")
    assigned = {
        regular[int(i)]: ("test" if j < n_test else "validation" if j < n_test + n_val else "train")
        for j, i in enumerate(perm)
    }
    assigned.update({key: "interaction" for key in interaction_groups})
    split_rows = {
        name: np.asarray(
            [i for i, row in enumerate(rows) if assigned[row["state_id"]] == name], dtype=np.int64
        )
        for name in SPLITS
    }
    train_rows = split_rows["train"]
    counts = Counter(int(index) for i in train_rows for index in rows[i]["indices"])
    feats = np.asarray(
        sorted(i for i, count in counts.items() if count >= min_frequency), dtype=np.int64
    )
    if not len(feats):
        raise ValueError("No features meet the training-only frequency threshold")
    positions = {int(index): j for j, index in enumerate(feats)}
    raw_x = np.zeros((len(rows), len(feats)), dtype=np.float32)
    for i, row in enumerate(rows):
        for index, value in zip(row["indices"], row["rates"]):
            if int(index) in positions:
                raw_x[i, positions[int(index)]] = np.log1p(value)
    mu = raw_x[train_rows].mean(0, dtype=np.float64).astype(np.float32)
    sd = (raw_x[train_rows].std(0, dtype=np.float64) + 1e-3).astype(np.float32)
    x = (raw_x - mu) / sd
    words = sorted({rows[i]["label"]["odor"] for i in train_rows if rows[i]["label"]["odor"]})
    word_pos = {word: i for i, word in enumerate(words)}
    word_offset = len(KEYS) * 6
    unknown_odor = word_offset + len(words)
    dopamine_offset = unknown_odor + 1
    direct = np.zeros((len(rows), dopamine_offset + 3), dtype=np.float32)
    unknown_counts = Counter()
    for i, row in enumerate(rows):
        label, vector = row["label"], direct[i]
        for stimulus in label["stimuli"]:
            key = KEYS.index(stimulus["key"])
            vector[key * 6 + SIDES.index(stimulus["side"])] = 1
            ratio = stimulus["rate_hz"] / STIMULI[stimulus["key"]].default_rate_hz
            vector[key * 6 + 3 + (0 if ratio < 0.8 else 1 if ratio <= 1.2 else 2)] = 1
        if label["odor"]:
            vector[word_offset + word_pos.get(label["odor"], len(words))] = 1
            if label["odor"] not in word_pos:
                unknown_counts[assigned[row["state_id"]]] += 1
        vector[dopamine_offset + {"reward": 1, "punish": 2}.get(label["dopamine"], 0)] = 1
    state_ids = [row["state_id"] for row in rows]
    row_ids = [row["id"] for row in rows]
    state_overlap = overlap_counts(state_ids, split_rows)
    assert not any(state_overlap.values()), state_overlap
    split_identity = {
        name: [row_ids[int(i)] for i in indices] for name, indices in split_rows.items()
    }
    gen_rows = generation_rows(split_rows["test"], n_gen, seed + 1)
    interaction_rows = np.asarray(
        [int(i) for i in split_rows["interaction"] if rows[i]["sugar_bitter"]], dtype=np.int64
    )
    # Projection may alias distinct states after training-only feature selection;
    # report this explicitly, separately from forbidden raw-state contamination.
    feature_ids = [sha256(vector.tobytes()).hexdigest() for vector in raw_x]
    source_manifest = sources or []
    manifest = {
        "schema_version": 1,
        "policy": "canonical state+caption dedup; equal model-visible states grouped; sugar+bitter groups held out",
        "config": {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "min_index": min_index,
            "min_frequency": min_frequency,
            "n_gen": n_gen,
            "split_unit": "unique model-visible state",
            "generation_seed": seed + 1,
        },
        "sources": source_manifest,
        "dataset_sha256": json_hash(source_manifest if source_manifest else sorted(row_ids)),
        "canonical_dataset_sha256": json_hash(sorted(row_ids)),
        "split_fingerprint": json_hash(split_identity),
        "counts": {
            "raw_rows": len(samples),
            "unique_rows": len(rows),
            "duplicates_removed": len(samples) - len(rows),
            "unique_state_groups": len(groups),
            "conflicting_target_state_groups": sum(len(m) > 1 for m in groups.values()),
            "split_rows": {name: len(indices) for name, indices in split_rows.items()},
            "split_groups": dict(Counter(assigned.values())),
            "interaction_metric_rows": len(interaction_rows),
            "interaction_quarantined_non_sb_rows": len(split_rows["interaction"])
            - len(interaction_rows),
        },
        "preprocessing": {
            "fit_split": "train",
            "feature_count": len(feats),
            "features_sha256": sha256(feats.tobytes()).hexdigest(),
            "mu_sha256": sha256(mu.tobytes()).hexdigest(),
            "sd_sha256": sha256(sd.tobytes()).hexdigest(),
            "transform": "log1p(rate), then (x - train_mean)/(train_std + 0.001)",
            "odor_vocabulary": words,
            "unknown_odor_column": unknown_odor,
            "unknown_odor_rows": {name: unknown_counts[name] for name in SPLITS},
            "fixed_stimulus_schema": KEYS,
        },
        "audit": {
            "state_group_overlap": state_overlap,
            "state_target_overlap": overlap_counts(row_ids, split_rows),
            "projected_feature_overlap": overlap_counts(feature_ids, split_rows),
            "sugar_bitter_train_rows": sum(rows[int(i)]["sugar_bitter"] for i in train_rows),
            "sugar_bitter_validation_rows": sum(
                rows[int(i)]["sugar_bitter"] for i in split_rows["validation"]
            ),
            "sugar_bitter_test_rows": sum(rows[int(i)]["sugar_bitter"] for i in split_rows["test"]),
        },
        "generation_test_row_ids": [row_ids[int(i)] for i in gen_rows],
        "rows": [
            {
                "id": row["id"],
                "state_id": row["state_id"],
                "split": assigned[row["state_id"]],
                "sugar_bitter": row["sugar_bitter"],
                "sources": row["sources"],
            }
            for row in rows
        ],
    }
    return {
        "X": x,
        "D": direct,
        "feats": feats,
        "mu": mu,
        "sd": sd,
        "words": words,
        "captions": [row["target"] for row in rows],
        "splits": split_rows,
        "generation_rows": gen_rows,
        "interaction_rows": interaction_rows,
        "manifest": manifest,
    }


def write_prepared(prepared, output):
    output = Path(output)
    # The caller reserves an empty directory. Never silently replace artifacts.
    manifest_path = output / "manifest.json"
    arrays_path = output / "prepared.npz"
    if manifest_path.exists() or arrays_path.exists():
        raise FileExistsError(f"Prepared artifacts already exist in {output}")
    manifest = prepared["manifest"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    with arrays_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            **{key: prepared[key] for key in ("X", "D", "feats", "mu", "sd")},
            captions=np.asarray(prepared["captions"]),
            row_ids=np.asarray([row["id"] for row in manifest["rows"]]),
            generation_rows=prepared["generation_rows"],
            interaction_metric_rows=prepared["interaction_rows"],
            **{f"{name}_rows": indices for name, indices in prepared["splits"].items()},
        )
    summary = {
        key: manifest[key]
        for key in (
            "schema_version",
            "dataset_sha256",
            "canonical_dataset_sha256",
            "split_fingerprint",
            "counts",
            "preprocessing",
            "audit",
        )
    }
    summary["prepared_npz_sha256"] = file_hash(arrays_path)
    (output / "preparation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    return summary
