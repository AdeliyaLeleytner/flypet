"""Train-only preprocessing of the continuing-runtime calibration observations.

No stimulus/history text, random seeds, memory weights or scalar readouts become
reader inputs. The initial task is current-state readout from downstream neurons.
The development split is exploratory: the parent calibration was inspected.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np

from .neural_records import file_hash, array_hash, write_json

FIELDS = ("approach_hz", "avoid_hz", "valence")
CHANNELS = [f"log_rate_bin_{i}" for i in range(5)] + [
    "log_mean_rate",
    "voltage_above_rest",
    "signed_log_synaptic",
    "refractory",
]
VALIDATION_FAMILIES = {"pair01-reward_a", "pair03-punish_a", "pair05-opposed", "pair07-naive"}
TEST_FAMILIES = {"pair00-opposed", "pair02-reward_a", "pair04-naive", "pair06-punish_a"}


def split_for(family):
    return (
        "validation"
        if family in VALIDATION_FAMILIES
        else "development_test"
        if family in TEST_FAMILIES
        else "train"
    )


def read_phase(neural, phase, indices, *, dt_ms=0.1):
    end = int(neural["phase_end_tick"][phase])
    start = int(neural["phase_end_tick"][phase - 1]) if phase else 0
    ticks, neurons = neural["spike_tick"], neural["spike_neuron_index"]
    duration_s = (end - start) * dt_ms / 1000
    lookup = np.full(neural["voltage_mv"].shape[1], -1, dtype=np.int32)
    lookup[indices] = np.arange(len(indices))
    chosen = (ticks >= start) & (ticks < end)
    pos = lookup[neurons[chosen]]
    relative = ticks[chosen] - start
    keep = pos >= 0
    pos, relative = pos[keep], relative[keep]
    bins = np.minimum(4, relative * 5 // (end - start))
    counts = np.bincount(pos * 5 + bins, minlength=len(indices) * 5).reshape(len(indices), 5)
    features = np.zeros((len(indices), len(CHANNELS)), dtype=np.float32)
    features[:, :5] = np.log1p(counts / (duration_s / 5))
    features[:, 5] = np.log1p(counts.sum(axis=1) / duration_s)
    features[:, 6] = (neural["voltage_mv"][phase, indices] + 52) / 7
    g = neural["synaptic_mv"][phase, indices] / 20
    features[:, 7] = np.sign(g) * np.log1p(np.abs(g))
    features[:, 8] = neural["refractory_remaining_ms"][phase, indices] / 2.2
    return features


def target_caption(values):
    return '{"approach_hz":%.1f,"avoid_hz":%.1f,"valence":%.3f}' % tuple(float(v) for v in values)


def parse_reply(text):
    import re

    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group())
        if set(obj) != set(FIELDS) or any(type(obj[k]) not in (int, float) for k in FIELDS):
            return None
        values = np.array([obj[k] for k in FIELDS], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values[:2] < 0) or abs(values[2]) > 1:
            return None
        return values.tolist()
    except (ValueError, TypeError, KeyError):
        return None


def score(texts, targets):
    parsed = [parse_reply(x) for x in texts]
    valid = [i for i, x in enumerate(parsed) if x is not None]
    actual = np.asarray(targets, dtype=np.float64)
    return {
        "n": len(texts),
        "valid": len(valid),
        "valid_fraction": len(valid) / max(1, len(texts)),
        "mae_valid": dict(
            zip(
                FIELDS,
                np.abs(np.asarray([parsed[i] for i in valid]) - actual[valid])
                .mean(axis=0)
                .tolist(),
            )
        )
        if valid
        else None,
    }


def prepare(source, output, *, max_neurons=4096):
    from . import connectome as C

    source, output = Path(source), Path(output)
    manifest = json.loads((source / "manifest.json").read_text())
    verification = json.loads((source / "verification.json").read_text())
    if manifest.get("status") != "complete" or not verification.get("ok"):
        raise ValueError("Reader preparation requires a completed verified dataset")
    if output.exists():
        raise FileExistsError(output)
    roots = np.load(source / "root_ids.npy", allow_pickle=False)
    forbidden = np.load(source / "input_model_indices.npy", allow_pickle=False)
    eligible = np.ones(len(roots), dtype=bool)
    eligible[forbidden] = False
    examples = []
    usage = np.zeros(len(roots), dtype=np.int64)
    for folder in sorted((source / "episodes").iterdir()):
        row = json.loads((folder / "episode.json").read_text())
        partition = row.get("split", split_for(row["family"]))
        if partition not in ("train", "validation", "development_test"):
            raise ValueError("Unknown source split")
        for name in ("episode.json", "neural.npz"):
            relative = str((folder / name).relative_to(source))
            if file_hash(folder / name) != manifest["files"][relative]:
                raise ValueError("Source checksum mismatch: " + relative)
        with np.load(folder / "neural.npz", allow_pickle=False) as z:
            if partition == "train":
                for phase in (0, 4, 5):
                    end = z["phase_end_tick"][phase]
                    start = z["phase_end_tick"][phase - 1] if phase else 0
                    mask = (z["spike_tick"] >= start) & (z["spike_tick"] < end)
                    usage += np.bincount(z["spike_neuron_index"][mask], minlength=len(roots))
        for phase in (0, 4, 5):
            examples.append(
                {
                    "id": f"{row['episode']:04d}-phase{phase}",
                    "episode": row["episode"],
                    "family": row["family"],
                    "split": partition,
                    "phase": phase,
                    "neural_file": str((folder / "neural.npz").relative_to(source)),
                }
            )
    candidates = np.flatnonzero(eligible & (usage > 0))
    if not len(candidates):
        raise ValueError("No active downstream training neurons")
    ranked = candidates[np.argsort(-usage[candidates], kind="stable")]
    selected = np.sort(ranked[:max_neurons]).astype(np.int32)
    annotations = C.annotations().reindex(roots)
    is_mbon = (annotations.cell_class.astype(str) == "MBON").to_numpy()
    nt = annotations.top_nt.astype(str).to_numpy()
    ct = annotations.cell_type.astype(str).to_numpy()
    ap = np.flatnonzero(is_mbon & ((nt == "acetylcholine") | (ct == "MBON11")))
    av = np.flatnonzero(is_mbon & (nt == "glutamate"))
    features, targets = [], []
    opened = None
    current = None
    try:
        for row in examples:
            if row["neural_file"] != current:
                if opened is not None:
                    opened.close()
                current = row["neural_file"]
                opened = np.load(source / current, allow_pickle=False)
                arrays = {key: opened[key] for key in opened.files}
            phase = row["phase"]
            features.append(
                read_phase(arrays, phase, selected, dt_ms=manifest["environment"]["dt_ms"])
            )
            end = arrays["phase_end_tick"][phase]
            start = arrays["phase_end_tick"][phase - 1] if phase else 0
            chosen = (arrays["spike_tick"] >= start) & (arrays["spike_tick"] < end)
            counts = np.bincount(arrays["spike_neuron_index"][chosen], minlength=len(roots))
            duration_s = (end - start) * manifest["environment"]["dt_ms"] / 1000
            approach, avoid = (
                float(counts[ap].sum() / duration_s),
                float(counts[av].sum() / duration_s),
            )
            targets.append([approach, avoid, (approach - avoid) / (approach + avoid + 1)])
    finally:
        if opened is not None:
            opened.close()
    features = np.stack(features)
    targets = np.asarray(targets, dtype=np.float32)
    # Exact repeated observable states are assigned only to the earliest split.
    # Mostly quiet controls; retain their rows as explicitly excluded diagnostics.
    seen = {}
    order = sorted(
        range(len(examples)),
        key=lambda i: {"train": 0, "validation": 1, "development_test": 2}[examples[i]["split"]],
    )
    duplicate_exclusions = []
    for i in order:
        key = array_hash(features[i])
        row = examples[i]
        if key in seen and examples[seen[key]]["split"] != row["split"]:
            row["original_split"] = row["split"]
            row["split"] = "duplicate_control"
            duplicate_exclusions.append({"id": row["id"], "matches": examples[seen[key]]["id"]})
        else:
            seen.setdefault(key, i)
    train = np.array([i for i, r in enumerate(examples) if r["split"] == "train"])
    mean = features[train].mean(axis=0)
    std = np.maximum(features[train].std(axis=0), 0.05)
    normalized = np.clip((features - mean) / std, -10, 10).astype(np.float32)
    # The actual model input, including clipping/float32 rounding, also has to
    # be free of exact cross-split duplicates.
    seen_normalized = {}
    for i in order:
        row = examples[i]
        if row["split"] == "duplicate_control":
            continue
        key = array_hash(normalized[i])
        if key in seen_normalized and examples[seen_normalized[key]]["split"] != row["split"]:
            row["original_split"] = row["split"]
            row["split"] = "duplicate_control"
            duplicate_exclusions.append(
                {
                    "id": row["id"],
                    "matches": examples[seen_normalized[key]]["id"],
                    "stage": "normalized",
                }
            )
        else:
            seen_normalized.setdefault(key, i)
    meta, vocab = [], {}
    for field in ("cell_class", "top_nt", "side"):
        values = annotations.iloc[selected][field].astype(str).fillna("").tolist()
        names = sorted(set(values))
        table = {v: i for i, v in enumerate(names)}
        vocab[field] = names
        meta.append([table[v] for v in values])
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "features.npz",
        x=normalized,
        targets=targets,
        neuron_indices=selected,
        neuron_root_ids=roots[selected],
        metadata=np.array(meta, dtype=np.int64).T,
        mean=mean,
        std=std,
    )
    for i, row in enumerate(examples):
        row["target"] = dict(zip(FIELDS, map(float, targets[i])))
        row["caption"] = target_caption(targets[i])
    (output / "examples.jsonl").write_text("".join(json.dumps(r) + "\n" for r in examples))
    report = {
        "status": "complete",
        "task": "current_state_latent_readout",
        "scope": manifest.get("scope", "exploratory development on inspected calibration"),
        "source_manifest_sha256": file_hash(source / "manifest.json"),
        "source_directory": source.name,
        "counts": dict(Counter(r["split"] for r in examples)),
        "channels": CHANNELS,
        "n_neurons": len(selected),
        "metadata_vocabulary": vocab,
        "preprocessing_fit": "train only",
        "writable_inputs_excluded": True,
        "memory_weights_in_input": False,
        "recipe_text_in_input": False,
        "neural_indices_overlap_writable": int(np.isin(selected, forbidden).sum()),
        "duplicate_exclusions": duplicate_exclusions,
        "validation_families": sorted(
            {r["family"] for r in examples if r["split"] == "validation"}
        ),
        "development_test_families": sorted(
            {r["family"] for r in examples if r["split"] == "development_test"}
        ),
        "files": {name: file_hash(output / name) for name in ("features.npz", "examples.jsonl")},
        "preparation_code_sha256": {
            str(Path(__file__).relative_to(Path(__file__).parents[1])): file_hash(Path(__file__))
        },
    }
    write_json(output / "manifest.json", report)
    return report
