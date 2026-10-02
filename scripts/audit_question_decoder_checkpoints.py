"""Posthoc CPU audit of question-channel use in six completed decoder fits.

Uses cached states and checkpoints only, without targets or answer generation.
Prefix sensitivity is a mechanistic diagnostic, not a new quality evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

# Support the documented ``python /path/to/scripts/...py`` command from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from flypet.question_decoder import CachedStateQuestionDecoder, QUERY_OBJECTS


SEEDS = (713, 714, 715)
ARMS = ("plain", "conditioned")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = CachedStateQuestionDecoder.from_checkpoint(checkpoint).eval().requires_grad_(False)
    require(
        all(torch.isfinite(value).all() for value in model.state_dict().values()),
        f"Nonfinite checkpoint tensor: {path}",
    )
    return model


def structure(model):
    state = model.state_dict()
    return {
        "state_tensor_count": len(state),
        "state_scalar_count": sum(value.numel() for value in state.values()),
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "tensor_shapes": {name: list(value.shape) for name, value in state.items()},
        "features": model.feature_dim,
        "prefix_tokens": model.tokens,
        "embedding_dim": model.embedding_dim,
        "conditioned": model.conditioned,
        "normalized": model.normalized,
    }


def query_columns(model):
    return model.decoder[0].weight[:, model.feature_dim :].detach().double()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.size > 0 and np.isfinite(values).all(), "Invalid sensitivity values")
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(values.max()),
    }


def prefix_sensitivity(model, features, *, batch_size=16):
    """Change query identity while holding every raw physiological state fixed."""
    require(
        features.ndim == 2 and len(features) > 0 and features.shape[1] == model.feature_dim,
        "Cached state shape is incompatible with decoder",
    )
    require(np.isfinite(features).all(), "Nonfinite cached states")
    pairs = ((0, 1), (0, 2), (1, 2))
    changes = {pair: [] for pair in pairs}
    equal = {pair: [] for pair in pairs}
    bf16_changes = {pair: [] for pair in pairs}
    bf16_equal = {pair: [] for pair in pairs}
    bf16_component_fraction = {pair: [] for pair in pairs}
    base_rms = []
    with torch.no_grad():
        for offset in range(0, len(features), batch_size):
            x = torch.from_numpy(features[offset : offset + batch_size]).float()
            prefixes = [
                model(x, torch.full((len(x),), query, dtype=torch.long))
                for query in range(len(QUERY_OBJECTS))
            ]
            require(all(torch.isfinite(prefix).all() for prefix in prefixes), "Nonfinite prefix")
            # teacher_inputs/inference_inputs cast these prefixes to the frozen
            # Qwen embedding dtype. Measure that actual interface as well.
            lm_prefixes = [prefix.to(torch.bfloat16) for prefix in prefixes]
            require(
                all(torch.isfinite(prefix).all() for prefix in lm_prefixes),
                "Nonfinite bfloat16 prefix",
            )
            base_rms.extend(prefixes[0].double().square().mean((1, 2)).sqrt().tolist())
            for pair in pairs:
                a, b = (prefixes[index] for index in pair)
                changes[pair].extend(
                    (a.double() - b.double()).square().mean((1, 2)).sqrt().tolist()
                )
                equal[pair].extend((a == b).flatten(1).all(1).tolist())
                a_lm, b_lm = (lm_prefixes[index] for index in pair)
                component_changed = (a_lm != b_lm).flatten(1)
                bf16_equal[pair].extend((~component_changed.any(1)).tolist())
                bf16_component_fraction[pair].extend(component_changed.double().mean(1).tolist())
                bf16_changes[pair].extend(
                    (a_lm.double() - b_lm.double()).square().mean((1, 2)).sqrt().tolist()
                )
    result = {}
    for pair in pairs:
        name = QUERY_OBJECTS[pair[0]] + "_vs_" + QUERY_OBJECTS[pair[1]]
        result[name] = {
            "prefix_difference_rms": distribution(changes[pair]),
            "exactly_equal_state_count": sum(equal[pair]),
            "changed_state_count": len(features) - sum(equal[pair]),
            "lm_bfloat16": {
                "prefix_difference_rms": distribution(bf16_changes[pair]),
                "exactly_equal_state_count": sum(bf16_equal[pair]),
                "changed_state_count": len(features) - sum(bf16_equal[pair]),
                "changed_component_fraction": distribution(bf16_component_fraction[pair]),
            },
        }
    return {
        "rows": len(features),
        "feature_shape": list(features.shape),
        "key_query_prefix_rms": distribution(base_rms),
        "query_pairs": result,
    }


def audit(results: Path, data: Path, *, batch_size=16):
    require(batch_size > 0, "batch_size must be positive")
    reports = {}
    for path in results.rglob("report.json"):
        match = re.fullmatch(r"seed(713|714|715)", path.parent.name)
        if match:
            seed = int(match.group(1))
            require(
                seed not in reports, f"Duplicate seed {seed}; pass only the final complete campaign"
            )
            reports[seed] = path
    require(set(reports) == set(SEEDS), "All three seed reports are required")
    fingerprints = {
        "data_manifest_sha256": sha(data / "manifest.json"),
        "features_sha256": sha(data / "initial_features.npz"),
        "test_features_sha256": sha(data / "test_features.npz"),
    }
    with np.load(data / "initial_features.npz", allow_pickle=False) as arrays:
        train = arrays["train_x"].copy()
        train_mean, train_std = arrays["train_mean"].copy(), arrays["train_std"].copy()
    with np.load(data / "test_features.npz", allow_pickle=False) as arrays:
        test = arrays["x"].copy()
        require(len(arrays["id"]) == len(test), "Test state/ID count mismatch")
    cache = {"train": train, "test": test}
    report_data = {}
    # Complete-fit and integrity checks precede every sensitivity computation.
    for seed in SEEDS:
        path = reports[seed]
        report = json.loads(path.read_text())
        require(
            report.get("seed") == seed
            and report.get("complete") is True
            and report.get("status") == "complete",
            f"Incomplete seed {seed}",
        )
        require(
            all(report.get(key) == value for key, value in fingerprints.items()),
            f"Cached states or manifest differ for seed {seed}",
        )
        require(set(report["arms"]) == set(ARMS), f"Missing paired arm for seed {seed}")
        for arm in ARMS:
            record = report["arms"][arm]
            require(
                record.get("complete") is True and record["updates"] == report["config"]["steps"],
                f"Incomplete fit {seed}/{arm}",
            )
            require(
                sha(path.parent / arm / "best.pt") == record["checkpoint_sha256"],
                f"Checkpoint hash mismatch: {seed}/{arm}",
            )
            require(
                record["initial_decoder_sha256"] == report["initial_decoder_sha256"],
                f"Different initialization: {seed}/{arm}",
            )
        report_data[seed] = report
    require(
        len({(r["config"]["steps"], r["config"]["batch_size"]) for r in report_data.values()}) == 1,
        "Training exposures differ across seeds",
    )
    output = {
        "status": "complete",
        "all_six_models_complete": True,
        "scope": "Posthoc query-channel and prefix-sensitivity diagnostic on all cached rows; no labels or generated answers used. Does not establish quality or capacity advantage.",
        "audit_script_sha256": sha(Path(__file__).resolve()),
        "lm_input_dtype": "torch.bfloat16",
        "changed_component_fraction_definition": "Fraction of prefix token-by-embedding components unequal after both prefixes are cast to bfloat16; summarized across states. Surviving differences do not imply useful effects on generation.",
        "cache_sha256": fingerprints,
        "cache_counts": {
            name: {
                "rows": len(x),
                "shape": list(x.shape),
                "unique_feature_vectors": len(np.unique(x, axis=0)),
            }
            for name, x in cache.items()
        },
        "seeds": {},
        "torch_version": torch.__version__,
        "device": "cpu",
    }
    for seed in SEEDS:
        directory, report = reports[seed].parent, report_data[seed]
        initial_path = directory / "initial_decoder.pt"
        initial = load_model(initial_path)
        require(
            tensor_digest(initial.state_dict()) == report["initial_decoder_sha256"],
            f"Initial tensor hash mismatch: {seed}",
        )
        require(
            torch.count_nonzero(query_columns(initial)).item() == 0,
            f"Initial query columns are nonzero: {seed}",
        )
        initial.conditioned = True
        initial_probe = prefix_sensitivity(
            initial, train[: min(16, len(train))], batch_size=batch_size
        )
        require(
            all(x["changed_state_count"] == 0 for x in initial_probe["query_pairs"].values()),
            f"Initial prefix depends on query: {seed}",
        )
        entry = {
            "report_sha256": sha(reports[seed]),
            "initial_checkpoint_sha256": sha(initial_path),
            "initial_tensor_sha256": tensor_digest(initial.state_dict()),
            "initial_query_columns_exactly_zero": True,
            "initial_prefix_probe": initial_probe,
            "arms": {},
        }
        for arm in ARMS:
            path = directory / arm / "best.pt"
            model = load_model(path)
            require(
                model.conditioned == (arm == "conditioned"), f"Wrong restored arm: {seed}/{arm}"
            )
            require(model.normalized, f"Missing shared normalization: {seed}/{arm}")
            for name, expected in (("feature_mean", train_mean), ("feature_std", train_std)):
                require(
                    torch.equal(getattr(model, name), torch.from_numpy(expected).float()),
                    f"Normalization changed: {seed}/{arm}/{name}",
                )
            columns = query_columns(model)
            if arm == "plain":
                require(
                    torch.count_nonzero(columns).item() == 0, f"Plain query columns changed: {seed}"
                )
            changes = {
                name: prefix_sensitivity(model, x, batch_size=batch_size)
                for name, x in cache.items()
            }
            if arm == "plain":
                require(
                    all(
                        pair["changed_state_count"] == 0
                        for split in changes.values()
                        for pair in split["query_pairs"].values()
                    ),
                    f"Plain prefix uses query: {seed}",
                )
                require(
                    all(
                        pair["lm_bfloat16"]["changed_state_count"] == 0
                        for split in changes.values()
                        for pair in split["query_pairs"].values()
                    ),
                    f"Plain bfloat16 prefix uses query: {seed}",
                )
            entry["arms"][arm] = {
                "checkpoint_sha256": sha(path),
                "state_tensor_sha256": tensor_digest(model.state_dict()),
                "structure": structure(model),
                "query_columns_shape": list(columns.shape),
                "query_columns_l2": float(columns.square().sum().sqrt()),
                "query_column_l2": dict(
                    zip(QUERY_OBJECTS, columns.square().sum(0).sqrt().tolist())
                ),
                "query_columns_max_abs": float(columns.abs().max()),
                "sensitivity": changes,
            }
        output["seeds"][str(seed)] = entry
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    result = audit(args.results, args.data, batch_size=args.batch_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"status": result["status"], "cache_counts": result["cache_counts"]}))
