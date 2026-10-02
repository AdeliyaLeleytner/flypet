"""Post-hoc fixed ridge reference on the shared initial physiological states.

This reproduces the existing ridge_lambda10 recipe without tuning. It is a
question-specific numerical classifier, not an LLM generation condition, a
performance ceiling, or an exposure-matched architecture comparison.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
OBJECTS = ("key", "coin", "ring")
LABELS = ("red", "blue", "green", "yellow")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def original_recipe_reference(source, features):
    """Execute the unchanged original main() prefix, stopping before its MLP.

    This supplies a regression oracle from the actual prior source rather than
    comparing two calls to the new implementation. The source prefix has been
    inspected: it loads the stated feature file, normalizes training data, and
    fits the three fixed ridge systems. It performs no file writes or searches.
    """
    tree = ast.parse(Path(source).read_text())
    main = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    stop = next(
        i
        for i, node in enumerate(main.body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "model" for target in node.targets)
    )
    body = main.body[:stop]
    require(
        any(isinstance(node, ast.For) for node in body), "prior ridge loop could not be located"
    )
    body.append(
        ast.Return(
            value=ast.Tuple(
                elts=[
                    ast.Name(id=name, ctx=ast.Load())
                    for name in ("train_logits", "val_logits", "report", "mean", "std")
                ],
                ctx=ast.Load(),
            )
        )
    )
    function = ast.FunctionDef(name="reference", args=main.args, body=body, decorator_list=[])
    # Preserve the original metric implementation used in the prior report.
    metric_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "metrics"
    )
    module = ast.fix_missing_locations(
        ast.Module(body=[metric_function, function], type_ignores=[])
    )
    namespace = {"torch": torch, "np": np, "nn": nn, "Path": Path, "json": json, "hashlib": hashlib}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["reference"](SimpleNamespace(features=str(features)))


def fit_ridge(x, q, y):
    # Exact original recipe: float32 Torch normalization, then float64 ridge.
    mean = x.mean(0)
    std = x.std(0).clamp_min(0.01)
    normalized = ((x - mean) / std).clamp(-10, 10)
    weights = []
    for obj in range(3):
        xx = torch.cat([normalized[q == obj], torch.ones(int((q == obj).sum()), 1)], 1).double()
        target = nn.functional.one_hot(y[q == obj], 4).double()
        reg = torch.eye(xx.shape[1], dtype=torch.float64) * 10
        reg[-1, -1] = 0
        weights.append(torch.linalg.solve(xx.T @ xx + reg, xx.T @ target))
    return mean, std, weights


def predict_scores(x, q, mean, std, weights):
    normalized = ((x - mean) / std).clamp(-10, 10)
    scores = torch.zeros(len(x), 4)
    for obj in range(3):
        xx = torch.cat([normalized[q == obj], torch.ones(int((q == obj).sum()), 1)], 1).double()
        scores[q == obj] = (xx @ weights[obj]).float()
    return scores


def records_from_scores(rows, scores):
    prediction = scores.argmax(1).tolist()
    records = []
    for row, pred, value in zip(rows, prediction, scores.tolist()):
        records.append(
            {
                key: row[key]
                for key in ("id", "group", "assignment", "query_object", "query_is_last_mentioned")
            }
            | {
                "truth": row["answer"],
                "prediction": LABELS[pred],
                "correct": LABELS[pred] == row["answer"],
                "ridge_scores": value,
            }
        )
    return records


def summarize(records):
    metrics = {"n": len(records)}
    for name, rows in (
        ("all", records),
        ("earlier_object", [r for r in records if not r["query_is_last_mentioned"]]),
        ("most_recent_object", [r for r in records if r["query_is_last_mentioned"]]),
    ):
        metrics[name] = {
            "n": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
        }
    histories, pairs = {}, {}
    for row in records:
        histories.setdefault((row["group"], row["assignment"]), []).append(row)
        pairs.setdefault((row["group"], row["query_object"]), []).append(row)
    require(all(len(rows) == 2 for rows in histories.values()), "incomplete history/question pair")
    require(all(len(rows) == 2 for rows in pairs.values()), "incomplete counterfactual pair")
    metrics["both_queries_correct"] = sum(
        all(r["correct"] for r in rows) for rows in histories.values()
    ) / len(histories)
    metrics["counterfactual_both_correct"] = sum(
        all(r["correct"] for r in rows) for rows in pairs.values()
    ) / len(pairs)
    return metrics


def controls(rows, x, q, mean, std, weights):
    scores = predict_scores(x, q, mean, std, weights)
    normal = records_from_scores(rows, scores)
    constant = records_from_scores(
        rows, predict_scores(mean[None].expand(len(x), -1), q, mean, std, weights)
    )
    lookup = {
        (row["group"], row["assignment"], row["query_object"]): i for i, row in enumerate(rows)
    }
    donor_indices = [
        lookup[(row["group"], 1 - row["assignment"], row["query_object"])] for row in rows
    ]
    swapped = records_from_scores(rows, scores[donor_indices])
    for row, donor in zip(swapped, donor_indices):
        row["donor_id"] = rows[donor]["id"]
    return {
        name: {"metrics": summarize(records), "records": records}
        for name, records in (
            ("intact", normal),
            ("mean_state", constant),
            ("swapped_state", swapped),
        )
    }


def evaluate(args):
    started = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    torch.set_num_threads(4)
    torch.manual_seed(713)
    data = args.data
    manifest = json.loads((data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        require(sha(data / name) == expected, f"prepared data changed: {name}")
    with np.load(data / "initial_features.npz", allow_pickle=False) as z:
        cache = {key: z[key] for key in z.files}
    with np.load(args.legacy_features, allow_pickle=False) as z:
        legacy = {key: z[key] for key in z.files}
    require(
        sha(args.legacy_features) == manifest["source_hashes"]["results/initial_state_probe.npz"],
        "legacy source cache hash differs",
    )
    prior = json.loads(args.legacy_analysis.read_text())
    require(
        prior["features_sha256"] == sha(args.legacy_features),
        "prior analysis refers to different features",
    )
    for key, value in legacy.items():
        require(
            key in cache and np.array_equal(value, cache[key]),
            f"new/legacy feature or metadata mismatch: {key}",
        )
    rows = {
        split: read_rows(data / "tasks" / f"{split}.jsonl") for split in ("train", "validation")
    }
    rows["test"] = read_rows(data / "tasks/test_evaluation.jsonl")
    for split in ("train", "validation"):
        require(
            cache[split + "_id"].tolist() == [row["id"] for row in rows[split]],
            f"{split} ID mismatch",
        )
        require(
            cache[split + "_q"].tolist()
            == [OBJECTS.index(row["query_object"]) for row in rows[split]],
            "question mismatch",
        )
        require(
            cache[split + "_y"].tolist() == [LABELS.index(row["answer"]) for row in rows[split]],
            "label mismatch",
        )
    marker = json.loads((data / "test_cache_ready.json").read_text())
    require(sha(data / "test_features.npz") == marker["sha256"], "shared test cache hash mismatch")
    require(
        marker["test_input_sha256"] == sha(data / "tasks/test_features.jsonl"),
        "test cache input changed",
    )
    with np.load(data / "test_features.npz", allow_pickle=False) as z:
        test_x, test_ids = z["x"], z["id"].tolist()
    require(
        test_ids == [row["id"] for row in rows["test"]],
        "shared test cache/canonical evaluation ID mismatch",
    )
    x = {s: torch.from_numpy(cache[s + "_x"]) for s in ("train", "validation")}
    x["test"] = torch.from_numpy(test_x)
    q = {s: torch.tensor([OBJECTS.index(row["query_object"]) for row in rows[s]]) for s in rows}
    y = torch.from_numpy(cache["train_y"])
    mean, std, weights = fit_ridge(x["train"], q["train"], y)
    train_scores = predict_scores(x["train"], q["train"], mean, std, weights)
    validation_scores = predict_scores(x["validation"], q["validation"], mean, std, weights)
    ref_train, ref_validation, reference_report, ref_mean, ref_std = original_recipe_reference(
        args.recipe_source, args.legacy_features
    )
    regression = {
        "source_prefix_executed_without_its_mlp": True,
        "training_predictions_exact": torch.equal(train_scores.argmax(1), ref_train.argmax(1)),
        "validation_predictions_exact": torch.equal(
            validation_scores.argmax(1), ref_validation.argmax(1)
        ),
        "training_score_max_abs_difference": float((train_scores - ref_train).abs().max()),
        "validation_score_max_abs_difference": float(
            (validation_scores - ref_validation).abs().max()
        ),
        "mean_exact": torch.equal(mean, ref_mean),
        "std_exact": torch.equal(std, ref_std),
        "original_report_metrics": prior["models"]["ridge_lambda10"],
        "source_recomputed_metrics": reference_report["models"]["ridge_lambda10"],
    }
    old_matches = all(
        np.isclose(
            reference_report["models"]["ridge_lambda10"][split][key],
            prior["models"]["ridge_lambda10"][split][key],
            atol=1e-6,
            rtol=1e-6,
        )
        for split in ("train", "validation")
        for key in prior["models"]["ridge_lambda10"][split]
    )
    regression["prior_train_validation_metrics_reproduced"] = bool(old_matches)
    regression["passed"] = bool(
        old_matches
        and regression["training_predictions_exact"]
        and regression["validation_predictions_exact"]
        and regression["mean_exact"]
        and regression["std_exact"]
    )
    report = {
        "status": "complete" if regression["passed"] else "regression_failed",
        "scope": "Post-hoc fixed numerical ridge reference, not LLM generation, a ceiling, or an exposure-matched architecture comparison.",
        "decision_timing": "Requested after the first seed's primary LLM test outcome was inspected; existing ridge_lambda10 recipe reused without tuning or model selection.",
        "test_status": "Exploratory reuse of the same previously evaluated test split.",
        "gpu_used": False,
        "device": "cpu",
        "torch_version": torch.__version__,
        "created_at_utc": created_at,
        "recipe": {
            "lambda": 10,
            "heads": "one four-score linear head per queried object",
            "intercept_penalized": False,
            "normalization": "Original torch.float32 train mean/sample std, floor .01, clip [-10,10]; linear systems solved in float64 and scores cast to float32.",
            "fitted_coefficients": sum(weight.numel() for weight in weights),
            "hyperparameter_search": False,
        },
        "normalizer_rounding_relative_to_softprefix_cache": {
            "mean_exact": np.array_equal(mean.numpy(), cache["train_mean"]),
            "std_exact": np.array_equal(std.numpy(), cache["train_std"]),
            "mean_max_abs_difference": float(np.max(np.abs(mean.numpy() - cache["train_mean"]))),
            "std_max_abs_difference": float(np.max(np.abs(std.numpy() - cache["train_std"]))),
        },
        "regression": regression,
        "metric_scope": "Cross-model references should compare accuracy. The archived ridge cross-entropy is softmax over four uncalibrated ridge scores; it is not the LLM full-vocabulary first-token NLL.",
        "score_label_order": list(LABELS),
        "source_hashes": {
            **{
                f"data/{name}": sha(data / name)
                for name in (
                    "manifest.json",
                    "initial_features.npz",
                    "test_features.npz",
                    "test_cache_ready.json",
                    "tasks/train.jsonl",
                    "tasks/validation.jsonl",
                    "tasks/test_features.jsonl",
                    "tasks/test_evaluation.jsonl",
                )
            },
            "legacy/initial_state_probe.npz": sha(args.legacy_features),
            "legacy/initial_state_probe_analysis.json": sha(args.legacy_analysis),
            "scripts/analyze_physiology_probe.py": sha(args.recipe_source),
            "scripts/evaluate_question_decoder_ridge.py": sha(Path(__file__)),
        },
        "weights_sha256": [tensor_sha(weight) for weight in weights],
        "mean_sha256": tensor_sha(mean),
        "std_sha256": tensor_sha(std),
    }
    # Operational pointers are retained locally for source verification; paper
    # receipts consume only the anonymous keys and hashes above.
    report["source_file_paths"] = {
        f"data/{name}": str((data / name).resolve())
        for name in (
            "manifest.json",
            "initial_features.npz",
            "test_features.npz",
            "test_cache_ready.json",
            "tasks/train.jsonl",
            "tasks/validation.jsonl",
            "tasks/test_features.jsonl",
            "tasks/test_evaluation.jsonl",
        )
    }
    report["source_file_paths"].update(
        {
            "legacy/initial_state_probe.npz": str(args.legacy_features.resolve()),
            "legacy/initial_state_probe_analysis.json": str(args.legacy_analysis.resolve()),
            "scripts/analyze_physiology_probe.py": str(args.recipe_source.resolve()),
            "scripts/evaluate_question_decoder_ridge.py": str(Path(__file__).resolve()),
        }
    )
    if regression["passed"]:
        report["train"] = summarize(records_from_scores(rows["train"], train_scores))
        report["validation"] = controls(
            rows["validation"], x["validation"], q["validation"], mean, std, weights
        )
        report["test"] = controls(rows["test"], x["test"], q["test"], mean, std, weights)
    else:
        report["reason"] = (
            "The original train/validation recipe did not reproduce; no test metrics were computed or repaired by tuning."
        )
    report["elapsed_seconds"] = time.monotonic() - started
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--legacy-features", type=Path, required=True)
    parser.add_argument("--legacy-analysis", type=Path, required=True)
    parser.add_argument(
        "--recipe-source", type=Path, default=ROOT / "scripts/analyze_physiology_probe.py"
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temp.replace(args.out)
    print(
        json.dumps(
            {
                "status": report["status"],
                "regression": report["regression"],
                "validation": {
                    key: value["metrics"] for key, value in report.get("validation", {}).items()
                },
                "test": {key: value["metrics"] for key, value in report.get("test", {}).items()},
            },
            indent=2,
        )
    )
    if report["status"] != "complete":
        raise SystemExit(1)
