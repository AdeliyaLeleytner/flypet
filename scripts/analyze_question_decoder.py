"""Strict paired analysis of the three-seed fixed-state question-decoder study.

All six models must be complete, with identical data and state-cache hashes.
Intervals resample whole counterfactual groups and remain descriptive because
this test split was already used by the preceding physiological pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import re

import numpy as np

SEEDS = (713, 714, 715)
ARMS = ("plain", "conditioned")
LABELS = ("red", "blue", "green", "yellow")
FINGERPRINTS = (
    "data_manifest_sha256",
    "features_sha256",
    "test_features_sha256",
    "model",
    "revision",
)
METRICS = (
    "accuracy",
    "earlier_object_accuracy",
    "most_recent_object_accuracy",
    "valid_fraction",
    "both_queries_correct",
    "counterfactual_both_correct",
    "counterfactual_earlier_both_correct",
    "counterfactual_most_recent_both_correct",
    "nll",
    "earlier_object_nll",
    "most_recent_object_nll",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_records(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):
        raw = raw.get("records", raw.get("predictions"))
    require(isinstance(raw, list) and bool(raw), f"no prediction records in {path}")
    return raw


def validate_records(records: list[dict], context: str = "records") -> dict[str, dict]:
    by_id = {}
    groups = defaultdict(list)
    for row in records:
        identifier = row["id"]
        require(identifier not in by_id, f"{context}: duplicate row ID {identifier}")
        require(row["truth"] in LABELS, f"{context}: unknown truth")
        require(row["query_object"] in ("key", "coin", "ring"), f"{context}: unknown query object")
        require(row["assignment"] in (0, 1), f"{context}: assignment must be 0 or 1")
        require(
            identifier == f"{row['group']}-a{row['assignment']}-{row['query_object']}",
            f"{context}: ID/metadata mismatch",
        )
        require(
            isinstance(row["valid"], bool)
            and isinstance(row["correct"], bool)
            and isinstance(row["query_is_last_mentioned"], bool),
            f"{context}: nonboolean flags",
        )
        require(
            row["valid"] == (row["prediction"] in LABELS),
            f"{context}: prediction/validity mismatch",
        )
        require(
            row["correct"] == (row["valid"] and row["prediction"] == row["truth"]),
            f"{context}: stored correctness does not match prediction and truth",
        )
        lp = np.asarray(row["answer_log_probs"], dtype=float)
        require(
            lp.shape == (4,) and np.isfinite(lp).all(),
            f"{context}: invalid answer log probabilities",
        )
        require(
            np.all(lp <= 1e-5) and np.exp(lp).sum() <= 1.0001,
            f"{context}: answer probabilities are not a valid subdistribution",
        )
        nll = float(row["first_answer_token_nll"])
        require(
            np.isfinite(nll)
            and np.isclose(nll, -lp[LABELS.index(row["truth"])], atol=1e-5, rtol=1e-5),
            f"{context}: target NLL does not match answer log probabilities",
        )
        require(
            isinstance(row["generated"], str) and isinstance(row["token_ids"], list),
            f"{context}: missing raw generation evidence",
        )
        by_id[identifier] = row
        groups[row["group"]].append(row)
    for name, group in groups.items():
        require(len(group) == 4, f"{context}: incomplete group {name}")
        require(
            count_values(group, "assignment") == [(0, 2), (1, 2)],
            f"{context}: unpaired assignments",
        )
        questions = {row["query_object"] for row in group}
        require(len(questions) == 2, f"{context}: expected two questions per history")
        for question in questions:
            pair = [row for row in group if row["query_object"] == question]
            require(
                len(pair) == 2 and len({row["query_is_last_mentioned"] for row in pair}) == 1,
                f"{context}: inconsistent paired question metadata",
            )
        require(
            sum(row["query_is_last_mentioned"] for row in group) == 2,
            f"{context}: expected earlier and last-mentioned object questions",
        )
        earlier = [row for row in group if not row["query_is_last_mentioned"]]
        later = [row for row in group if row["query_is_last_mentioned"]]
        require(
            len({row["truth"] for row in earlier}) == 2
            and len({row["truth"] for row in later}) == 1,
            f"{context}: counterfactual truth structure is invalid",
        )
    return by_id


def count_values(rows: list[dict], key: str) -> list[tuple]:
    counts = defaultdict(int)
    for row in rows:
        counts[row[key]] += 1
    return sorted(counts.items())


def validate_alignment(reference: dict[str, dict], other: dict[str, dict], context: str) -> None:
    require(set(reference) == set(other), f"{context}: row IDs do not match")
    keys = ("group", "assignment", "query_object", "query_is_last_mentioned", "truth")
    for identifier in reference:
        require(
            all(reference[identifier][key] == other[identifier][key] for key in keys),
            f"{context}: truth/question/group metadata mismatch on {identifier}",
        )


def load_evaluation_reference(data: Path, expected_manifest_sha256: str) -> dict[str, dict]:
    manifest_path = data / "manifest.json"
    require(
        sha256(manifest_path) == expected_manifest_sha256, "evaluation data manifest hash mismatch"
    )
    manifest = json.loads(manifest_path.read_text())
    path = data / "tasks/test_evaluation.jsonl"
    require(
        sha256(path) == manifest["files"]["tasks/test_evaluation.jsonl"],
        "evaluation target file hash mismatch",
    )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(len({row["id"] for row in rows}) == len(rows), "duplicate canonical evaluation ID")
    return {row["id"]: {**row, "truth": row["answer"]} for row in rows}


def validate_interventions(base: dict[str, dict], other: dict[str, dict], kind: str) -> None:
    validate_alignment(base, other, kind)
    evidence_keys = ("prediction", "valid", "generated", "token_ids", "answer_log_probs")
    if kind == "swapped_state":
        for identifier, row in other.items():
            donor_id = f"{row['group']}-a{1 - row['assignment']}-{row['query_object']}"
            donor = base[donor_id]
            require(
                all(row[key] == donor[key] for key in evidence_keys),
                f"swapped_state: {identifier} does not use its matched donor prediction",
            )
    elif kind == "mean_state":
        seen = {}
        for row in other.values():
            question = row["query_object"]
            if question in seen:
                require(
                    all(row[key] == seen[question][key] for key in evidence_keys),
                    "mean_state: same question received different constant-state output",
                )
            seen[question] = row
    else:
        raise ValueError(f"unknown intervention {kind}")


def group_metrics(records: dict[str, dict]) -> tuple[list[str], np.ndarray]:
    groups = defaultdict(list)
    for row in records.values():
        groups[row["group"]].append(row)
    names = sorted(groups)
    values = []
    for name in names:
        rows = groups[name]
        earlier = [row for row in rows if not row["query_is_last_mentioned"]]
        later = [row for row in rows if row["query_is_last_mentioned"]]
        acc = lambda subset: float(np.mean([row["correct"] for row in subset]))
        nll = lambda subset: float(np.mean([row["first_answer_token_nll"] for row in subset]))
        both_queries = np.mean(
            [all(row["correct"] for row in rows if row["assignment"] == a) for a in (0, 1)]
        )
        both_earlier, both_later = (
            all(row["correct"] for row in earlier),
            all(row["correct"] for row in later),
        )
        values.append(
            [
                acc(rows),
                acc(earlier),
                acc(later),
                np.mean([row["valid"] for row in rows]),
                both_queries,
                (both_earlier + both_later) / 2,
                both_earlier,
                both_later,
                nll(rows),
                nll(earlier),
                nll(later),
            ]
        )
    return names, np.asarray(values, dtype=float)


def metric_dict(values: np.ndarray) -> dict[str, float]:
    return dict(zip(METRICS, map(float, values)))


def bootstrap_contrast(
    values: np.ndarray, *, repetitions: int = 10000, seed: int = 20260925
) -> dict:
    """values[model_seed, paired_group, metric], already oriented as a contrast.

    The ordinary interval conditions on the trained seeds; the additional
    hierarchical interval resamples both the three seed indices and whole groups.
    Both arms use identical indices because differences are formed beforehand.
    """
    require(values.ndim == 3 and np.isfinite(values).all(), "invalid paired contrast matrix")
    n_seeds, n_groups, _ = values.shape
    require(n_groups > 0 and n_seeds > 0 and repetitions > 0, "empty bootstrap")
    rng = np.random.default_rng(seed)
    fixed, hierarchical = [], []
    seed_mean = values.mean(0)
    for start in range(0, repetitions, 250):
        n = min(250, repetitions - start)
        group_ids = rng.integers(n_groups, size=(n, n_groups))
        seed_ids = rng.integers(n_seeds, size=(n, n_seeds))
        fixed.append(seed_mean[group_ids].mean(1))
        sampled = values[seed_ids[:, :, None], group_ids[:, None, :], :]
        hierarchical.append(sampled.mean((1, 2)))
    fixed, hierarchical = np.concatenate(fixed), np.concatenate(hierarchical)
    point = values.mean((0, 1))
    result = {}
    for index, name in enumerate(METRICS):
        result[name] = {
            "difference": float(point[index]),
            "group_bootstrap_ci95": list(map(float, np.quantile(fixed[:, index], [0.025, 0.975]))),
            "seed_and_group_bootstrap_ci95": list(
                map(float, np.quantile(hierarchical[:, index], [0.025, 0.975]))
            ),
        }
    return {
        "n_seeds": n_seeds,
        "n_groups": n_groups,
        "repetitions": repetitions,
        "rng_seed": seed,
        "scope": "Descriptive intervals on a reused exploratory test split. Three seeds do not establish population-level architecture superiority.",
        "metrics": result,
    }


def analyze(results: Path, *, data_dir: Path | None = None, repetitions: int = 10000) -> dict:
    reports = defaultdict(list)
    for path in results.rglob("report.json"):
        match = re.fullmatch(r"seed(713|714|715)", path.parent.name)
        if match:
            reports[int(match.group(1))].append(path)
    require(
        set(reports) == set(SEEDS) and all(len(reports[s]) == 1 for s in SEEDS),
        "need exactly one report for each of seeds 713, 714, 715; do not aggregate missing or duplicate runs",
    )
    sources = {}
    report_data = {}
    data = {}
    reference = None
    fingerprints = None
    group_ids = None
    selected = {}
    for seed in SEEDS:
        path = reports[seed][0]
        report = json.loads(path.read_text())
        sources[str(path.resolve())] = sha256(path)
        require(
            report.get("status") == "complete" and report.get("complete") is True,
            f"seed {seed} is incomplete; failed runs cannot be dropped",
        )
        require(
            report.get("seed", report.get("config", {}).get("seed")) == seed, "report seed mismatch"
        )
        require(
            all(key in report and report[key] for key in FINGERPRINTS),
            "missing source fingerprints",
        )
        this_fingerprint = {key: report[key] for key in FINGERPRINTS}
        for key in FINGERPRINTS[:3]:
            require(
                re.fullmatch(r"[0-9a-f]{64}", this_fingerprint[key]) is not None,
                f"invalid hash fingerprint {key}",
            )
        if fingerprints is None:
            fingerprints = this_fingerprint
            if data_dir is None:
                candidates = [results.resolve() / "data"] + [
                    p / "data" for p in results.resolve().parents
                ]
                matching = [
                    p
                    for p in candidates
                    if (p / "manifest.json").is_file()
                    and sha256(p / "manifest.json") == report["data_manifest_sha256"]
                ]
                require(bool(matching), "canonical evaluation data not found; supply --data")
                data_dir = matching[0]
            reference = load_evaluation_reference(data_dir, report["data_manifest_sha256"])
            for evaluation_path in (
                data_dir / "manifest.json",
                data_dir / "tasks/test_evaluation.jsonl",
            ):
                sources[str(evaluation_path.resolve())] = sha256(evaluation_path)
        require(
            fingerprints == this_fingerprint,
            "data, initial/test state cache, model or revision differs across seeds",
        )
        require(set(report["arms"]) == set(ARMS), "both plain and conditioned arms are required")
        report_data[seed] = report
        selected[str(seed)] = {}
        for arm in ARMS:
            status = report["arms"][arm]
            require(status.get("complete") is True, f"seed {seed}/{arm} is incomplete")
            require(
                status.get("test") is not None, f"seed {seed}/{arm} has no completed test metrics"
            )
            require(
                status["updates"] == report["config"]["steps"],
                f"seed {seed}/{arm} has a truncated training run",
            )
            require(
                0 <= status["best_step"] <= status["updates"], "invalid selected checkpoint step"
            )
            require(
                status.get("validation")
                and status.get("train_selected") is not None
                and status.get("validation_selected") is not None,
                "missing selected checkpoint evidence",
            )
            selected[str(seed)][arm] = {
                key: status[key]
                for key in (
                    "updates",
                    "best_step",
                    "validation",
                    "train_selected",
                    "validation_selected",
                )
            }
            data[seed, arm] = {}
            for intervention, filename in (
                ("test", "test.json"),
                ("mean_state", "mean_state.json"),
                ("swapped_state", "swapped_state.json"),
            ):
                prediction_path = path.parent / arm / filename
                records = validate_records(
                    load_records(prediction_path), f"{seed}/{arm}/{intervention}"
                )
                sources[str(prediction_path.resolve())] = sha256(prediction_path)
                validate_alignment(reference, records, f"{seed}/{arm}/{intervention}")
                groups, values = group_metrics(records)
                if group_ids is None:
                    group_ids = groups
                require(groups == group_ids, "group order mismatch")
                require(
                    len(records) == 288 and len(groups) == 72,
                    "expected all 288 rows in 72 whole groups",
                )
                if intervention == "test":
                    base_records = records
                else:
                    validate_interventions(base_records, records, intervention)
                data[seed, arm][intervention] = values
    # Equal total exposures across both architecture arms and all seeds.
    configs = [report_data[s]["config"] for s in SEEDS]
    require(
        len({(config["steps"], config["batch_size"]) for config in configs}) == 1,
        "optimizer exposure differs across seeds",
    )
    metrics = {
        str(seed): {
            arm: {
                condition: metric_dict(values.mean(0))
                for condition, values in data[seed, arm].items()
            }
            for arm in ARMS
        }
        for seed in SEEDS
    }
    contrast = np.stack(
        [data[seed, "conditioned"]["test"] - data[seed, "plain"]["test"] for seed in SEEDS]
    )
    paired = bootstrap_contrast(contrast, repetitions=repetitions)
    paired["orientation"] = (
        "conditioned minus plain; positive accuracy is better, negative NLL is better"
    )
    effects = {}
    for arm in ARMS:
        effects[arm] = {}
        for condition in ("mean_state", "swapped_state"):
            values = np.stack(
                [data[seed, arm]["test"] - data[seed, arm][condition] for seed in SEEDS]
            )
            effect = bootstrap_contrast(values, repetitions=repetitions)
            effect["orientation"] = (
                f"intact minus {condition}; positive accuracy or negative NLL means intact states perform better"
            )
            effects[arm][condition] = effect
    return {
        "status": "complete",
        "all_six_models_complete": True,
        "seeds": list(SEEDS),
        "n_test_rows_per_model": len(reference),
        "n_paired_groups": len(group_ids),
        "test_status": "Exploratory reuse of the exact previous pilot test split; not fresh confirmation.",
        "evidence_scope": "Frozen initial encoder/core states, fixed model and data; tests the decoder question interface, not trained-connectome superiority or a new memory mechanism.",
        "source_fingerprints": fingerprints,
        "source_files_sha256": sources,
        "selected_checkpoints": selected,
        "per_seed": metrics,
        "aggregate": {
            arm: {
                condition: metric_dict(
                    np.stack([data[s, arm][condition] for s in SEEDS]).mean((0, 1))
                )
                for condition in ("test", "mean_state", "swapped_state")
            }
            for arm in ARMS
        },
        "per_seed_conditioned_minus_plain": {
            str(seed): metric_dict(contrast[i].mean(0)) for i, seed in enumerate(SEEDS)
        },
        "per_seed_paired_intervals": {
            str(seed): bootstrap_contrast(contrast[i : i + 1], repetitions=repetitions)
            for i, seed in enumerate(SEEDS)
        },
        "conditioned_minus_plain": paired,
        "state_effects": effects,
        "metric_definitions": {
            "accuracy": "Strict free generation of one allowed color, eight-token budget; invalid outputs count as errors.",
            "nll": "Negative first-answer-token log probability for recipient truth; all four color log probabilities retained.",
            "both_queries_correct": "Both questions on a particular history are correct, averaged across both assignments.",
            "counterfactual_both_correct": "Both assignments are answered correctly for the same queried object, averaged over both objects.",
            "bootstrap": "Paired whole counterfactual groups; the same sampled group and seed indices apply to both arms.",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--data",
        type=Path,
        help="Canonical local data directory; otherwise discovered above results.",
    )
    args = parser.parse_args()
    result = analyze(args.results, data_dir=args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temp = args.out.with_suffix(args.out.suffix + ".tmp")
    temp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temp.replace(args.out)
    print(
        json.dumps(
            {"aggregate": result["aggregate"], "contrast": result["conditioned_minus_plain"]},
            indent=2,
        )
    )
