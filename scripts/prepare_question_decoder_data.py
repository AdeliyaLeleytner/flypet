"""Audit and package fixed initial brain states for the question-decoder experiment.

No representation is refitted here. Features are copied verbatim from the previous
pilot; normalization is fitted to training rows only. The previously evaluated
test split is exploratory reuse, not a fresh confirmatory holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import numpy as np

OBJECTS = ("key", "coin", "ring")
LOCATIONS = ("red", "blue", "green", "yellow")
SPLITS = ("train", "validation", "test")
PILOT = "results/20260924T152840Z-2c5587/pilot"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def checkpoint_state_digest(path: Path) -> str:
    # Match the original pilot's tensor-content digest, distinct from file SHA.
    import torch

    state = torch.load(path, map_location="cpu", weights_only=True)
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_dataset(rows_by_split: dict[str, list[dict]]) -> dict:
    """Verify row truth, whole counterfactual groups, and disjoint partitions."""
    seen_ids, seen_groups, seen_histories = set(), set(), set()
    result = {}
    for split in SPLITS:
        rows = rows_by_split[split]
        groups = defaultdict(list)
        ids, histories = set(), set()
        for row in rows:
            require(row["split"] == split, f"wrong split on {row['id']}")
            require(row["id"] not in ids, f"duplicate row ID: {row['id']}")
            ids.add(row["id"])
            require(row["query_object"] in OBJECTS, "unknown question object")
            require(row["query"] == f"Where is the {row['query_object']} now?", "question mismatch")
            events = row["events"]
            require(len(events) == 4, "expected four events")
            require(
                all(obj in OBJECTS and loc in LOCATIONS for obj, loc in events), "unknown event"
            )
            history = [f"The {obj} moved to the {loc} box." for obj, loc in events]
            require(row["history"] == history, "history/event mismatch")
            truth = dict(events)[row["query_object"]]
            require(row["answer"] == truth, f"incorrect target on {row['id']}")
            require(
                row["query_is_last_mentioned"] == (row["query_object"] == events[-1][0]),
                "last-mentioned metadata mismatch",
            )
            require(
                row["id"] == f"{row['group']}-a{row['assignment']}-{row['query_object']}",
                "row ID and metadata mismatch",
            )
            groups[row["group"]].append(row)
            histories.add(tuple(history))
        require(not ids & seen_ids, "row IDs cross partitions")
        require(not set(groups) & seen_groups, "counterfactual groups cross partitions")
        require(not histories & seen_histories, "exact histories cross partitions")
        for group, group_rows in groups.items():
            require(len(group_rows) == 4, f"incomplete group {group}")
            assignments = {a: [r for r in group_rows if r["assignment"] == a] for a in (0, 1)}
            for pair in assignments.values():
                require(len(pair) == 2, "each history must have two questions")
                require(
                    pair[0]["history"] == pair[1]["history"],
                    "same assignment has different histories",
                )
                require(
                    len({r["query_object"] for r in pair}) == 2, "duplicate question in history"
                )
            first, second = assignments[0][0]["events"], assignments[1][0]["events"]
            require(
                Counter(map(tuple, first)) == Counter(map(tuple, second)),
                "paired event multisets differ",
            )
            require(
                first[0] == second[2]
                and first[2] == second[0]
                and first[1] == second[1]
                and first[3] == second[3],
                "counterfactual target moves were not swapped",
            )
        balance = Counter(r["answer"] for r in rows)
        require(
            set(balance) == set(LOCATIONS) and len(set(balance.values())) == 1,
            f"unbalanced answers in {split}",
        )
        result[split] = {
            "rows": len(rows),
            "groups": len(groups),
            "unique_histories": len(histories),
            "answer_counts": dict(sorted(balance.items())),
        }
        seen_ids.update(ids)
        seen_groups.update(groups)
        seen_histories.update(histories)
    return result


def audit_probe_alignment(
    arrays: dict[str, np.ndarray], rows_by_split: dict[str, list[dict]]
) -> dict:
    """Recover row IDs only after checking every saved alignment field.

    The legacy cache had no IDs. For the distractor question, answer and group do
    not identify the counterfactual assignment. Equality to the paired target
    question's feature vector additionally checks that ambiguity. If two states
    are identical this distinction has no numerical effect on the experiment.
    """
    expected_keys = {
        f"{split}_{suffix}"
        for split in ("train", "validation")
        for suffix in ("x", "y", "q", "last", "group")
    }
    require(set(arrays) == expected_keys, "cache must contain only known train/validation arrays")
    report = {}
    for split in ("train", "validation"):
        rows = rows_by_split[split]
        x = arrays[split + "_x"]
        require(x.ndim == 2 and x.shape[0] == len(rows), f"feature shape mismatch in {split}")
        require(np.isfinite(x).all(), f"nonfinite features in {split}")
        expected = {
            "y": np.array([LOCATIONS.index(r["answer"]) for r in rows]),
            "q": np.array([OBJECTS.index(r["query_object"]) for r in rows]),
            "last": np.array([r["query_is_last_mentioned"] for r in rows]),
            "group": np.array([r["group"] for r in rows]),
        }
        for suffix, values in expected.items():
            require(
                np.array_equal(arrays[f"{split}_{suffix}"], values),
                f"cached {split}_{suffix} is not aligned to original rows",
            )
        pairs = defaultdict(list)
        for index, row in enumerate(rows):
            pairs[row["group"], row["assignment"]].append(index)
        for indices in pairs.values():
            require(
                len(indices) == 2 and np.array_equal(x[indices[0]], x[indices[1]]),
                f"paired questions have misaligned physiological features in {split}",
            )
        report[split] = {
            "metadata_exact_match": True,
            "same_history_features_exact_match": True,
            "rows": len(rows),
            "features": x.shape[1],
        }
    require(arrays["train_x"].shape[1] == arrays["validation_x"].shape[1], "feature width mismatch")
    return report


def train_normalization(
    train_x: np.ndarray, std_floor: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    require(train_x.ndim == 2 and len(train_x) > 1, "need at least two training states")
    require(np.isfinite(train_x).all(), "nonfinite training states")
    mean = train_x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = train_x.astype(np.float64).std(axis=0, ddof=1)
    return mean, np.maximum(std, std_floor).astype(np.float32)


def split_test_payload(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    # Whitelists ensure new label fields cannot silently reach the exporter.
    feature_keys = ("id", "history", "query", "query_object")
    evaluation_keys = (
        "id",
        "group",
        "assignment",
        "query",
        "query_object",
        "answer",
        "query_is_last_mentioned",
        "split",
    )
    return (
        [{key: row[key] for key in feature_keys} for row in rows],
        [{key: row[key] for key in evaluation_keys} for row in rows],
    )


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare(source: Path, destination: Path) -> dict:
    source, destination = source.resolve(), destination.resolve()
    require(
        source != destination and source not in destination.parents,
        "source run must stay read-only",
    )
    report_path = source / PILOT / "report.json"
    report = json.loads(report_path.read_text())
    task_manifest_path = source / "data/task/manifest.json"
    task_manifest = json.loads(task_manifest_path.read_text())
    graph_manifest_path = source / "data/graph/manifest.json"
    graph_manifest = json.loads(graph_manifest_path.read_text())
    graph_path = source / "data/graph/graph.npz"
    checkpoint_path = source / PILOT / "initial.pt"
    probe_path = source / "results/initial_state_probe.npz"
    probe_metadata_path = source / "results/initial_state_probe_metadata.json"
    probe_metadata = json.loads(probe_metadata_path.read_text())
    seal_path = source / PILOT / "SHA256.json"
    seal = json.loads(seal_path.read_text())
    require(
        probe_metadata["source"] == "initial.pt"
        and probe_metadata["splits"] == ["train", "validation"],
        "cache did not originate from the initial train/validation state",
    )
    require(
        probe_metadata.get("settle_ticks", 0) == 0,
        "settling-tail cache is a different representation",
    )
    require(sha256(checkpoint_path) == seal["initial.pt"], "initial checkpoint file hash mismatch")
    require(
        checkpoint_state_digest(checkpoint_path) == report["initial_bridge_sha256"],
        "initial checkpoint tensor digest mismatch",
    )
    for name in (
        "report.json",
        "initial_state_probe.npz",
        "initial_state_probe_metadata.json",
        "export_physiology_probe.py",
    ):
        require(
            sha256(source / PILOT / name) == seal[name], f"sealed pilot artifact mismatch: {name}"
        )
    require(
        sha256(graph_path)
        == graph_manifest["graph_sha256"]
        == report["graph_manifest"]["graph_sha256"],
        "graph hash mismatch",
    )
    rows_by_split = {}
    for split in SPLITS:
        path = source / "data/task" / f"{split}.jsonl"
        require(
            sha256(path)
            == task_manifest["files"][path.name]
            == report["task_manifest"]["files"][path.name],
            f"source {split} hash mismatch",
        )
        rows_by_split[split] = load_rows(path)
    data_audit = audit_dataset(rows_by_split)
    with np.load(probe_path, allow_pickle=False) as cache:
        arrays = {key: cache[key] for key in cache.files}
    alignment = audit_probe_alignment(arrays, rows_by_split)
    # The copied, sealed pilot cache and top-level analysis cache must be identical.
    require(
        sha256(probe_path) == sha256(source / PILOT / "initial_state_probe.npz"),
        "top-level probe differs from original pilot artifact",
    )
    for split in ("train", "validation"):
        arrays[split + "_id"] = np.asarray([row["id"] for row in rows_by_split[split]])
    arrays["train_mean"], arrays["train_std"] = train_normalization(arrays["train_x"])
    require(not destination.exists() or not any(destination.iterdir()), "destination is not empty")
    destination.mkdir(parents=True, exist_ok=True)
    tasks = destination / "tasks"
    tasks.mkdir()
    for split in ("train", "validation"):
        shutil.copy2(source / "data/task" / f"{split}.jsonl", tasks / f"{split}.jsonl")
    test_features, test_evaluation = split_test_payload(rows_by_split["test"])
    for name, rows in (("test_features", test_features), ("test_evaluation", test_evaluation)):
        (tasks / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    np.savez_compressed(destination / "initial_features.npz", **arrays)
    shutil.copy2(graph_path, destination / "graph.npz")
    shutil.copy2(checkpoint_path, destination / "initial.pt")
    config = {
        "config": report["config"],
        "model": report["model"],
        "revision": report["revision"],
        "embedding_std": report["embedding_std"],
        "graph_manifest": graph_manifest,
        "initial_bridge_sha256": report["initial_bridge_sha256"],
        "initial_file_sha256": sha256(checkpoint_path),
        "graph_sha256": graph_manifest["graph_sha256"],
        "source_sha256": report["source_sha256"],
        "objects": list(OBJECTS),
        "labels": list(LOCATIONS),
        "features": probe_metadata["features"],
        "settle_ticks": 0,
    }
    write_json(destination / "source_config.json", config)
    source_paths = [
        report_path,
        task_manifest_path,
        graph_manifest_path,
        graph_path,
        checkpoint_path,
        seal_path,
        probe_path,
        probe_metadata_path,
        source / PILOT / "export_physiology_probe.py",
    ]
    source_paths += [source / "data/task" / f"{split}.jsonl" for split in SPLITS]
    manifest = {
        "format_version": 1,
        "scope": "Fixed INITIAL physiological state; question-decoder comparison; same states across decoder arms.",
        "test_status": "Exploratory reuse: this exact test split was already evaluated by the prior physiology pilot; not a fresh confirmatory holdout.",
        "source_hashes": {str(path.relative_to(source)): sha256(path) for path in source_paths},
        "dataset_audit": data_audit,
        "cache_alignment_audit": alignment,
        "row_id_provenance": "Added from original JSONL after exact group/question/answer/last-mentioned and within-history feature checks; legacy cache did not store row IDs.",
        "normalization": {
            "fit_split": "train",
            "n_rows": len(arrays["train_x"]),
            "std_ddof": 1,
            "std_floor": 0.01,
            "clip": [-10.0, 10.0],
            "arrays": ["train_mean", "train_std"],
            "formula": "clip((raw_features - train_mean) / train_std, -10, 10)",
            "features_stored_raw": True,
        },
        "test_separation": {
            "feature_export_input": "tasks/test_features.jsonl",
            "evaluation_only": "tasks/test_evaluation.jsonl",
            "explicit_answer_label_field_present": False,
            "history_contains_task_facts": True,
            "protection": "Supervision/evaluation answer field excluded; history is legitimate brain input, never sent to the LM in primary arms.",
            "test_features_present_in_initial_cache": False,
        },
        "files": {
            str(path.relative_to(destination)): sha256(path)
            for path in sorted(destination.rglob("*"))
            if path.is_file()
        },
    }
    write_json(destination / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.source, args.out)
    print(
        json.dumps(
            {
                "dataset_audit": result["dataset_audit"],
                "files": result["files"],
                "test_status": result["test_status"],
            },
            indent=2,
        )
    )
