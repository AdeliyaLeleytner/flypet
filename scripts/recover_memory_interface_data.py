#!/usr/bin/env python3
"""Recover complete history prefixes without changing frozen simulator sources.

Original interrupted shards are read-only. New completed shards hard-link their
referenced immutable raw/checkpoint artifacts and rebuild features from raw rates.
Separate fixed plans rerun only the unfinished history suffixes from reset.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypet.experiments import ROOT, sha256_file, verify_artifacts, write_json
from flypet.memory_interface_data import check_example_splits, pet_fingerprints


def subset_plan(plan, histories):
    result = copy.deepcopy(plan)
    result["histories"] = histories
    result["expected_histories"] = len(histories)
    result["expected_examples"] = len(histories) * len(plan["protocol"]["odors"])
    n_naive = (
        len(plan["probe_seed_rule"]["split_namespaces"])
        * len(plan["protocol"]["seeds"])
        * len(plan["protocol"]["odors"])
    )
    result["expected_records"] = n_naive + sum(
        len(h["steps"]) + len(plan["protocol"]["odors"]) for h in histories
    )
    return result


def prepare(dataset):
    dataset = Path(dataset)
    evidence = dataset / "recovery_evidence"
    evidence.mkdir(exist_ok=False)
    for name in ("progress.json", "orchestration.json"):
        if (dataset / name).exists():
            shutil.copy2(dataset / name, evidence / ("before-" + name))
    summary = {
        "status": "prepared",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reason": "Original process sessions disappeared between task turns; no simulation processes survived. Exact termination cause was not recorded.",
        "policy": "Preserve original shards. Retain complete prefixes and rerun only unfinished history suffixes from reset using unchanged protocol and seeds.",
        "simulation_sources_unchanged": True,
        "shards": [],
        "merge_labels": [],
    }
    for label in ("part0", "part1", "part2"):
        src = dataset / "shards" / label
        plan = json.loads((src / "plan.json").read_text())
        manifest = json.loads((src / "manifest.json").read_text())
        for name, info in manifest["files"].items():
            if sha256_file(ROOT / name) != info["sha256"]:
                raise AssertionError(f"Frozen source/data changed: {name}")
        if manifest["pet_before"] != pet_fingerprints():
            raise AssertionError("Personal pet changed before recovery")
        records = [json.loads(line) for line in (src / "records.jsonl").read_text().splitlines()]
        examples = [json.loads(line) for line in (src / "examples.jsonl").read_text().splitlines()]
        post_by_history = {}
        for r in records:
            if r["phase"] == "post":
                post_by_history.setdefault(r["history"], []).append(r)
        prefix = []
        for h in plan["histories"]:
            rs = post_by_history.get(h["history_id"], [])
            if len(rs) != len(plan["protocol"]["odors"]):
                break
            if sorted(r["metadata"]["probe_index"] for r in rs) != list(
                range(len(plan["protocol"]["odors"]))
            ):
                raise AssertionError("Duplicate/missing probe in completed prefix")
            prefix.append(h)
        if not prefix or len(prefix) == len(plan["histories"]):
            raise AssertionError("Expected a nonempty incomplete original shard")
        ids = {h["history_id"] for h in prefix}
        keep_records = [r for r in records if r["phase"] == "naive" or r["history"] in ids]
        keep_examples = [e for e in examples if e["history_id"] in ids]
        completed_plan = subset_plan(plan, prefix)
        remaining_plan = subset_plan(plan, plan["histories"][len(prefix) :])
        if (
            len(keep_records) != completed_plan["expected_records"]
            or len(keep_examples) != completed_plan["expected_examples"]
        ):
            raise AssertionError("Completed prefix count mismatch")
        complete_label, remaining_label = label + "_completed", label + "_remaining"
        dst = dataset / "shards" / complete_label
        dst.mkdir(exist_ok=False)
        write_json(dst / "plan.json", completed_plan)
        for name in (
            "protocol.json",
            "validation.json",
            "feature_schema.json",
            "input_profiles.json",
            "brain_root_ids.npy",
        ):
            shutil.copy2(src / name, dst / name)
        paths = set()
        for r in keep_records:
            for key in ("raw", "vnc", "memory_before", "memory_after"):
                if r.get(key):
                    paths.add(r[key]["path"])
        for relative in sorted(paths):
            target = dst / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(src / relative, target)
        for name, rows in [("records.jsonl", keep_records), ("examples.jsonl", keep_examples)]:
            with (dst / name).open("w") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        schema = json.loads((src / "feature_schema.json").read_text())
        indices = np.arange(schema["n_stim"], schema["n_neurons"], dtype=np.int32)
        rates = np.zeros((len(keep_examples), len(indices)), dtype=np.float32)
        for j, e in enumerate(keep_examples):
            with np.load(dst / e["raw"]["path"], allow_pickle=False) as z:
                mask = z["idx"] >= schema["n_stim"]
                rates[j, z["idx"][mask] - schema["n_stim"]] = z["rates_hz"][mask]
        np.savez_compressed(
            dst / "features.npz",
            rates=rates,
            neuron_indices=indices,
            example_ids=np.asarray([e["example_id"] for e in keep_examples]),
            mbon_rates=np.asarray([e["mbon_rates_hz"] for e in keep_examples], dtype=np.float32),
            mbon_indices=np.asarray(schema["mbon_indices"], dtype=np.int32),
        )
        del rates
        verification = verify_artifacts(dst)
        if not verification["ok"]:
            raise AssertionError(verification["errors"])
        log = Path(f"/tmp/fly-memory-interface-v2-{label}.log")
        elapsed = None
        if log.exists():
            shutil.copy2(log, evidence / f"{label}-interrupted.log")
            for line in log.read_text().splitlines():
                if line.startswith('{"history_complete"'):
                    item = json.loads(line)
                    if item["history_complete"] in ids:
                        elapsed = item["elapsed_s"]
        if elapsed is None:
            raise AssertionError("Missing observed elapsed time for completed prefix")
        provenance = {
            "original_shard": str(src.relative_to(dataset)),
            "original_manifest_sha256": sha256_file(src / "manifest.json"),
            "original_records_sha256": sha256_file(src / "records.jsonl"),
            "original_examples_sha256": sha256_file(src / "examples.jsonl"),
            "included_records": len(keep_records),
            "included_examples": len(keep_examples),
            "excluded_incomplete_records": len(records) - len(keep_records),
            "excluded_incomplete_examples": len(examples) - len(keep_examples),
            "features_reconstructed_from_saved_raw_rates": True,
            "raw_and_memory_artifacts_hardlinked_without_modification": True,
            "exact_individual_peak_rss_unavailable": "Original processes exited without /usr/bin/time final statistics; sampled combined RSS is preserved in before-progress.json.",
            "original_observed_elapsed_s_through_last_complete_history": elapsed,
        }
        manifest.update(
            status="complete",
            actual_records=len(keep_records),
            expected_records={"MI": completed_plan["expected_records"]},
            actual_histories=len(prefix),
            expected_histories=len(prefix),
            actual_examples=len(keep_examples),
            expected_examples=len(keep_examples),
            plan_sha256=sha256_file(dst / "plan.json"),
            split_example_counts=check_example_splits(keep_examples),
            elapsed_s=elapsed,
            pet_after=pet_fingerprints(),
            pet_unchanged=True,
            recovery=provenance,
        )
        manifest["artifacts"] = {
            name: {"sha256": sha256_file(dst / name), "bytes": (dst / name).stat().st_size}
            for name in (
                "examples.jsonl",
                "features.npz",
                "feature_schema.json",
                "input_profiles.json",
                "records.jsonl",
                "verification.json",
            )
        }
        write_json(dst / "manifest.json", manifest)
        remaining_path = evidence / f"{remaining_label}-plan.json"
        write_json(remaining_path, remaining_plan)
        row = {
            "original_label": label,
            "completed_label": complete_label,
            "remaining_label": remaining_label,
            "completed_histories": len(prefix),
            "remaining_histories": len(remaining_plan["histories"]),
            "remaining_plan": str(remaining_path.relative_to(dataset)),
            **provenance,
        }
        summary["shards"].append(row)
        summary["merge_labels"].extend([complete_label, remaining_label])
        print(
            json.dumps(
                {
                    "prepared": label,
                    "completed_histories": len(prefix),
                    "remaining_histories": len(remaining_plan["histories"]),
                }
            ),
            flush=True,
        )
    summary["physical_expected_records"] = sum(
        json.loads((dataset / "shards" / r["completed_label"] / "plan.json").read_text())[
            "expected_records"
        ]
        + json.loads((dataset / r["remaining_plan"]).read_text())["expected_records"]
        for r in summary["shards"]
    )
    summary["logical_expected_records"] = json.loads((dataset / "plan.json").read_text())[
        "expected_records"
    ]
    summary["duplicate_naive_reference_records"] = (
        summary["physical_expected_records"] - summary["logical_expected_records"]
    )
    write_json(evidence / "recovery.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset), indent=2))
