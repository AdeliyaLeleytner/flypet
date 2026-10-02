#!/usr/bin/env python3
"""Merge complete independent shards of one frozen memory-interface protocol.

Raw arrays and checkpoints remain byte-for-byte in their shard directories.
Only record IDs and relative paths are prefixed in the merged indices.  Repeated
naive runs remain visible physical records and supply a cross-process audit.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypet.experiments import ROOT, protocol_sha256, sha256_file, verify_artifacts, write_json
from flypet.memory_interface_data import check_example_splits, pet_fingerprints


def artifact_prefix(artifact, prefix):
    if artifact:
        artifact["path"] = str(prefix / artifact["path"])


def prefix_record(record, label):
    r = copy.deepcopy(record)
    prefix = Path("shards") / label
    r["record_id"] = f"{label}-{r['record_id']}"
    for key in ("raw", "vnc", "memory_before", "memory_after"):
        artifact_prefix(r.get(key), prefix)
    r["execution_shard"] = label
    return r


def prefix_example(example, label):
    e = copy.deepcopy(example)
    prefix = Path("shards") / label
    e["record_id"] = f"{label}-{e['record_id']}"
    e["state_npz"] = str(prefix / e["state_npz"])
    artifact_prefix(e["raw"], prefix)
    artifact_prefix(e["checkpoint"], prefix)
    e["naive_reference"]["record_id"] = f"{label}-{e['naive_reference']['record_id']}"
    artifact_prefix(e["naive_reference"]["raw"], prefix)
    e["execution_shard"] = label
    return e


def merge(output, labels):
    output = Path(output)
    if any(
        (output / name).exists() for name in ("records.jsonl", "examples.jsonl", "features.npz")
    ):
        raise FileExistsError("Merged indices already exist; refusing overwrite")
    started = time.monotonic()
    plan = json.loads((output / "plan.json").read_text())
    manifests, plans, examples, naive_groups = [], [], [], defaultdict(list)
    rates, mbon_rates, feature_ids = [], [], []
    neuron_indices = mbon_indices = None
    n_records = 0
    expected_history_ids = [h["history_id"] for h in plan["histories"]]
    observed_history_ids = []
    for label in labels:
        shard = output / "shards" / label
        manifest = json.loads((shard / "manifest.json").read_text())
        shard_plan = json.loads((shard / "plan.json").read_text())
        if manifest["status"] != "complete" or not manifest["pet_unchanged"]:
            raise ValueError(f"Shard is incomplete or changed pet: {label}")
        if protocol_sha256(shard_plan["protocol"]) != plan["protocol_sha256"]:
            raise ValueError(f"Scientific protocol differs in {label}")
        if manifests and manifest["files"] != manifests[0]["files"]:
            raise ValueError(f"Simulation source/data hashes differ in {label}")
        manifests.append(manifest)
        plans.append(shard_plan)
        observed_history_ids.extend(h["history_id"] for h in shard_plan["histories"])
        for name in (
            "feature_schema.json",
            "input_profiles.json",
            "brain_root_ids.npy",
            "validation.json",
        ):
            if not (output / name).exists():
                shutil.copy2(shard / name, output / name)
            elif sha256_file(shard / name) != sha256_file(output / name):
                raise ValueError(f"Shared schema or neuron order differs: {label}/{name}")
        with (output / "records.jsonl").open("a") as dest, (shard / "records.jsonl").open() as src:
            for line in src:
                r = prefix_record(json.loads(line), label)
                dest.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")
                n_records += 1
                if r["phase"] == "naive":
                    key = (r["metadata"]["history_split"], r["compound"], r["seed"])
                    naive_groups[key].append(r)
        shard_examples = []
        with (
            (output / "examples.jsonl").open("a") as dest,
            (shard / "examples.jsonl").open() as src,
        ):
            for line in src:
                e = prefix_example(json.loads(line), label)
                dest.write(json.dumps(e, ensure_ascii=False, allow_nan=False) + "\n")
                shard_examples.append(e)
        examples.extend(shard_examples)
        with np.load(shard / "features.npz", allow_pickle=False) as z:
            if list(z["example_ids"]) != [e["example_id"] for e in shard_examples]:
                raise ValueError(f"Feature/example row mismatch in {label}")
            if neuron_indices is None:
                neuron_indices, mbon_indices = z["neuron_indices"], z["mbon_indices"]
            elif not (
                np.array_equal(z["neuron_indices"], neuron_indices)
                and np.array_equal(z["mbon_indices"], mbon_indices)
            ):
                raise ValueError(f"Feature indices differ in {label}")
            rates.append(z["rates"])
            mbon_rates.append(z["mbon_rates"])
            feature_ids.append(z["example_ids"])
    if observed_history_ids != expected_history_ids or len(set(observed_history_ids)) != len(
        expected_history_ids
    ):
        raise AssertionError("Histories missing, duplicated, or out of original protocol order")
    if len(examples) != plan["expected_examples"] or len(
        {e["example_id"] for e in examples}
    ) != len(examples):
        raise AssertionError("Unexpected or duplicate examples")
    split_counts = check_example_splits(examples)
    heldout = plan["heldout_compound"]
    for e in examples:
        if e["split"] in ("train", "validation", "test") and (
            heldout == e["probe_compound"] or heldout in e["training_compounds"]
        ):
            raise AssertionError("Held-out chemical leaked into an ordinary split")
    joined_rates = np.concatenate(rates)
    if not np.isfinite(joined_rates).all():
        raise AssertionError("Nonfinite rates")
    joined_mbon = np.concatenate(mbon_rates)
    mbon_positions = np.searchsorted(neuron_indices, mbon_indices)
    if not np.array_equal(neuron_indices[mbon_positions], mbon_indices) or not np.array_equal(
        joined_rates[:, mbon_positions], joined_mbon
    ):
        raise AssertionError("Downstream feature matrix and named MBON rates disagree")
    schema = json.loads((output / "feature_schema.json").read_text())
    signs = np.asarray(schema["mbon_signs"])
    approach = joined_mbon[:, signs > 0].sum(axis=1)
    avoid = joined_mbon[:, signs < 0].sum(axis=1)
    readout = (approach - avoid) / (approach + avoid + 1.0)
    targets = np.asarray(
        [[e["target"][k] for k in ("approach_hz", "avoid_hz", "valence")] for e in examples]
    )
    if not (
        np.array_equal(approach, targets[:, 0])
        and np.array_equal(avoid, targets[:, 1])
        and np.allclose(readout, targets[:, 2], atol=1e-7, rtol=0)
    ):
        raise AssertionError("MBON readout does not reproduce saved targets")
    np.savez_compressed(
        output / "features.npz",
        rates=joined_rates,
        neuron_indices=neuron_indices,
        example_ids=np.concatenate(feature_ids),
        mbon_rates=joined_mbon,
        mbon_indices=mbon_indices,
    )
    del rates, joined_rates
    # All simulation shards run identical naive references independently. These are
    # reproducibility repetitions, not extra examples or biological replicates.
    comparisons = []
    for key, records in sorted(naive_groups.items()):
        if len(records) != len(labels):
            raise AssertionError(f"Missing naive replica: {key}")
        with np.load(output / records[0]["raw"]["path"], allow_pickle=False) as z:
            reference_times = z["spike_times_s"]
        equal_counts = len({r["raw"]["count_sha256"] for r in records}) == 1
        equal_times = True
        for r in records[1:]:
            with np.load(output / r["raw"]["path"], allow_pickle=False) as z:
                ts = z["spike_times_s"]
                equal_times = (
                    equal_times
                    and ts.shape == reference_times.shape
                    and bool(np.allclose(ts, reference_times, atol=1e-9, rtol=0))
                )
        comparisons.append(
            {
                "history_split": key[0],
                "compound": key[1],
                "seed": key[2],
                "record_ids": [r["record_id"] for r in records],
                "equal_counts": equal_counts,
                "equal_times_atol_1ns": equal_times,
            }
        )
    audit = {
        "n_groups": len(comparisons),
        "n_replicas_per_group": len(labels),
        "n_comparisons_to_first": len(comparisons) * (len(labels) - 1),
        "all_equal_counts": all(r["equal_counts"] for r in comparisons),
        "all_equal_times_atol_1ns": all(r["equal_times_atol_1ns"] for r in comparisons),
        "groups": comparisons,
    }
    expected_naive_groups = (
        len(plan["probe_seed_rule"]["split_namespaces"])
        * len(plan["protocol"]["seeds"])
        * len(plan["protocol"]["odors"])
    )
    if len(comparisons) != expected_naive_groups:
        raise AssertionError("Unexpected naive reproducibility-group count")
    write_json(output / "naive_reproducibility.json", audit)
    if not audit["all_equal_counts"] or not audit["all_equal_times_atol_1ns"]:
        raise AssertionError("Cross-process naive reference mismatch")
    verification = verify_artifacts(output)
    if not verification["ok"]:
        raise AssertionError(verification["errors"])
    manifest = copy.deepcopy(manifests[0])
    physical_expected = sum(p["expected_records"] for p in plans)
    if n_records != physical_expected:
        raise AssertionError("Physical record count mismatch")
    manifest.update(
        {
            "status": "complete",
            "execution": "independent_cpu_process_shards",
            "plan_sha256": sha256_file(output / "plan.json"),
            "expected_records": {"MI": physical_expected},
            "actual_records": n_records,
            "logical_expected_records": plan["expected_records"],
            "duplicate_naive_reference_records": n_records - plan["expected_records"],
            "expected_histories": plan["expected_histories"],
            "actual_histories": len(observed_history_ids),
            "expected_examples": plan["expected_examples"],
            "actual_examples": len(examples),
            "split_example_counts": split_counts,
            "merge_elapsed_s": time.monotonic() - started,
            "mbon_readout_verification": {
                "n_examples": len(examples),
                "features_equal_named_mbon_rates": True,
                "exact_approach_and_avoid_rates": True,
                "valence_atol_1e7": True,
            },
            "shards": [],
            "pet_after": pet_fingerprints(),
        }
    )
    manifest["pet_unchanged"] = all(m["pet_unchanged"] for m in manifests) and all(
        m["pet_before"] == manifest["pet_after"] for m in manifests
    )
    command_starts = []
    for label, m in zip(labels, manifests):
        shard = output / "shards" / label
        row = {
            "label": label,
            "manifest": str(Path("shards") / label / "manifest.json"),
            "manifest_sha256": sha256_file(shard / "manifest.json"),
            "elapsed_s": m["elapsed_s"],
            "actual_records": m["actual_records"],
            "actual_histories": m["actual_histories"],
            "actual_examples": m["actual_examples"],
        }
        log = Path(f"/tmp/fly-memory-interface-v2-{label}.log")
        if m.get("recovery"):
            row["recovery"] = m["recovery"]
            original_label = Path(m["recovery"]["original_shard"]).name
            log = Path(f"/tmp/fly-memory-interface-v2-{original_label}.log")
        if log.exists():
            # macOS birth time is when shell redirection created the process log,
            # before command execution.  It supplies the actual parallel wall
            # interval including launch offsets and merge, not a sum of CPU jobs.
            if hasattr(log.stat(), "st_birthtime"):
                command_starts.append(log.stat().st_birthtime)
            shutil.copy2(log, shard / "execution.log")
            match = re.search(r"^\s*(\d+)\s+maximum resident set size\s*$", log.read_text(), re.M)
            row["maximum_resident_set_size_bytes"] = int(match[1]) if match else None
            row["execution_log"] = str(Path("shards") / label / "execution.log")
        manifest["shards"].append(row)
    if len(command_starts) == len(labels):
        manifest["execution_started_unix"] = min(command_starts)
        manifest["execution_finished_unix"] = time.time()
        manifest["elapsed_s"] = (
            manifest["execution_finished_unix"] - manifest["execution_started_unix"]
        )
        manifest["elapsed_s_definition"] = (
            "First shard shell launch through completed merge, including launch offsets and orchestration waits; partial superseded runs excluded."
        )
    else:
        manifest.pop("elapsed_s", None)
    if (output / "progress.json").exists():
        manifest["resource_observer"] = json.loads((output / "progress.json").read_text())
    manifest["resource_pauses"] = (
        json.loads((output / "resource_pauses.json").read_text())
        if (output / "resource_pauses.json").exists()
        else []
    )
    recovery_path = output / "recovery_evidence" / "recovery.json"
    if recovery_path.exists():
        manifest["recovery"] = json.loads(recovery_path.read_text())
        manifest["recovery"]["metadata_sha256"] = sha256_file(recovery_path)
        manifest["recovery"]["before_resource_observer"] = json.loads(
            (recovery_path.parent / "before-progress.json").read_text()
        )
    manifest["artifacts"] = {
        name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
        for name in (
            "examples.jsonl",
            "features.npz",
            "feature_schema.json",
            "input_profiles.json",
            "records.jsonl",
            "naive_reproducibility.json",
            "verification.json",
        )
    }
    manifest["merge_tool"] = {
        "path": "scripts/merge_memory_interface_data.py",
        "sha256": sha256_file(Path(__file__)),
    }
    manifest["simulation_source_snapshot"] = []
    for name, info in manifest["files"].items():
        if name.endswith(".py"):
            source = ROOT / name
            if sha256_file(source) != info["sha256"]:
                raise AssertionError(f"Frozen simulation source changed during execution: {name}")
            target = output / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            manifest["simulation_source_snapshot"].append(
                {"original": name, "path": str(Path("source") / name), "sha256": info["sha256"]}
            )
    shutil.copy2(Path(__file__), output / "source" / "scripts" / Path(__file__).name)
    write_json(output / "manifest.json", manifest)
    orchestration = (
        json.loads((output / "orchestration.json").read_text())
        if (output / "orchestration.json").exists()
        else {}
    )
    orchestration.update(
        {
            "status": "complete",
            "workers": len(labels),
            "logical_expected_records": plan["expected_records"],
            "physical_records": n_records,
            "duplicate_naive_reference_records": n_records - plan["expected_records"],
            "histories": len(observed_history_ids),
            "examples": len(examples),
        }
    )
    write_json(output / "orchestration.json", orchestration)
    if not manifest["pet_unchanged"]:
        raise AssertionError("Pet changed during shard execution")
    return {
        "status": "complete",
        "actual_records": n_records,
        "actual_examples": len(examples),
        "actual_histories": len(observed_history_ids),
        "naive_audit_groups": len(comparisons),
        "all_naive_repeats_equal": True,
        "pet_unchanged": manifest["pet_unchanged"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", nargs="+", default=["part0", "part1", "part2"])
    args = parser.parse_args()
    print(json.dumps(merge(args.output, args.shards), indent=2))


if __name__ == "__main__":
    main()
