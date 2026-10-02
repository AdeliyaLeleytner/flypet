#!/usr/bin/env python3
"""Fresh-process replay of the first fixed-order learned test probe."""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypet.experiments import replay_record, write_json, sha256_file
from flypet.memory_interface_data import pet_fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    args = parser.parse_args()
    dataset = args.dataset
    if (dataset / "dataset_ready.json").exists():
        raise FileExistsError("Dataset replay gate already exists")
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["status"] != "complete" or not manifest["pet_unchanged"]:
        raise AssertionError("Merged dataset is incomplete or personal pet changed")
    records = (json.loads(line) for line in (dataset / "records.jsonl").open())
    record = next(
        r for r in records if r["phase"] == "post" and r["metadata"]["history"]["split"] == "test"
    )
    before = pet_fingerprints()
    started = time.monotonic()
    report = replay_record(dataset, record["record_id"])
    if not (report["equal_sparse_counts"] and report["equal_spike_times_atol_1e9_s"]):
        raise AssertionError(report)
    gate = {
        "status": "ready",
        "dataset": str(dataset),
        "manifest_sha256": sha256_file(dataset / "manifest.json"),
        "selection_rule": "first learned post record in original history order whose history split is test",
        "record_id": record["record_id"],
        "history_split": record["metadata"]["history"]["split"],
        "memory_strength": record["memory_before"]["strength"],
        "replay_report": "replay-" + record["record_id"] + ".json",
        "replay": report,
        "fresh_process_elapsed_s": time.monotonic() - started,
        "pet_unchanged": before == pet_fingerprints(),
        "recovery": {
            "original_coordinator_replay_hook": "ImportError before simulator initialization; no valid replay was produced by that hook.",
            "replacement_gate_script": "scripts/replay_memory_interface_gate.py",
            "replacement_gate_script_sha256": sha256_file(Path(__file__)),
        },
    }
    if not gate["pet_unchanged"]:
        raise AssertionError("Personal pet changed during replay")
    write_json(dataset / "dataset_ready.json", gate)
    print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()
