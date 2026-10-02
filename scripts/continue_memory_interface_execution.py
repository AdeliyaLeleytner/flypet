#!/usr/bin/env python3
"""Detached, bounded continuation coordinator for the fixed 20260924 dataset."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/memory_interface_20260924_v2"
PYTHON = ROOT / ".venv/bin/python"


def write(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    tmp.replace(path)


def process_rss(root_pids):
    rows = subprocess.check_output(["ps", "-axo", "pid=,ppid=,rss="], text=True)
    processes = [tuple(map(int, line.split())) for line in rows.splitlines()]
    selected = set(root_pids)
    for _ in range(5):
        selected.update(pid for pid, ppid, rss in processes if ppid in selected)
    return sum(rss * 1024 for pid, ppid, rss in processes if pid in selected)


def main():
    recovery = json.loads((DATA / "recovery_evidence/recovery.json").read_text())
    orchestration = json.loads((DATA / "orchestration.json").read_text())
    started = time.time()
    workers = []
    orchestration.update(
        status="running_recovery",
        coordinator_pid=os.getpid(),
        recovery_started_unix=started,
        detached_processes=True,
        logical_expected_records=recovery["logical_expected_records"],
        physical_expected_records=recovery["physical_expected_records"],
        duplicate_naive_reference_records=recovery["duplicate_naive_reference_records"],
        merge_labels=recovery["merge_labels"],
        recovery_workers=[],
    )
    for row in recovery["shards"]:
        label = row["remaining_label"]
        output = DATA / "shards" / label
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite recovery shard {output}")
        log = Path(f"/tmp/fly-memory-interface-v2-{label}.log")
        with log.open("x") as f:
            command = [
                "/usr/bin/time",
                "-l",
                str(PYTHON),
                "-u",
                "scripts/gen_memory_interface_data.py",
                "--output",
                str(output),
                "--plan",
                str(DATA / row["remaining_plan"]),
            ]
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdin=subprocess.DEVNULL,
                stdout=f,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        workers.append((process, label, log))
        orchestration["recovery_workers"].append(
            {
                "label": label,
                "pid": process.pid,
                "process_group": process.pid,
                "log": str(log),
                "command": command,
            }
        )
        write(DATA / "orchestration.json", orchestration)
    previous = json.loads((DATA / "recovery_evidence/before-progress.json").read_text())
    maximum_rss = previous["maximum_observed_combined_rss_bytes"]
    try:
        while True:
            rows = []
            for process, label, log in workers:
                histories, records = 0, 0
                for line in log.read_text().splitlines():
                    if line.startswith('{"record"'):
                        records += 1
                    elif line.startswith('{"history_complete"'):
                        histories += 1
                rows.append(
                    {
                        "shard": label,
                        "records": records,
                        "histories_complete": histories,
                        "alive": process.poll() is None,
                        "returncode": process.returncode,
                    }
                )
            rss = process_rss([p.pid for p, _, _ in workers if p.poll() is None])
            maximum_rss = max(maximum_rss, rss)
            progress = {
                "observed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "status": "recovering",
                "preserved_completed_histories": 303,
                "shards": rows,
                "total_histories_complete": 303 + sum(r["histories_complete"] for r in rows),
                "total_physical_records": 3456 + sum(r["records"] for r in rows),
                "combined_rss_bytes": rss,
                "maximum_observed_combined_rss_bytes": maximum_rss,
                "rss_limit_bytes": 8589934592,
                "rss_limit_exceeded": maximum_rss > 8589934592,
                "note": "RSS sampled every 15 seconds; previous interrupted observer maximum retained. Exact recovery process peaks recorded by time -l.",
            }
            write(DATA / "progress.json", progress)
            if any(p.returncode not in (None, 0) for p, _, _ in workers):
                raise RuntimeError("A recovery simulation worker failed; inspect logs")
            if all(p.returncode == 0 for p, _, _ in workers):
                break
            if time.time() - started > 3600:
                raise TimeoutError("One-hour bounded recovery deadline exceeded")
            time.sleep(15)
        orchestration["status"] = "merging"
        write(DATA / "orchestration.json", orchestration)
        merge_log = DATA / "recovery_evidence/merge.log"
        with merge_log.open("x") as f:
            subprocess.run(
                [
                    "/usr/bin/time",
                    "-l",
                    str(PYTHON),
                    "-u",
                    "scripts/merge_memory_interface_data.py",
                    "--output",
                    str(DATA),
                    "--shards",
                    *recovery["merge_labels"],
                ],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        # This subprocess constructs its own brain and loads the saved learned memory.
        replay_code = """import json, pathlib, time
from flypet.experiments import replay_record, write_json, sha256_file
from flypet.memory_interface_data import pet_fingerprints
p = pathlib.Path("data/memory_interface_20260924_v2")
records = (json.loads(line) for line in (p / "records.jsonl").open())
record = next(r for r in records if r["phase"] == "post" and r["metadata"]["history"]["split"] == "test")
before = pet_fingerprints()
started = time.monotonic()
report = replay_record(p, record["record_id"])
if not (report["equal_sparse_counts"] and report["equal_spike_times_atol_1e9_s"]):
    raise AssertionError(report)
gate = {"status": "ready", "dataset": str(p), "manifest_sha256": sha256_file(p / "manifest.json"),
    "selection_rule": "first learned post record in original history order whose history split is test",
    "record_id": record["record_id"], "history_split": record["metadata"]["history"]["split"],
    "memory_strength": record["memory_before"]["strength"],
    "replay_report": "replay-" + record["record_id"] + ".json", "replay": report,
    "fresh_process_elapsed_s": time.monotonic() - started, "pet_unchanged": before == pet_fingerprints()}
if not gate["pet_unchanged"]:
    raise AssertionError("Personal pet changed during replay")
write_json(p / "dataset_ready.json", gate)
print(json.dumps(gate, indent=2))
"""
        with (DATA / "recovery_evidence/learned-test-replay.log").open("x") as f:
            subprocess.run(
                ["/usr/bin/time", "-l", str(PYTHON), "-u", "-c", replay_code],
                cwd=ROOT,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=True,
            )
        orchestration = json.loads((DATA / "orchestration.json").read_text())
        orchestration.update(
            status="complete_with_learned_replay",
            recovery_finished_unix=time.time(),
            dataset_ready="dataset_ready.json",
        )
        write(DATA / "orchestration.json", orchestration)
        progress["status"] = "complete_with_learned_replay"
        write(DATA / "progress.json", progress)
        print(
            json.dumps(
                {"status": "complete_with_learned_replay", "gate": str(DATA / "dataset_ready.json")}
            ),
            flush=True,
        )
    except BaseException as exc:
        for process, _, _ in workers:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        orchestration["status"] = "recovery_failed"
        orchestration["error"] = f"{type(exc).__name__}: {exc}"
        write(DATA / "orchestration.json", orchestration)
        raise


if __name__ == "__main__":
    main()
