#!/usr/bin/env python3
"""Prepare or execute the authorized three-seed prospective memory-forecast experiment.

--dry-run performs local dataset checks, CPU preparation and an allowlisted
snapshot only. --quote searches offers and records the cost without renting.
--launch uses that reviewed quote (at most three minutes old) and must be used only after
the root agent's ready-to-run instruction. Never uploads credentials or homes.
Each worker harvests its artifacts and destroys its own instance immediately.
A detached watchdog enforces a 90-minute deadline for these exact job labels.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
from uuid import uuid4

import rerun_projector_clean as h

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
RESOURCE_SECONDS = 90 * 60
TRAIN_SECONDS = 3600
BUDGET_USD = 3.50
FAILED_ATTEMPT_RESERVE_USD = 0.50
FAILED_ATTEMPTS = [
    ROOT / "data/memory_forecast_reader_20260924_gpu",
    ROOT / "data/memory_forecast_reader_20260924_gpu_retry1",
]
EXCLUDED_OFFER_IDS = {51256825, 50454636}
MAX_HOURLY_USD = 0.60
MAX_NETWORK_USD_PER_GB = 0.03
NETWORK_RESERVE_GB = 8.0  # total inbound+outbound per instance, conservatively
SOURCE_FILES = (
    "flypet/__init__.py",
    "flypet/memory_reader_data.py",
    "flypet/memory_forecast_data.py",
    "scripts/train_memory_forecast.py",
)
DATA_FILES = (
    "manifest.json",
    "examples.jsonl",
    "features.npz",
    "prospective_context.json",
    "expected_predictions.json",
)
REQUIRED_RESULTS = (
    "neural_projector.pt",
    "history_projector.pt",
    "null_projector.pt",
    "manifest.json",
    "report.json",
    "predictions.json",
    "preprocessing.npz",
)
REMOTE = "/workspace/fly"


def training_args(seed, dataset="data/forecast_dataset", output=f"{REMOTE}/results"):
    return [
        "python",
        "-u",
        "scripts/train_memory_forecast.py",
        "--data",
        dataset,
        "--out",
        output,
        "--model",
        MODEL,
        "--revision",
        REVISION,
        "--epochs",
        "12",
        "--hidden",
        "256",
        "--tokens",
        "8",
        "--batch",
        "16",
        "--generation-batch",
        "24",
        "--generation-per-split",
        "0",
        "--dtype",
        "bfloat16",
        "--device",
        "cuda",
        "--seed",
        str(seed),
        "--modes",
        "neural",
        "history",
        "null",
        "--max-seconds",
        str(TRAIN_SECONDS),
    ]


def dataset_ready(dataset):
    """Verify derived-task parity, completed source simulations and real smoke."""
    dataset = Path(dataset).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    gate = json.loads((dataset / "gpu_ready.json").read_text())
    if (
        gate.get("status") != "ready"
        or gate.get("manifest_sha256") != h.sha(dataset / "manifest.json")
        or gate.get("smoke_complete") is not True
    ):
        raise ValueError("Forecast ready gate is absent, stale or did not pass real-model smoke")
    smoke_path = (ROOT / gate["smoke_report"]).resolve()
    if not smoke_path.is_relative_to(dataset) or h.sha(smoke_path) != gate["smoke_report_sha256"]:
        raise ValueError("Forecast smoke report differs from ready gate")
    smoke = json.loads(smoke_path.read_text())
    if (
        smoke.get("complete") is not True
        or smoke.get("smoke_only") is not True
        or smoke.get("model_revision") != REVISION
        or gate.get("model_revision") != REVISION
    ):
        raise ValueError("Forecast technical smoke did not complete")
    for name, expected in gate["code_sha256"].items():
        if name not in SOURCE_FILES or h.sha(ROOT / name) != expected:
            raise ValueError(f"Forecast source changed after smoke: {name}")
    if set(gate["code_sha256"]) != set(SOURCE_FILES) - {"flypet/__init__.py"}:
        raise ValueError("Forecast smoke gate must cover all three training source files")
    original = ROOT / "data/memory_interface_20260924_v2"
    analysis = ROOT / "data/memory_interface_analysis_20260924_v2"
    source_gate = json.loads((original / "dataset_ready.json").read_text())
    source_manifest = json.loads((original / "manifest.json").read_text())
    verification = json.loads((original / "verification.json").read_text())
    if (
        source_manifest.get("status") != "complete"
        or not verification.get("ok")
        or verification.get("errors")
        or source_manifest.get("pet_unchanged") is not True
        or source_gate.get("pet_unchanged") is not True
        or source_gate.get("manifest_sha256") != h.sha(original / "manifest.json")
        or source_gate.get("replay", {}).get("equal_sparse_counts") is not True
        or source_gate.get("replay", {}).get("equal_spike_times_atol_1e9_s") is not True
    ):
        raise ValueError("Original dataset/replay/pet-isolation gate does not pass")
    for name, expected in manifest["sources"].items():
        prefix, rel = name.split("/", 1)
        base = {"source": original, "analysis": analysis}.get(prefix)
        if (
            base is None
            or not (base / rel).resolve().is_relative_to(base.resolve())
            or h.sha(base / rel) != expected
        ):
            raise ValueError(f"Forecast derivation source changed: {name}")
    if (
        manifest.get("task") != "prospective_delta_valence"
        or manifest.get("target_response_excluded") is not True
        or manifest.get("diagnostic_chemicals") != ["methyl acetate", "1-hexanol"]
        or manifest.get("cpu_raw_features_exact_match") is not True
        or manifest.get("cpu_delta_targets_exact_match") is not True
        or sum(manifest.get("counts", {}).values()) != 1440
    ):
        raise ValueError("Forecast task provenance/counts differ from authorized task")
    files = {}
    for name in DATA_FILES:
        path = dataset / name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked forecast input: {name}")
        files[name] = {"sha256": h.sha(path), "bytes": path.stat().st_size}
        if name != "manifest.json" and manifest["derived_files"].get(name) != files[name]["sha256"]:
            raise ValueError(f"Forecast dataset checksum mismatch: {name}")
    return {
        "dataset_manifest_sha256": h.sha(dataset / "manifest.json"),
        "ready_gate_sha256": h.sha(dataset / "gpu_ready.json"),
        "smoke_report_sha256": h.sha(smoke_path),
        "status": "complete",
        "task": manifest["task"],
        "actual_examples": 1440,
        "actual_histories": 360,
        "parent_manifest_sha256": source_gate["manifest_sha256"],
        "target_response_excluded": True,
        "inputs": files,
    }


def prepare_snapshot(dataset, output):
    """All CPU-only gates finish before looking for or creating paid resources."""
    dataset = Path(dataset).resolve()
    identity = dataset_ready(dataset)
    args = training_args(0, str(dataset), str(output / "cpu_preparation"))
    args[0] = sys.executable
    args[2] = str(ROOT / "scripts/train_memory_forecast.py")
    args.append("--prepare-only")
    prepared = h.run(args, timeout=300, check=False)
    (output / "cpu_preparation.log").write_bytes(prepared.stdout + prepared.stderr)
    if prepared.returncode:
        raise RuntimeError("CPU preparation failed; no GPU resources were created")
    manifest = json.loads((output / "cpu_preparation/manifest.json").read_text())
    if (
        not manifest.get("counts", {}).get("train")
        or not manifest["counts"].get("validation")
        or manifest.get("preprocessing", {}).get("fit_split") != "train"
        or not manifest.get("split_fingerprint")
    ):
        raise ValueError(
            "CPU preparation lacks training-only preprocessing or nonempty fitting splits"
        )
    heldout = {
        k: v for k, v in manifest["counts"].items() if k not in ("train", "validation", "excluded")
    }
    if not heldout or not all(v > 0 for v in heldout.values()):
        raise ValueError("CPU preparation lacks nonempty held-out evaluation partitions")
    if sum(manifest["counts"].values()) != identity["actual_examples"]:
        raise ValueError("Prepared row counts differ from the complete dataset manifest")
    if dataset_ready(dataset) != identity:
        raise RuntimeError("Dataset changed during preparation")
    h.save(output / "dataset_identity.json", identity)
    files = {name: ROOT / name for name in SOURCE_FILES}
    files.update({f"data/forecast_dataset/{name}": dataset / name for name in DATA_FILES})
    source_manifest = {}
    for name, path in files.items():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Missing or symlinked source file: {name}")
        source_manifest[name] = {"sha256": h.sha(path), "bytes": path.stat().st_size}
    for name, expected in manifest["code_sha256"].items():
        if source_manifest.get(name, {}).get("sha256") != expected:
            raise ValueError(f"Reader code changed after CPU preparation: {name}")
    # Preserve provenance without sending unrelated generator paths or pet state.
    source_manifest["dataset_identity.json"] = {
        "sha256": h.sha(output / "dataset_identity.json"),
        "bytes": (output / "dataset_identity.json").stat().st_size,
    }
    files["dataset_identity.json"] = output / "dataset_identity.json"
    h.save(output / "source_manifest.json", source_manifest)
    package = output / "source.tar.gz"
    with tarfile.open(package, "w:gz") as archive:
        for name, path in files.items():
            archive.add(path, arcname=name, recursive=False)
        archive.add(
            output / "source_manifest.json", arcname="source_manifest.json", recursive=False
        )
    for name, path in files.items():
        if h.sha(path) != source_manifest[name]["sha256"]:
            raise RuntimeError(f"Input changed during snapshot: {name}")
    # Budget assumes the source upload is comfortably inside the 8 GB envelope.
    if package.stat().st_size > 1024**3:
        raise ValueError("Snapshot exceeds 1 GiB; update transfer estimate before provisioning")
    return package, manifest


def select_offers(offers):
    candidates = []
    for offer in offers:
        try:
            price = float(offer["dph_total"])
            network = max(float(offer["inet_up_cost"]), float(offer["inet_down_cost"]))
            if (
                offer["gpu_name"] not in ("RTX 4090", "RTX_4090")
                or int(offer["num_gpus"]) != 1
                or not 0 < price <= MAX_HOURLY_USD
                or not 0 <= network <= MAX_NETWORK_USD_PER_GB
            ):
                continue
        except (KeyError, TypeError, ValueError):
            continue
        candidates.append(offer)
    candidates.sort(key=lambda offer: float(offer["dph_total"]))
    selected, machines = [], set()
    for offer in candidates:
        # Prefer separate machines so one host failure cannot stop all seeds.
        if offer.get("machine_id") in machines:
            continue
        selected.append({k: offer.get(k) for k in h.OFFER_KEYS})
        machines.add(offer.get("machine_id"))
        if len(selected) == 3:
            break
    if len(selected) < 3:
        chosen_ids = {offer["id"] for offer in selected}
        for offer in candidates:
            if offer["id"] in chosen_ids:
                continue
            selected.append({k: offer.get(k) for k in h.OFFER_KEYS})
            chosen_ids.add(offer["id"])
            if len(selected) == 3:
                break
    selected.sort(key=lambda offer: float(offer["dph_total"]))
    if len(selected) != 3:
        raise ValueError("Need three distinct suitable RTX 4090 offers within the rate caps")
    compute = sum(float(o["dph_total"]) for o in selected) * RESOURCE_SECONDS / 3600
    transfer = sum(
        max(float(o["inet_up_cost"]), float(o["inet_down_cost"])) * NETWORK_RESERVE_GB
        for o in selected
    )
    estimate = {
        "max_compute_and_disk_usd": compute,
        "network_reserve_usd": transfer,
        "conservative_total_usd": compute + transfer,
        "budget_usd": BUDGET_USD,
        "network_gb_per_instance": NETWORK_RESERVE_GB,
        "resource_seconds": RESOURCE_SECONDS,
        "training_seconds": TRAIN_SECONDS,
    }
    if estimate["conservative_total_usd"] > BUDGET_USD:
        raise ValueError("Offers exceed the new-job budget envelope")
    return selected, estimate


def instance_list():
    result = h.api("show", "instances")
    if not isinstance(result, list):
        raise ValueError("Unrecognized instance listing; cannot verify cleanup")
    return result


def owned_instances(receipt):
    ids = {int(e["instance_id"]) for e in receipt.get("instances", [])}
    labels = set(receipt["labels"])
    return [i for i in instance_list() if int(i["id"]) in ids or i.get("label") in labels]


def destroy_verified(ident):
    response = h.destroy(ident)
    for _ in range(4):
        if not any(int(item["id"]) == int(ident) for item in instance_list()):
            return {"response": response, "verified_absent": True, "destroyed_unix": time.time()}
        time.sleep(3)
    return {"response": response, "verified_absent": False}


def watchdog(receipt_path):
    """Separate OS process, independent of the training/controller process."""
    path = Path(receipt_path)
    while True:
        receipt = json.loads(path.read_text())
        if receipt.get("cleanup_complete"):
            return
        left = float(receipt["deadline_unix"]) - time.time()
        if left > 0:
            time.sleep(min(30, left))
            continue
        try:
            outcomes = {
                str(i["id"]): destroy_verified(int(i["id"])) for i in owned_instances(receipt)
            }
            remaining = [i["id"] for i in owned_instances(receipt)]
            h.save(
                path.parent / "watchdog_cleanup.json",
                {
                    "checked_unix": time.time(),
                    "outcomes": outcomes,
                    "remaining_ids": remaining,
                    "complete": not remaining,
                },
            )
            if not remaining:
                return
        except Exception as exc:
            h.log(f"watchdog cleanup attempt failed: {type(exc).__name__}")
        time.sleep(15)


SETUP = r"""set -euo pipefail
cd /workspace/fly
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
python - <<'PY'
import hashlib,json
from pathlib import Path
for name,info in json.loads(Path('source_manifest.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==info['sha256'],name
PY
python -m pip install --disable-pip-version-check --no-cache-dir 'numpy==2.2.6' 'transformers==5.17.0' 'accelerate' 'safetensors' > setup.log 2>&1
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json
revision='c1899de289a04d12100db370d81485cdf75e47ca'
p=snapshot_download('Qwen/Qwen3-0.6B', revision=revision, allow_patterns=[
 'config.json','generation_config.json','model.safetensors','model.safetensors.index.json',
 'model-*.safetensors','tokenizer.json','tokenizer_config.json','vocab.json','merges.txt',
 'added_tokens.json','special_tokens_map.json'])
assert Path(p).name==revision,(p,revision)
Path('model_snapshot.json').write_text(json.dumps({'model':'Qwen/Qwen3-0.6B','revision':revision,
 'snapshot_commit':Path(p).name},indent=2)+'\n')
PY
python -m pip freeze > environment.txt
nvidia-smi > gpu_environment.txt
"""

SEAL = r"""set -euo pipefail
cd /workspace/fly
mkdir -p results
for f in setup.log setup_stdout.log source_manifest.json dataset_identity.json environment.txt gpu_environment.txt model_snapshot.json stdout.log launcher.log; do
    test ! -f "$f" || cp "$f" results/
done
python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('results')
hashes={str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest()
        for x in p.rglob('*') if x.is_file() and x.name!='SHA256.json'}
(p/'SHA256.json').write_text(json.dumps(hashes,sort_keys=True,indent=2)+'\n')
PY
"""


def extract_verified(archive_path, local):
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or not path.parts
                or path.parts[0] != "results"
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
            ):
                raise ValueError(f"Unsafe result archive member: {member.name}")
        archive.extractall(local, filter="data")
    result = local / "results"
    checksums = json.loads((result / "SHA256.json").read_text())
    for name, expected in checksums.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not (result / path).is_file():
            raise ValueError(f"Invalid checksum entry: {name}")
        if h.sha(result / path) != expected:
            raise ValueError(f"Fetched checksum mismatch: {name}")
    actual = {
        str(p.relative_to(result))
        for p in result.rglob("*")
        if p.is_file() and p.name != "SHA256.json"
    }
    if actual != set(checksums):
        raise ValueError("Fetched files are not all covered by checksums")
    return result, checksums


def harvest(info, local, entry):
    h.ssh(info, "bash -s", input=SEAL.encode(), timeout=90)
    transfer = h.ssh(info, "tar czf - -C /workspace/fly results", timeout=240)
    archive = local / "results.tar.gz"
    archive.write_bytes(transfer.stdout)
    result, checksums = extract_verified(archive, local)
    entry["verified_local_files"] = len(checksums)
    entry["result_archive_sha256"] = h.sha(archive)
    return result, checksums


def check_finished(result, checksums, expected_split):
    for name in REQUIRED_RESULTS:
        if name not in checksums:
            raise ValueError(f"Required result missing: {name}")
    if int((result / "exit_code.txt").read_text()) != 0:
        raise RuntimeError("Remote training returned a nonzero exit code")
    report = json.loads((result / "report.json").read_text())
    manifest = json.loads((result / "manifest.json").read_text())
    if (
        report.get("complete") is not True
        or report.get("experiment_complete") is not True
        or report.get("smoke_only") is not False
        or report.get("model_revision") != REVISION
    ):
        raise ValueError("Incomplete result or unexpected model revision")
    if manifest.get("split_fingerprint") != expected_split:
        raise ValueError("Remote dataset split differs from CPU preparation")
    for mode in ("neural", "history", "null", "oracle_numeric_text_positive_control"):
        if not report.get("evaluation", {}).get(mode):
            raise ValueError(f"Missing evaluation: {mode}")
    if not report.get("no_change_baseline"):
        raise ValueError("Missing prospective no-change baseline")
    if not report.get("state_swap"):
        raise ValueError("Missing state-swap evaluation")
    return report


def wait_ready(entry, deadline, cancel):
    limit = min(deadline, entry["created_unix"] + 600)
    while time.time() < limit and not cancel.is_set():
        info = h.api("show", "instance", entry["instance_id"])
        if isinstance(info, list):
            info = info[0] if info else {}
        entry["status"] = {k: info.get(k) for k in h.STATUS_KEYS}
        if info.get("actual_status") == "running" and info.get("ssh_host") and info.get("ssh_port"):
            probe = h.ssh(
                info,
                "python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__)'",
                check=False,
                timeout=45,
            )
            if probe.returncode == 0:
                return info
        cancel.wait(15)
    raise TimeoutError("Instance readiness cancelled or exceeded its 10-minute bound")


def worker(entry, output, package, deadline, expected_split, cancel, persist):
    local = output / f"seed_{entry['seed']}"
    local.mkdir(exist_ok=False)
    info, error, harvested = None, None, False
    try:
        info = wait_ready(entry, deadline, cancel)
        actual_price = float(info["dph_total"])
        actual_network = max(float(info["inet_up_cost"]), float(info["inet_down_cost"]))
        if actual_price > MAX_HOURLY_USD or actual_network > MAX_NETWORK_USD_PER_GB:
            raise ValueError(
                "Live instance rates exceed the authorized caps; stopping before setup"
            )
        if cancel.is_set():
            raise RuntimeError("Run cancelled before setup")
        h.ssh(
            info,
            "mkdir -p /workspace/fly && tar xzf - -C /workspace/fly",
            input=package.read_bytes(),
            timeout=240,
        )
        setup = h.ssh(info, "bash -s", input=SETUP.encode(), timeout=600, check=False)
        (local / "setup_stdout.log").write_bytes(setup.stdout + setup.stderr)
        if setup.returncode:
            raise RuntimeError(f"Setup exited {setup.returncode}")
        if cancel.is_set() or time.time() + TRAIN_SECONDS + 360 > deadline:
            raise RuntimeError(
                "Insufficient watchdog margin for a complete training/evaluation run"
            )
        args = training_args(entry["seed"])
        entry["command"] = shlex.join(args)
        remote = (
            "#!/usr/bin/env bash\nset -uo pipefail\ncd /workspace/fly\n"
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false\n"
            f"timeout --signal=TERM --kill-after=30s {TRAIN_SECONDS + 30}s {shlex.join(args)} > stdout.log 2>&1\n"
            "rc=$?\nmkdir -p results\nprintf '%s\\n' \"$rc\" > results/exit_code.txt\ntouch results/DONE\n"
        )
        h.ssh(info, "cat > /workspace/fly/run.sh", input=remote.encode())
        h.ssh(
            info, "nohup bash /workspace/fly/run.sh </dev/null >/workspace/fly/launcher.log 2>&1 &"
        )
        entry["launched_unix"] = time.time()
        persist()
        previous = ""
        while time.time() < deadline - 300 and not cancel.is_set():
            status = h.ssh(
                info,
                "test -f /workspace/fly/results/DONE; rc=$?; tail -2 /workspace/fly/stdout.log; exit $rc",
                check=False,
                timeout=60,
            )
            progress = status.stdout.decode(errors="replace").strip()
            if progress != previous:
                h.log(f"forecast seed {entry['seed']}: {progress[-900:]}")
                previous = progress
            if status.returncode == 0:
                break
            if status.returncode not in (1,):
                raise RuntimeError(f"Remote monitoring failed with SSH status {status.returncode}")
            cancel.wait(20)
        else:
            raise TimeoutError("Training cancelled or fetch-margin deadline reached")
        result, checksums = harvest(info, local, entry)
        harvested = True
        report = check_finished(result, checksums, expected_split)
        entry.update(
            {
                "training_complete": True,
                "exit_code": 0,
                "generation_rows": report["generation_rows"],
                "completed_unix": time.time(),
            }
        )
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        entry["error"] = error
        (local / "failure.log").write_text(error + "\n")
        if info is not None and not harvested:
            try:
                h.ssh(
                    info,
                    "pkill -TERM -f '[p]ython -u scripts/train_memory_forecast.py' || true",
                    check=False,
                    timeout=30,
                )
                h.ssh(
                    info,
                    "for i in $(seq 1 20); do pgrep -f '[p]ython -u scripts/train_memory_forecast.py' >/dev/null || break; sleep 1; done; "
                    "pkill -KILL -f '[p]ython -u scripts/train_memory_forecast.py' || true",
                    check=False,
                    timeout=30,
                )
                if entry.get("launched_unix"):
                    h.ssh(
                        info,
                        "for i in $(seq 1 5); do test -f /workspace/fly/results/DONE && break; sleep 1; done",
                        check=False,
                        timeout=15,
                    )
                harvest(info, local, entry)
                harvested = True
            except Exception as salvage_error:
                entry["salvage_error"] = f"{type(salvage_error).__name__}: {salvage_error}"
    finally:
        try:
            latest = h.api("show", "instance", entry["instance_id"])
            if isinstance(latest, list):
                latest = latest[0] if latest else {}
            entry["last_provider_status"] = {k: latest.get(k) for k in h.STATUS_KEYS}
        except Exception as exc:
            entry["last_provider_status_error"] = type(exc).__name__
        try:
            entry["cleanup"] = destroy_verified(entry["instance_id"])
        except Exception as exc:
            entry["cleanup"] = {"verified_absent": False, "error": type(exc).__name__}
        entry["artifacts_harvested"] = harvested
        h.save(local / "receipt.json", entry)
        persist()
    if error or not entry["cleanup"]["verified_absent"]:
        raise RuntimeError(error or "Worker cleanup could not be verified")
    return entry


def invoice_receipt(receipt):
    """Only retain invoice rows associated with this job, never account details."""
    wanted = {int(e["instance_id"]) for e in receipt["instances"]}
    wanted.update(int(i) for i in receipt.get("uncertain_create_cleanup", {}))
    charges, errors = [], []
    for label in receipt["labels"]:
        try:
            raw = h.api("show", "invoices", "--only_charges", "--instance_label", label)
            if isinstance(raw, list):
                rows = raw
            elif isinstance(raw, dict) and ("invoices" in raw or "rows" in raw):
                rows = raw.get("invoices", raw.get("rows"))
            else:
                raise ValueError("Unrecognized invoice response")
            for row in rows:
                if int(row.get("instance_id", -1)) in wanted:
                    charges.append(
                        {
                            k: row.get(k)
                            for k in (
                                "instance_id",
                                "amount",
                                "quantity",
                                "rate",
                                "timestamp",
                                "type",
                                "description",
                            )
                        }
                    )
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}")
    # Providers may return the same row more than once despite a label filter.
    charges = list({json.dumps(row, sort_keys=True): row for row in charges}.values())
    amount = sum((Decimal(str(row["amount"])) for row in charges), Decimal("0"))
    return {
        "checked_unix": time.time(),
        "charges": charges,
        "reported_total_usd": str(amount) if charges or not errors else None,
        "partial_invoice_fetch": bool(errors),
        "errors": errors,
        "billing_may_settle_later": True,
        "separate_persistent_resources_created": False,
    }


def prior_stage_guard():
    previous = ROOT / "data/memory_interface_reader_20260924_gpu"
    resource = json.loads((previous / "resource_receipt.json").read_text())
    billing = json.loads((previous / "billing_and_cleanup_receipt.json").read_text())
    if (
        resource.get("cleanup_complete") is not True
        or resource.get("failures")
        or len(resource.get("instances", [])) != 3
        or not all(
            e.get("training_complete") and e.get("cleanup", {}).get("verified_absent")
            for e in resource["instances"]
        )
        or billing.get("all_created_resources_destroyed") is not True
    ):
        raise ValueError("Current-response stage must complete and clean up before forecast rental")
    # Carry the preceding run's entire quoted envelope, even when actual charges
    # are lower, so provider settlement cannot exhaust the session budget.
    failed_bound, failed_hashes = 0.0, []
    for directory in FAILED_ATTEMPTS:
        failed = json.loads((directory / "resource_receipt.json").read_text())
        if (
            failed.get("cleanup_complete") is not True
            or not failed.get("failures")
            or failed.get("remaining_created_instance_ids")
        ):
            raise ValueError("Failed provisioning attempts must be preserved and fully cleaned up")
        failed_bound += (
            float(failed["estimated_compute_disk_usd"])
            + float(failed["estimate"]["network_reserve_usd"])
            + 0.03
        )
        if owned_instances(failed):
            raise ValueError("A prior forecast provisioning resource is still present")
        failed_hashes.append(h.sha(directory / "resource_receipt.json"))
    if failed_bound > FAILED_ATTEMPT_RESERVE_USD:
        raise ValueError("Failed provisioning reserve is insufficient")
    prior_bound = (
        0.354 + float(resource["estimate"]["conservative_total_usd"]) + FAILED_ATTEMPT_RESERVE_USD
    )
    if prior_bound + BUDGET_USD > 10.0:
        raise ValueError("Forecast stage could exceed the user-authorized total $10")
    return {
        "prior_session_spend_bound_usd": prior_bound,
        "failed_forecast_attempt_reserve_usd": FAILED_ATTEMPT_RESERVE_USD,
        "failed_attempt_receipt_sha256": failed_hashes,
        "forecast_stage_total_cap_usd": 4.0,
        "prior_reader_reported_usd": billing.get("reported_total_usd"),
        "prior_reader_estimated_compute_disk_usd": billing.get("estimated_compute_disk_usd"),
        "prior_reader_resource_receipt_sha256": h.sha(previous / "resource_receipt.json"),
        "session_total_bound_usd": prior_bound + BUDGET_USD,
    }


def fresh_offers():
    python = Path.home() / ".local/share/uv/tools/vastai/bin/python"
    result = h.run([str(python), str(ROOT / "deploy/search_vast_fresh.py")], timeout=60)
    offers = json.loads(result.stdout)
    if not isinstance(offers, list):
        raise ValueError("Fresh offer endpoint schema is invalid")
    return [
        o
        for o in offers
        if o.get("rentable") is True and o.get("rented") is False and o.get("is_bid") is False
    ]


def quote(output, package):
    prior = prior_stage_guard()
    offers = fresh_offers()
    offers = [offer for offer in offers if int(offer["id"]) not in EXCLUDED_OFFER_IDS]
    selected, estimate = select_offers(offers)
    estimate.update(
        {
            "quoted_unix": time.time(),
            "source_archive_sha256": h.sha(package),
            "prior_stage": prior,
            "fresh_endpoint": "/search/asks/",
            "hard_new_resources_envelope_usd": 3.42,
        }
    )
    h.save(output / "selected_offers.json", selected)
    h.save(output / "cost_estimate.json", estimate)  # concrete cost before create
    print(
        json.dumps(
            {"selected_offers": selected, "estimate": estimate, "resources_created": 0}, indent=2
        ),
        flush=True,
    )


def prepared_output(output):
    ready = json.loads((output / "dry_run.json").read_text())
    package = output / "source.tar.gz"
    if (
        ready.get("status") != "ready"
        or h.sha(package) != ready["package_sha256"]
        or h.sha(output / "cpu_preparation/manifest.json") != ready["preparation_manifest_sha256"]
        or h.sha(output / "source_manifest.json") != ready["source_manifest_sha256"]
    ):
        raise ValueError("Prepared snapshot or CPU audit changed; prepare a fresh output directory")
    if ready["commands"] != [training_args(i) for i in range(3)]:
        raise ValueError("Training arguments changed after preparation")
    return package, json.loads((output / "cpu_preparation/manifest.json").read_text())


class OfferUnavailable(RuntimeError):
    pass


def create_instance_recorded(offer, label, output, seed, attempt=0):
    args = [
        "vastai",
        "create",
        "instance",
        str(offer["id"]),
        "--image",
        h.IMAGE,
        "--disk",
        "40",
        "--ssh",
        "--direct",
        "--cancel-unavail",
        "--label",
        label,
        "--raw",
    ]
    response = h.run(args, check=False)
    for suffix, data in (("stdout", response.stdout), ("stderr", response.stderr)):
        path = output / f"create_seed_{seed}_attempt_{attempt}_{suffix}.log"
        path.write_bytes(data)
        path.chmod(0o600)
    try:
        parsed = json.loads(response.stdout)
    except json.JSONDecodeError:
        parsed = None
    if (
        response.returncode == 0
        and isinstance(parsed, dict)
        and parsed.get("success")
        and parsed.get("new_contract")
    ):
        return parsed
    try:
        error = json.loads(response.stderr)
    except json.JSONDecodeError:
        error = parsed if isinstance(parsed, dict) else {}
    if isinstance(error, dict) and "no_such_ask" in str(error.get("msg", "")):
        matching = [item for item in instance_list() if item.get("label") == label]
        if len(matching) == 1:
            return {
                "success": True,
                "new_contract": int(matching[0]["id"]),
                "recovered_from_label": True,
            }
        if not matching:
            raise OfferUnavailable(
                f"seed {seed} offer {offer['id']} disappeared; provider no_such_ask and exact label absent"
            )
        raise RuntimeError("Multiple instances share exact label after provider error")
    # A malformed response is an uncertain mutation: search exact label before
    # deciding failure; never retry the create operation itself blindly.
    for attempt in range(3):
        matching = [item for item in instance_list() if item.get("label") == label]
        if len(matching) == 1:
            h.log(f"recovered seed {seed} create by exact label after malformed provider response")
            return {
                "success": True,
                "new_contract": int(matching[0]["id"]),
                "recovered_from_label": True,
            }
        if len(matching) > 1:
            raise RuntimeError(
                "Provider created multiple instances for an exact label; cleanup required"
            )
        if attempt < 2:
            time.sleep(3)
    raise RuntimeError(
        f"Provider create failed for seed {seed}; captured response logs, exit {response.returncode}"
    )


def execute(output, package, prep):
    prior = prior_stage_guard()
    if (output / "resource_receipt.json").exists():
        raise FileExistsError(
            "This output already has a resource receipt; never create duplicate jobs"
        )
    selected = json.loads((output / "selected_offers.json").read_text())
    estimate = json.loads((output / "cost_estimate.json").read_text())
    checked, recomputed = select_offers(selected)
    if (
        checked != selected
        or estimate["source_archive_sha256"] != h.sha(package)
        or not 0 <= time.time() - estimate["quoted_unix"] <= 180
        or any(estimate[key] != value for key, value in recomputed.items())
    ):
        raise ValueError("Quote is stale or changed; run --quote again before renting")
    now = time.time()
    job = uuid4().hex[:10]
    receipt = {
        "job_id": job,
        "started_unix": now,
        "deadline_unix": now + RESOURCE_SECONDS,
        "controller_pid": os.getpid(),
        "budget_usd": BUDGET_USD,
        "prior_stage": prior,
        "stage": "prospective_delta_valence",
        "total_user_budget_usd": 10.0,
        "estimate": estimate,
        "image": h.IMAGE,
        "model": MODEL,
        "model_revision": REVISION,
        "source_archive_sha256": h.sha(package),
        "split_fingerprint": prep["split_fingerprint"],
        "labels": [f"fly-forecast-{job}-seed{i}" for i in range(3)],
        "instances": [],
        "cleanup_complete": False,
        "failures": [],
    }
    path, lock, cancel = output / "resource_receipt.json", threading.RLock(), threading.Event()

    def persist():
        with lock:
            h.save(path, receipt)

    persist()
    with (output / "watchdog.log").open("wb") as log:
        guard = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--watchdog", str(path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt["watchdog_pid"] = guard.pid
    persist()

    def interrupt(signum, frame):
        cancel.set()
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)
    try:
        used_offer_ids = set(EXCLUDED_OFFER_IDS)
        used_machines = set()
        receipt["provisioning_quotes"] = []
        receipt["hard_new_resources_envelope_usd"] = 3.42
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {}
            for seed in range(3):
                created = False
                for attempt in range(3):
                    if time.time() > now + 600 or cancel.is_set():
                        raise TimeoutError("Bounded just-in-time provisioning deadline reached")
                    candidates = fresh_offers()
                    candidates = [
                        o
                        for o in candidates
                        if o.get("gpu_name") in ("RTX 4090", "RTX_4090")
                        and int(o.get("num_gpus", 0)) == 1
                        and int(o["id"]) not in used_offer_ids
                        and o.get("machine_id") not in used_machines
                        and 0 < float(o["dph_total"]) <= MAX_HOURLY_USD
                        and max(float(o["inet_up_cost"]), float(o["inet_down_cost"]))
                        <= MAX_NETWORK_USD_PER_GB
                    ]
                    candidates.sort(key=lambda o: float(o["dph_total"]))
                    if not candidates:
                        receipt["failures"].append(f"No eligible fresh offer for seed {seed}")
                        break
                    offer = {key: candidates[0].get(key) for key in h.OFFER_KEYS}
                    used_offer_ids.add(int(offer["id"]))
                    quoted = {
                        "seed": seed,
                        "attempt": attempt,
                        "quoted_unix": time.time(),
                        "offer": offer,
                        "max_90min_compute_disk_usd": float(offer["dph_total"]) * 1.5,
                        "max_transfer_usd": max(
                            float(offer["inet_up_cost"]), float(offer["inet_down_cost"])
                        )
                        * 8,
                    }
                    receipt["provisioning_quotes"].append(quoted)
                    persist()  # concrete per-offer rate/cost is durable before create
                    h.log(
                        f"forecast seed {seed} fresh offer {offer['id']} at ${float(offer['dph_total']):.6f}/h before create"
                    )
                    try:
                        result = create_instance_recorded(
                            offer, receipt["labels"][seed], output, seed, attempt
                        )
                    except OfferUnavailable as exc:
                        quoted["outcome"] = "no_such_ask_no_contract"
                        h.log(str(exc))
                        persist()
                        continue
                    if not result.get("success") or not result.get("new_contract"):
                        raise RuntimeError(f"Instance creation failed for seed {seed}")
                    entry = {
                        "seed": seed,
                        "instance_id": int(result["new_contract"]),
                        "offer": offer,
                        "created_unix": time.time(),
                        "label": receipt["labels"][seed],
                    }
                    receipt["instances"].append(entry)
                    used_machines.add(offer["machine_id"])
                    quoted["outcome"] = "created"
                    persist()
                    h.log(
                        f"forecast seed {seed} created instance {entry['instance_id']} at ${float(offer['dph_total']):.4f}/h"
                    )
                    futures[
                        pool.submit(
                            worker,
                            entry,
                            output,
                            package,
                            receipt["deadline_unix"],
                            prep["split_fingerprint"],
                            cancel,
                            persist,
                        )
                    ] = entry
                    created = True
                    break
                if not created:
                    receipt["failures"].append(
                        f"Seed {seed} exhausted bounded fresh-offer attempts"
                    )
                    persist()
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    receipt["failures"].append(str(exc))
                    h.log(str(exc))
                persist()
        completed = [e for e in receipt["instances"] if e.get("training_complete")]
        if len(completed) == 3 and any(
            e["generation_rows"] != completed[0]["generation_rows"] for e in completed[1:]
        ):
            receipt["failures"].append("Generation rows differ across seeds")
    except BaseException as exc:
        cancel.set()
        receipt["failures"].append(f"{type(exc).__name__}: {exc}")
    finally:
        cancel.set()
        try:
            # Exact predeclared labels cover an uncertain create response too.
            for instance in owned_instances(receipt):
                ident = int(instance["id"])
                result = destroy_verified(ident)
                matching = next(
                    (e for e in receipt["instances"] if e["instance_id"] == ident), None
                )
                if matching is not None:
                    matching["cleanup"] = result
                else:
                    receipt.setdefault("uncertain_create_cleanup", {})[str(ident)] = result
            receipt["remaining_created_instance_ids"] = [
                int(i["id"]) for i in owned_instances(receipt)
            ]
            receipt["cleanup_complete"] = not receipt["remaining_created_instance_ids"]
        except Exception as exc:
            receipt["cleanup_verification_error"] = f"{type(exc).__name__}: {exc}"
        receipt["finished_unix"] = time.time()
        receipt["estimated_compute_disk_usd"] = sum(
            (e.get("cleanup", {}).get("destroyed_unix", time.time()) - e["created_unix"])
            / 3600
            * float(e["offer"]["dph_total"])
            for e in receipt["instances"]
        )
        persist()
        h.save(
            output / "billing_and_cleanup_receipt.json",
            invoice_receipt(receipt)
            | {
                "all_created_resources_destroyed": receipt["cleanup_complete"],
                "remaining_created_ids": receipt.get("remaining_created_instance_ids"),
                "estimated_compute_disk_usd": receipt["estimated_compute_disk_usd"],
                "budget_usd": BUDGET_USD,
            },
        )
    if receipt["failures"] or not receipt["cleanup_complete"]:
        raise SystemExit(1)


def self_test():
    """Pure local safety regression checks; never calls Vast or SSH."""
    import unittest.mock as mock

    offers = [
        {
            "id": i,
            "machine_id": i,
            "gpu_name": "RTX 4090",
            "num_gpus": 1,
            "dph_total": 0.60,
            "inet_up_cost": 0.03,
            "inet_down_cost": 0.03,
        }
        for i in range(3)
    ]
    _, estimate = select_offers(offers)
    assert estimate["conservative_total_usd"] <= 4
    assert "--revision" in training_args(0) and REVISION in training_args(0)
    assert training_args(0)[-1] == "3600"
    for invalid in ([offers[0]] * 3, [dict(o, dph_total=0.61) for o in offers]):
        try:
            select_offers(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid budget/host allocation accepted")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        entry = {"seed": 0, "instance_id": 99, "created_unix": time.time()}
        with (
            mock.patch(__name__ + ".wait_ready", side_effect=RuntimeError("fixture failure")),
            mock.patch.object(h, "api", return_value={}),
            mock.patch(
                __name__ + ".destroy_verified", return_value={"verified_absent": True}
            ) as cleanup,
        ):
            try:
                worker(
                    entry,
                    root,
                    root / "unused.tar.gz",
                    time.time() + 60,
                    "fixture",
                    threading.Event(),
                    lambda: None,
                )
            except RuntimeError:
                pass
            cleanup.assert_called_once_with(99)
            assert (root / "seed_0/failure.log").exists()
        archive = root / "bad.tar.gz"
        with tarfile.open(archive, "w:gz") as tf:
            member = tarfile.TarInfo("../escape")
            member.size = 1
            tf.addfile(member, io.BytesIO(b"x"))
        try:
            extract_verified(archive, root / "extract")
        except ValueError:
            pass
        else:
            raise AssertionError("Unsafe archive was accepted")
    print(
        json.dumps(
            {
                "self_test": "passed",
                "network_calls": 0,
                "max_envelope_usd": estimate["conservative_total_usd"],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run", action="store_true", help="CPU checks and snapshot only; no network/API calls"
    )
    mode.add_argument(
        "--quote",
        action="store_true",
        help="Search/save concrete offers and cost for a prepared output; no rental",
    )
    mode.add_argument(
        "--launch",
        action="store_true",
        help="Run three GPUs from the prepared output and fresh recorded quote",
    )
    mode.add_argument("--watchdog", type=Path, help=argparse.SUPPRESS)
    mode.add_argument(
        "--self-test", action="store_true", help="Local safety fixtures; no network/API calls"
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/memory_forecast_20260924_v1")
    parser.add_argument(
        "--out",
        type=Path,
        help="Fresh directory for --dry-run; same prepared directory for --quote/--launch",
    )
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog.resolve())
        return
    if args.self_test:
        self_test()
        return
    if args.out is None:
        parser.error("--out must name a fresh directory")
    if args.quote or args.launch:
        output = args.out.resolve()
        package, prepared = prepared_output(output)
        if args.quote:
            quote(output, package)
        else:
            execute(output, package, prepared)
        return
    # Refuse partial data before even reserving an output directory.
    dataset_ready(args.data)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    package, prepared = prepare_snapshot(args.data, output)
    if args.dry_run:
        h.save(
            output / "dry_run.json",
            {
                "status": "ready",
                "network_calls": 0,
                "package_sha256": h.sha(package),
                "package_bytes": package.stat().st_size,
                "preparation_manifest_sha256": h.sha(output / "cpu_preparation/manifest.json"),
                "source_manifest_sha256": h.sha(output / "source_manifest.json"),
                "split_fingerprint": prepared["split_fingerprint"],
                "audit": prepared.get("audit"),
                "commands": [training_args(i) for i in range(3)],
                "max_budget_usd": BUDGET_USD,
            },
        )
        print(json.dumps({"dry_run": "ready", "output": str(output), "network_calls": 0}))
        return


if __name__ == "__main__":
    main()
