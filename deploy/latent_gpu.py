#!/usr/bin/env python3
"""One owned, bounded Vast job with a detached watchdog and verified harvest.

The package is a prebuilt allowlisted archive with source_manifest.json. No
credentials are put on the worker. Only resources with this run's exact id or
unguessable label may be removed. All pre-existing resources are left alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time
import uuid

import rerun_projector_clean as h

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "data/latent_gpu_resources"
VAST_PY = Path.home() / ".local/share/uv/tools/vastai/bin/python"
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
COLS = [
    "id",
    "machine_id",
    "gpu_name",
    "gpu_ram",
    "num_gpus",
    "dph_total",
    "dph_base",
    "storage_cost",
    "inet_up_cost",
    "inet_down_cost",
    "reliability2",
    "cpu_cores_effective",
    "cpu_ram",
    "rentable",
    "rented",
    "is_bid",
]


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def offers():
    code = """import contextlib,io,json,sys
from vastai.api.client import VastClient
from vastai.cli.main import main
columns=json.loads(sys.argv[1]);orig=VastClient.put
def put(self,subpath,query_args=None,json_data=None,**kw):
 if subpath=='/search/asks/' and json_data and json_data.get('select_cols')==['*']:
  json_data=dict(json_data,select_cols=columns)
 return orig(self,subpath,query_args=query_args,json_data=json_data,**kw)
VastClient.put=put
sys.argv=['vastai','search','offers','num_gpus=1 gpu_ram>=44 reliability>0.99 inet_down>500 disk_space>60 rented=False rentable=True','--new','--storage','60','--limit','60','--order','dph_total','--raw']
buf=io.StringIO()
with contextlib.redirect_stdout(buf):
 try:main()
 except SystemExit as e:
  if e.code not in (0,None):raise
print(json.dumps([{k:r.get(k) for k in columns} for r in json.loads(buf.getvalue())]))
"""
    value = h.run([str(VAST_PY), "-c", code, json.dumps(COLS)], timeout=45)
    return json.loads(value.stdout)


def owned(receipt):
    return [
        x
        for x in h.api("show", "instances")
        if x.get("id") in receipt.get("instance_ids", []) or x.get("label") == receipt["label"]
    ]


def cleanup(receipt):
    deleted = {}
    for row in owned(receipt):
        if int(row["id"]) not in receipt["instance_ids"]:
            receipt["instance_ids"].append(int(row["id"]))
        deleted[str(row["id"])] = h.destroy(int(row["id"]))
    remaining = [x["id"] for x in owned(receipt)]
    return {"deleted": deleted, "remaining_ids": remaining, "verified_absent": not remaining}


def watchdog(path):
    path = Path(path).resolve()
    if not path.is_relative_to(RESOURCES.resolve()) or path.name != "receipt.json":
        raise ValueError("Watchdog receipt must belong to this campaign")
    while True:
        receipt = json.loads(path.read_text())
        if receipt.get("cleanup_complete"):
            return
        if time.time() >= receipt["deadline_unix"]:
            try:
                result = cleanup(receipt)
                save(path.parent / "watchdog_cleanup.json", result)
                if result["verified_absent"]:
                    return
            except Exception as exc:
                save(path.parent / "watchdog_error.json", {"time": time.time(), "error": str(exc)})
        time.sleep(10)


def seal_fetch(info, out):
    code = """from pathlib import Path
import hashlib,json,shutil
root=Path('/workspace/fly');p=root/'results';p.mkdir(exist_ok=True)
for name in ['stdout.log','setup.log','environment.txt','gpu_environment.txt','source_manifest.json']:
 if (root/name).exists():shutil.copy2(root/name,p/name)
files={str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in p.rglob('*') if x.is_file() and x.name!='SHA256.json'}
(p/'SHA256.json').write_text(json.dumps(files,indent=2))
"""
    h.ssh(info, "python -c " + shlex.quote(code), timeout=60)
    result = h.ssh(info, "tar czf - -C /workspace/fly results", timeout=180)
    archive = out / "results.tar.gz"
    archive.write_bytes(result.stdout)
    with tarfile.open(archive, "r:gz") as tar:
        for entry in tar.getmembers():
            path = Path(entry.name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or path.parts[0] != "results"
                or not (entry.isfile() or entry.isdir())
            ):
                raise ValueError("Unsafe result archive")
        tar.extractall(out, filter="data")
    hashes = json.loads((out / "results/SHA256.json").read_text())
    for name, digest in hashes.items():
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or h.sha(out / "results" / path) != digest:
            raise ValueError("Result hash mismatch")
    return len(hashes)


def wait_ready_owned(receipt, seed):
    deadline = min(receipt["deadline_unix"] - 300, receipt["created_unix"] + 480)
    while time.time() < deadline:
        info = h.api("show", "instance", receipt["instance_ids"][0])
        if isinstance(info, list):
            info = info[0] if info else {}
        if info.get("id") not in receipt["instance_ids"] or info.get("label") != receipt["label"]:
            raise RuntimeError("Owned lease identity disappeared during startup")
        message = str(info.get("status_msg", ""))
        if (
            "port is already allocated" in message
            or "failed to set up container networking" in message
        ):
            save(
                RESOURCES / receipt["id"] / "startup_failure.json",
                {
                    "failure_scope": "provider_startup",
                    "instance_id": info["id"],
                    "machine_id": receipt["offer"]["machine_id"],
                    "reason": "container networking failure",
                },
            )
            raise RuntimeError("Provider container networking failed")
        if info.get("actual_status") == "running" and info.get("ssh_host") and info.get("ssh_port"):
            result = h.ssh(
                info,
                "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'",
                timeout=35,
                check=False,
            )
            if result.returncode == 0 and b"True" in result.stdout:
                h.log(f"seed {seed} instance {info['id']} ready: {result.stdout.decode().strip()}")
                return info
        time.sleep(10)
    raise TimeoutError("GPU did not become ready within startup bound")


def launch(args):
    package = args.package.resolve()
    command = json.loads(args.command_json.read_text())
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("Job command must be a JSON list of arguments")
    if not 0 < args.hours <= 3 or not 0 < args.max_rate <= 1.5 or not 0 < args.job_cap <= 10:
        raise ValueError("Lease bounds exceed this launcher limit")
    # Aggregate conservative reservations from this new plan only.
    prior = 0.0
    for p in RESOURCES.glob("*/receipt.json"):
        r = json.loads(p.read_text())
        prior += r.get("cost_bound_used_usd", r["reserved_usd"])
    reserved = args.hours * args.max_rate + 1.5
    if reserved > args.job_cap or prior + reserved > 65:
        raise ValueError("Initial LLM-phase allocation would be exceeded")
    with tarfile.open(package, "r:gz") as tar:
        names = {x.name for x in tar.getmembers()}
        if "source_manifest.json" not in names:
            raise ValueError("Package has no source manifest")
        for entry in tar.getmembers():
            p = Path(entry.name)
            if p.is_absolute() or ".." in p.parts or not (entry.isfile() or entry.isdir()):
                raise ValueError("Unsafe package member")
    ident = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    out = RESOURCES / ident
    out.mkdir(parents=True)
    receipt = {
        "schema": "latent-gpu-v1",
        "id": ident,
        "label": "fly-latent-" + ident,
        "instance_ids": [],
        "created_intent_unix": time.time(),
        "deadline_unix": time.time() + args.hours * 3600,
        "reserved_usd": reserved,
        "package_sha256": h.sha(package),
        "command": command,
        "cleanup_complete": False,
    }
    path = out / "receipt.json"
    save(path, receipt)
    with (out / "watchdog.log").open("wb") as log:
        guard = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--watchdog", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    receipt["watchdog_pid"] = guard.pid
    save(path, receipt)

    def stopped(*_):
        raise KeyboardInterrupt("Controller interrupted")

    signal.signal(signal.SIGTERM, stopped)
    info = None
    error = None
    try:
        # Prefer recent architectures with enough RAM/CPU, not the slowest cheap 48GB card.
        supported = {
            "RTX 6000Ada",
            "RTX A6000",
            "L40S",
            "L40",
            "RTX 5880Ada",
            "A100 PCIE",
            "A100 SXM4",
            "A800 PCIE",
        }
        failed_machines = {
            json.loads(p.read_text())["machine_id"]
            for p in RESOURCES.glob("*/startup_failure.json")
        }
        candidates = [
            r
            for r in offers()
            if r.get("gpu_name") in supported
            and r.get("rentable")
            and not r.get("rented")
            and r.get("machine_id") not in failed_machines
            and not r.get("is_bid")
            and 0 < float(r["dph_total"]) <= args.max_rate
            and max(float(r["inet_up_cost"]), float(r["inet_down_cost"])) <= 0.03
            and float(r.get("cpu_cores_effective") or 0) >= args.min_cpu
            and float(r.get("cpu_ram") or 0) >= args.min_ram_gb * 1000
        ]
        if not candidates:
            raise RuntimeError("No eligible fresh GPU offer")
        candidates.sort(key=lambda r: (r["gpu_name"] == "RTX A6000", float(r["dph_total"])))
        for offer in candidates[:3]:
            receipt.setdefault("offers_tried", []).append(offer)
            save(path, receipt)
            result = h.api(
                "create",
                "instance",
                offer["id"],
                "--image",
                IMAGE,
                "--disk",
                "60",
                "--ssh",
                "--direct",
                "--label",
                receipt["label"],
            )
            if result.get("success") and result.get("new_contract"):
                receipt["instance_ids"] = [int(result["new_contract"])]
                receipt["offer"] = offer
                receipt["created_unix"] = time.time()
                save(path, receipt)
                break
        if not receipt["instance_ids"]:
            raise RuntimeError("No offer creation succeeded")
        info = wait_ready_owned(receipt, args.seed)
        if info.get("label") != receipt["label"] or float(info["dph_total"]) > args.max_rate:
            raise RuntimeError("Lease identity or price differs")
        receipt["live_rate"] = float(info["dph_total"])
        save(path, receipt)
        h.ssh(
            info,
            "mkdir -p /workspace/fly && tar xzf - -C /workspace/fly",
            input=package.read_bytes(),
            timeout=180,
        )
        setup = """set -euo pipefail
cd /workspace/fly
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 TOKENIZERS_PARALLELISM=false
python -m pip install --disable-pip-version-check 'numpy==2.2.6' 'transformers==5.17.0' 'peft==0.21.0' accelerate safetensors >setup.log 2>&1
python -m pip freeze >environment.txt
nvidia-smi >gpu_environment.txt
python - <<'PY'
from pathlib import Path
import json,hashlib
for name,digest in json.loads(Path('source_manifest.json').read_text()).items():
 assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest,name
PY
"""
        response = h.ssh(info, "bash -s", input=setup.encode(), timeout=600, check=False)
        (out / "setup_stdout.log").write_bytes(response.stdout + response.stderr)
        if response.returncode:
            raise RuntimeError("Remote setup failed")
        seconds = max(1, int(receipt["deadline_unix"] - time.time() - 300))
        job = """#!/bin/bash
set -uo pipefail
cd /workspace/fly
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
timeout --kill-after=20s %ss %s >stdout.log 2>&1
rc=$?
mkdir -p results
echo "$rc" >results/exit_code.txt
touch results/DONE
""" % (seconds, shlex.join(command))
        h.ssh(info, "cat > /workspace/fly/job.sh", input=job.encode())
        h.ssh(
            info, "nohup bash /workspace/fly/job.sh </dev/null >/workspace/fly/dispatch.log 2>&1 &"
        )
        receipt["job_started_unix"] = time.time()
        save(path, receipt)
        last = ""
        while time.time() < receipt["deadline_unix"] - 240:
            result = h.ssh(
                info,
                "test -f /workspace/fly/results/DONE; rc=$?; tail -2 /workspace/fly/stdout.log; exit $rc",
                timeout=45,
                check=False,
            )
            current = result.stdout.decode(errors="replace")[-1600:]
            if current != last:
                print(current, flush=True)
                last = current
            if result.returncode == 0:
                break
            time.sleep(20)
        else:
            raise TimeoutError("Lease reached cleanup reserve")
        receipt["verified_result_files"] = seal_fetch(info, out)
        receipt["result_path"] = str((out / "results").relative_to(ROOT))
        save(path, receipt)
        if (out / "results/exit_code.txt").read_text().strip() != "0":
            raise RuntimeError("Remote job failed; artifacts preserved")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        receipt["error"] = error
        save(path, receipt)
        if info is not None:
            try:
                receipt["partial_verified_files"] = seal_fetch(info, out)
            except Exception as e:
                receipt["harvest_error"] = str(e)
    finally:
        try:
            result = cleanup(receipt)
            receipt["cleanup"] = result
            receipt["cleanup_complete"] = result["verified_absent"]
        except Exception as exc:
            receipt["cleanup_error"] = str(exc)
            receipt["cleanup_complete"] = False
        receipt["finished_unix"] = time.time()
        hours = (
            receipt["finished_unix"] - receipt.get("created_unix", receipt["created_intent_unix"])
        ) / 3600
        receipt["cost_bound_used_usd"] = (
            hours * receipt.get("live_rate", args.max_rate) + 1.5 if receipt["instance_ids"] else 0
        )
        save(path, receipt)
        print(
            json.dumps(
                {
                    "receipt": str(path.relative_to(ROOT)),
                    "error": error,
                    "cleanup_complete": receipt["cleanup_complete"],
                    "cost_bound_usd": receipt["cost_bound_used_usd"],
                }
            ),
            flush=True,
        )
    if error or not receipt["cleanup_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--watchdog", type=Path)
    parser.add_argument("--quote", action="store_true")
    parser.add_argument("--package", type=Path)
    parser.add_argument("--command-json", type=Path)
    parser.add_argument("--hours", type=float, default=3)
    parser.add_argument("--max-rate", type=float, default=1.2)
    parser.add_argument("--job-cap", type=float, default=6)
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--min-cpu", type=int, default=8)
    parser.add_argument("--min-ram-gb", type=int, default=24)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.watchdog)
    elif args.quote:
        print(json.dumps(offers()))
    elif args.package and args.command_json:
        launch(args)
    else:
        parser.error("Supply --quote, --watchdog, or a package and command")
