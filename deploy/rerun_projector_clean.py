#!/usr/bin/env python3
"""Bounded, isolated three-seed Vast rerun; never changes runtime checkpoints.

Call only after reviewing the corrected training code. Uses signed-in vastai/SSH,
copies an explicit allowlist, verifies fetched file hashes, destroys every created
instance, and leaves a machine-readable resource/cost receipt. An independent
watchdog also destroys these exact IDs after three hours if orchestration dies.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime"
MAX_SECONDS = 3 * 3600
OFFER_KEYS = (
    "id",
    "machine_id",
    "gpu_name",
    "num_gpus",
    "dph_total",
    "dph_base",
    "storage_cost",
    "inet_down_cost",
    "inet_up_cost",
    "reliability2",
    "geolocation",
)
STATUS_KEYS = (
    "id",
    "label",
    "actual_status",
    "intended_status",
    "ssh_host",
    "ssh_port",
    "dph_total",
    "start_date",
    "gpu_name",
    "num_gpus",
    "total_dph",
    "inet_up_cost",
    "inet_down_cost",
    "inet_up_billed",
    "inet_down_billed",
    "storage_cost",
    "storage_total_cost",
    "disk_space",
)


def run(args, *, timeout=120, check=True, input=None):
    p = subprocess.run(args, input=input, capture_output=True, timeout=timeout)
    if check and p.returncode:
        raise RuntimeError(
            f"{args[0]} exited {p.returncode}: {p.stderr.decode(errors='replace')[-2000:]}"
        )
    return p


def api(*args):
    p = run(["vastai", *map(str, args), "--raw"])
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        # Some vastai releases print a human confirmation for destroy --raw.
        if args[:2] == ("destroy", "instance"):
            ident = int(args[2])
            remaining = api("show", "instances")
            if not any(i["id"] == ident for i in remaining):
                return {"success": True, "verified_absent_after_destroy": True}
        raise


def save(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def destroy(ident):
    for attempt in range(3):
        try:
            result = api("destroy", "instance", ident, "--yes")
            log(f"destroy instance {ident}: {result}")
            return result
        except Exception as exc:
            log(f"destroy retry {attempt + 1} for {ident}: {exc}")
            time.sleep(5)
    return {"success": False, "error": "destruction retries exhausted"}


def watchdog(path):
    while True:
        data = json.loads(path.read_text())
        if data.get("cleanup_complete"):
            return
        remaining = data["deadline_unix"] - time.time()
        if remaining <= 0:
            for entry in data.get("instances", []):
                destroy(entry["instance_id"])
            return
        time.sleep(min(30, remaining))


def ssh_args(info):
    return [
        "ssh",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(info["ssh_port"]),
        f"root@{info['ssh_host']}",
    ]


def ssh(info, command, *, timeout=120, check=True, input=None):
    return run(ssh_args(info) + [command], timeout=timeout, check=check, input=input)


def snapshot(out):
    files = sorted((ROOT / "flypet").glob("*.py"))
    files += [ROOT / "scripts/train_projector.py"]
    files += sorted((ROOT / "scripts").glob("projector_*.py"))
    files += [
        ROOT / "data/flywire_annotations_783.tsv",
        ROOT / "vendor/Drosophila_brain_model/Completeness_783.csv",
    ]
    for folder in ("state_dataset", "state_dataset_b", "state_dataset_c", "state_pairs"):
        files += sorted((ROOT / "data" / folder).glob("shard_*.npz"))
    manifest = {
        str(p.relative_to(ROOT)): {"sha256": sha(p), "bytes": p.stat().st_size} for p in files
    }
    save(out / "source_manifest.json", manifest)
    package = out / "source.tar.gz"
    with tarfile.open(package, "w:gz") as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(ROOT)), recursive=False)
        tf.add(out / "source_manifest.json", arcname="source_manifest.json")
    return package


def wait_ready(entry, deadline):
    ident = entry["instance_id"]
    while time.time() < min(deadline, entry["created_unix"] + 600):
        info = api("show", "instance", ident)
        if isinstance(info, list):
            info = info[0]
        entry["status"] = {k: info.get(k) for k in STATUS_KEYS}
        if info.get("ssh_host") and info.get("ssh_port") and info.get("actual_status") == "running":
            probe = ssh(
                info,
                "python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'",
                check=False,
            )
            if probe.returncode == 0:
                log(f"seed {entry['seed']} instance {ident} ready: {probe.stdout.decode().strip()}")
                return info
        time.sleep(15)
    raise TimeoutError(f"instance {ident} not ready after 10 minutes")


def worker(entry, out, package, deadline):
    seed = entry["seed"]
    local = out / f"seed_{seed}"
    local.mkdir()
    info = wait_ready(entry, deadline)
    entry["ssh"] = {k: info[k] for k in ("ssh_host", "ssh_port")}
    ssh(
        info,
        "mkdir -p /workspace/fly && tar xzf - -C /workspace/fly",
        input=package.read_bytes(),
        timeout=300,
    )
    setup = """set -euo pipefail
cd /workspace/fly
python -m pip install --disable-pip-version-check --no-cache-dir 'transformers==5.17.0' 'pandas==2.2.3' 'numpy==2.2.6' 'accelerate' 'safetensors' > setup.log 2>&1
python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path
import json
p = snapshot_download('Qwen/Qwen3-0.6B')
Path('model_snapshot.json').write_text(json.dumps({'model':'Qwen/Qwen3-0.6B', 'snapshot_commit':Path(p).name},indent=2)+'\\n')
PY
python -m pip freeze > environment.txt
python - <<'PY'
import json, hashlib
from pathlib import Path
for name, meta in json.loads(Path('source_manifest.json').read_text()).items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == meta['sha256'], name
PY
nvidia-smi > gpu_environment.txt
"""
    p = ssh(info, "bash -s", input=setup.encode(), timeout=600, check=False)
    (local / "setup_stdout.log").write_bytes(p.stdout + p.stderr)
    if p.returncode:
        detail = ssh(info, "tail -60 /workspace/fly/setup.log", check=False).stdout
        (local / "setup_failure.log").write_bytes(detail)
        raise RuntimeError(f"seed {seed} setup failed; see {local}")
    # The split and generation sample are shared; only the training seed differs.
    args = [
        "python",
        "-u",
        "scripts/train_projector.py",
        "--model",
        "Qwen/Qwen3-0.6B",
        "--epochs",
        "4",
        "--tokens",
        "16",
        "--batch",
        "8",
        "--dtype",
        "bfloat16",
        "--n_gen",
        "100",
        "--split-seed",
        "0",
        "--seed",
        str(seed),
        "--device",
        "cuda",
        "--out",
        "/workspace/fly/results",
    ]
    entry["command"] = shlex.join(args)
    remote = """#!/usr/bin/env bash
set -uo pipefail
cd /workspace/fly
export FLYPET_ROOT=/workspace/fly FLYPET_DATA=/workspace/fly/data HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
%s > stdout.log 2>&1
rc=$?
mkdir -p results
cp source_manifest.json environment.txt gpu_environment.txt model_snapshot.json stdout.log results/
echo "$rc" > results/exit_code.txt
python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('results')
d={str(x.relative_to(p)):hashlib.sha256(x.read_bytes()).hexdigest() for x in p.rglob('*') if x.is_file() and x.name != 'SHA256.json'}
(p/'SHA256.json').write_text(json.dumps(d,sort_keys=True,indent=2)+'\\n')
PY
touch results/DONE
""" % shlex.join(args)
    ssh(
        info, "cat > /workspace/fly/run.sh && chmod +x /workspace/fly/run.sh", input=remote.encode()
    )
    ssh(info, "nohup bash /workspace/fly/run.sh </dev/null >/workspace/fly/launcher.log 2>&1 &")
    log(f"seed {seed} launched on instance {entry['instance_id']}")
    previous = ""
    while time.time() < deadline - 180:
        p = ssh(
            info,
            "test -f /workspace/fly/results/DONE; done=$?; tail -3 /workspace/fly/stdout.log; exit $done",
            check=False,
        )
        progress = p.stdout.decode(errors="replace").strip()
        if progress != previous:
            log(f"seed {seed}: {progress[-1200:]}")
            previous = progress
        if p.returncode == 0:
            break
        time.sleep(30)
    else:
        ssh(info, "pkill -TERM -f '[p]ython -u scripts/train_projector.py' || true", check=False)
        raise TimeoutError(f"seed {seed} exceeded bounded rental deadline")
    fetched = ssh(info, "tar czf - -C /workspace/fly results", timeout=300)
    archive = local / "results.tar.gz"
    archive.write_bytes(fetched.stdout)
    with tarfile.open(archive, "r:gz") as tf:
        tf.extractall(local, filter="data")
    results = local / "results"
    checksums = json.loads((results / "SHA256.json").read_text())
    for name, expected in checksums.items():
        if sha(results / name) != expected:
            raise RuntimeError(f"fetched checksum mismatch: seed {seed}, {name}")
    rc = int((results / "exit_code.txt").read_text())
    if rc == 0:
        for required in (
            "brain_projector.pt",
            "direct_projector.pt",
            "manifest.json",
            "report.json",
            "predictions.json",
        ):
            if required not in checksums:
                raise RuntimeError(f"seed {seed} missing required artifact: {required}")
    entry["completed_unix"] = time.time()
    entry["verified_local_files"] = len(checksums)
    entry["exit_code"] = rc
    entry["result_archive_sha256"] = sha(archive)
    save(local / "receipt.json", entry)
    log(f"seed {seed}: fetched and SHA256-verified {len(checksums)} files, exit {rc}")
    if rc:
        raise RuntimeError(f"seed {seed} training failed with exit {rc}")
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path)
    ap.add_argument("--launch", action="store_true")
    ap.add_argument("--watchdog", type=Path)
    a = ap.parse_args()
    if a.watchdog:
        watchdog(a.watchdog)
        return
    if not a.launch or a.out is None:
        ap.error("requires --launch and fresh --out directory")
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    package = snapshot(out)
    offers = api(
        "search",
        "offers",
        "reliability>0.99 num_gpus=1 gpu_name=RTX_4090 inet_down>500 disk_space>40 inet_up_cost<0.03 inet_down_cost<0.03",
        "--storage",
        40,
        "--limit",
        12,
        "--order",
        "dph_total",
    )
    selected = [o for o in offers if o["dph_total"] < 0.60][:3]
    if len(selected) < 3:
        raise RuntimeError(
            "fewer than three suitable offers below $0.60/hour; no resources created"
        )
    selected = [{k: o.get(k) for k in OFFER_KEYS} for o in selected]
    save(out / "selected_offers.json", selected)
    receipt = {
        "started_unix": time.time(),
        "deadline_unix": time.time() + MAX_SECONDS,
        "hard_budget_usd": 8.0,
        "max_runtime_seconds": MAX_SECONDS,
        "image": IMAGE,
        "estimated_max_compute_storage_usd": sum(o["dph_total"] for o in selected) * 3,
        "source_tar_sha256": sha(package),
        "instances": [],
        "cleanup_complete": False,
    }
    path = out / "resource_receipt.json"
    save(path, receipt)
    watchdog_log = (out / "watchdog.log").open("wb")
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--watchdog", str(path)],
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    def terminate(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    failures = []
    try:
        for seed, offer in enumerate(selected):
            result = api(
                "create",
                "instance",
                offer["id"],
                "--image",
                IMAGE,
                "--disk",
                40,
                "--ssh",
                "--direct",
                "--cancel-unavail",
                "--label",
                f"fly-clean-20260924-seed{seed}",
            )
            if not result.get("success") or not result.get("new_contract"):
                raise RuntimeError(f"provision failed: {result}")
            entry = {
                "seed": seed,
                "instance_id": result["new_contract"],
                "offer": offer,
                "created_unix": time.time(),
            }
            receipt["instances"].append(entry)
            save(path, receipt)
            log(
                f"created seed {seed}: instance {entry['instance_id']}, ${offer['dph_total']:.4f}/hour"
            )
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = {
                pool.submit(worker, e, out, package, receipt["deadline_unix"]): e
                for e in receipt["instances"]
            }
            for future in as_completed(futures):
                entry = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    entry["error"] = str(exc)
                    failures.append(str(exc))
                    log(str(exc))
                finally:
                    try:
                        latest = api("show", "instance", entry["instance_id"])
                        entry["final_status"] = {k: latest.get(k) for k in STATUS_KEYS}
                    except Exception as exc:
                        entry["final_status_error"] = str(exc)
                    entry["destroy_response"] = destroy(entry["instance_id"])
                    entry["destroyed_unix"] = time.time()
                    save(path, receipt)
    finally:
        for entry in receipt["instances"]:
            if not entry.get("destroy_response", {}).get("success"):
                entry["destroy_response"] = destroy(entry["instance_id"])
                entry["destroyed_unix"] = time.time()
        current = api("show", "instances")
        remaining = {i["id"] for i in current} & {e["instance_id"] for e in receipt["instances"]}
        receipt["remaining_created_instance_ids"] = sorted(remaining)
        receipt["cleanup_complete"] = not remaining
        receipt["finished_unix"] = time.time()
        receipt["estimated_compute_storage_usd"] = sum(
            (e.get("destroyed_unix", time.time()) - e["created_unix"])
            / 3600
            * e["offer"]["dph_total"]
            for e in receipt["instances"]
        )
        receipt["failures"] = failures
        save(path, receipt)
        log(
            f"cleanup complete={receipt['cleanup_complete']}; estimated compute+storage=${receipt['estimated_compute_storage_usd']:.3f}"
        )
    if failures or not receipt["cleanup_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
