#!/usr/bin/env python3
"""One authorized infrastructure replacement, reusing the original source archive."""

import json
from pathlib import Path
import subprocess
import sys
import time

import rerun_projector_clean as h


def main():
    parent = Path(sys.argv[1]).resolve()
    original = json.loads((parent / "resource_receipt.json").read_text())
    failed = original["instances"][0]
    assert failed["seed"] == 0
    remaining = h.api("show", "instances")
    assert not any(i["id"] == failed["instance_id"] for i in remaining), (
        "destroy failed seed0 before replacement"
    )
    out = parent / "seed0_infrastructure_recovery"
    out.mkdir(exist_ok=False)
    offers = h.api(
        "search",
        "offers",
        "reliability>0.99 num_gpus=1 gpu_name=RTX_4090 inet_down>500 disk_space>40 inet_up_cost<0.03 inet_down_cost<0.03",
        "--storage",
        40,
        "--limit",
        15,
        "--order",
        "dph_total",
    )
    offer = next(
        o
        for o in offers
        if o["dph_total"] < 0.60 and o["machine_id"] != failed["offer"]["machine_id"]
    )
    offer = {k: offer.get(k) for k in h.OFFER_KEYS}
    receipt = {
        "started_unix": time.time(),
        "deadline_unix": original["deadline_unix"],
        "hard_budget_usd_total_with_original_run": 8,
        "instances": [],
        "reason": "Seed0 SSH host timed out during model download; one replacement authorized by root",
        "source_tar_sha256": h.sha(parent / "source.tar.gz"),
        "cleanup_complete": False,
    }
    path = out / "resource_receipt.json"
    h.save(path, receipt)
    watchdog_log = (out / "watchdog.log").open("wb")
    subprocess.Popen(
        [sys.executable, str(Path(h.__file__).resolve()), "--watchdog", str(path)],
        stdout=watchdog_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        result = h.api(
            "create",
            "instance",
            offer["id"],
            "--image",
            h.IMAGE,
            "--disk",
            40,
            "--ssh",
            "--direct",
            "--cancel-unavail",
            "--label",
            "fly-clean-20260924-seed0-recovery",
        )
        assert result.get("success") and result.get("new_contract"), result
        entry = {
            "seed": 0,
            "instance_id": result["new_contract"],
            "offer": offer,
            "created_unix": time.time(),
        }
        receipt["instances"].append(entry)
        h.save(path, receipt)
        h.log(
            f"created seed0 infrastructure replacement {entry['instance_id']}, ${offer['dph_total']:.4f}/hour"
        )
        h.worker(entry, out, parent / "source.tar.gz", receipt["deadline_unix"])
    finally:
        for entry in receipt["instances"]:
            try:
                latest = h.api("show", "instance", entry["instance_id"])
                entry["final_status"] = {k: latest.get(k) for k in h.STATUS_KEYS}
            except Exception as exc:
                entry["final_status_error"] = str(exc)
            entry["destroy_response"] = h.destroy(entry["instance_id"])
            entry["destroyed_unix"] = time.time()
        current = h.api("show", "instances")
        ids = {i["instance_id"] for i in receipt["instances"]}
        receipt["remaining_created_instance_ids"] = sorted(ids & {i["id"] for i in current})
        receipt["cleanup_complete"] = not receipt["remaining_created_instance_ids"]
        receipt["finished_unix"] = time.time()
        receipt["estimated_compute_storage_usd"] = sum(
            (e["destroyed_unix"] - e["created_unix"]) / 3600 * e["offer"]["dph_total"]
            for e in receipt["instances"]
        )
        h.save(path, receipt)
        h.log(
            f"replacement cleanup complete={receipt['cleanup_complete']}, estimatedcompute+storage=${receipt['estimated_compute_storage_usd']:.3f}"
        )


if __name__ == "__main__":
    main()
