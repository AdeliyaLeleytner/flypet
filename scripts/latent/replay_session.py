#!/usr/bin/env python3
"""Replay exported neural drives without loading a language model."""

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet.latent_sessions import BrainSession
from flypet.neural_records import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    record = json.loads(a.session.read_text())
    manifest = json.loads((a.bundle / "manifest.json").read_text())
    if (
        record.get("schema_version") != 1
        or record.get("initial_memory") != "naive"
        or record.get("step_ms") != 250
    ):
        raise ValueError("Unsupported session contract")
    if record["model_manifest"] != manifest:
        raise ValueError("Session and model bundle differ")
    with np.load(a.bundle / "writer_preprocessing.npz", allow_pickle=False) as z:
        basis = z["basis"]
    session = BrainSession(a.bundle, record["seed"])
    checks = []
    try:
        for index, event in enumerate(record["events"]):
            replay = event.get("replay", {})
            mode = event["mode"]
            snapshot = None
            if replay.get("initial_zero_probe"):
                snapshot = session.observe()
            if mode == "interact":
                channels = np.asarray(replay["channel_rates_hz"], dtype=np.float32)
                snapshot = session.step(basis @ channels, event["learning"])
            elif mode == "pin_reference":
                snapshot = session.pin_reference()
            elif mode in ("ask", "audit", "compare"):
                snapshot = session.observe()
            elif mode == "state_swap":
                continue
            else:
                raise ValueError("Unsupported replay event")
            expected = event["telemetry"]
            actual = snapshot["telemetry"]
            errors = {}
            for key in (
                "approach_hz",
                "avoid_hz",
                "valence",
                "active_neurons",
                "spikes",
                "memory_strength",
                "episode_time_ms",
                "steps",
            ):
                if abs(actual[key] - expected[key]) > 1e-6:
                    errors[key] = {"expected": expected[key], "actual": actual[key]}
            checks.append({"event": index, "ok": not errors, "errors": errors})
    finally:
        session.close()
    result = {
        "ok": all(c["ok"] for c in checks),
        "events_checked": len(checks),
        "checks": checks,
        "language_generation_replayed": False,
    }
    write_json(a.out, result)
    print(json.dumps(result))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
