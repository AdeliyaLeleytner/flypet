#!/usr/bin/env python3
"""Fresh-seed end-to-end twins check using the real writer, brain and reader."""

import argparse, json, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet.latent_inference import LatentModels
from flypet.latent_sessions import BrainSession
from flypet.neural_records import file_hash, write_json
from scripts.latent.evaluate_sealed_reader import SUMMARY, AUDIT, CHANGE, direction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "selection", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    lock = json.loads(a.selection.read_text())
    if (
        lock.get("bundle_manifest_sha256") != file_hash(a.bundle / "manifest.json")
        or lock.get("selection_data") != "validation"
    ):
        raise ValueError("Matching model-selection lock required")
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    models = LatentModels(a.bundle)
    texts = [
        "Give the fly moderate exposure to acetic acid. No reward or punishment.",
        "Give the fly moderate exposure to acetic acid. Pair it with a sugar reward.",
        "Give the fly moderate exposure to acetic acid. Pair it with an aversive punishment.",
    ]
    drives = [models.write(text) for text in texts]
    rows = []
    for seed in (94818, 94819, 94820, 94821):
        pair = [BrainSession(a.bundle, seed), BrainSession(a.bundle, seed)]
        history = [[], []]
        try:
            with ThreadPoolExecutor(max_workers=2) as threads:

                def step_both(vectors, learning, labels):
                    futures = [
                        threads.submit(session.step, vector, learning)
                        for session, vector in zip(pair, vectors)
                    ]
                    snapshots = [f.result() for f in futures]
                    for j, snapshot in enumerate(snapshots):
                        history[j].append(
                            {
                                "phase": labels[j],
                                "learning": learning,
                                "telemetry": snapshot["telemetry"],
                            }
                        )
                    return snapshots

                baseline = step_both([drives[0]["drive_hz"]] * 2, False, ["baseline"] * 2)
                keys = ["approach_hz", "avoid_hz", "valence", "spikes", "active_neurons"]
                same = all(
                    baseline[0]["telemetry"][key] == baseline[1]["telemetry"][key] for key in keys
                )
                if not same:
                    raise ValueError("Twin baselines differ")
                for session in pair:
                    session.pin_reference()
                for _ in range(3):
                    step_both(
                        [drives[1]["drive_hz"], drives[2]["drive_hz"]],
                        True,
                        ["reward", "punishment"],
                    )
                step_both(
                    [np.zeros_like(drives[0]["drive_hz"])] * 2, False, ["zero_input_rest"] * 2
                )
                final = step_both([drives[0]["drive_hz"]] * 2, False, ["shared_probe"] * 2)
            for j, snapshot in enumerate(final):
                answers = {
                    kind: models.read(
                        snapshot["current"],
                        snapshot["reference"],
                        question=question,
                        mode="comparison" if kind == "change" else "current",
                    )
                    for kind, question in [
                        ("summary", SUMMARY),
                        ("audit", AUDIT),
                        ("change", CHANGE),
                    ]
                }
                expected = int(np.sign(snapshot["telemetry"]["valence"]))
                predicted = direction(answers["summary"])
                rows.append(
                    {
                        "seed": seed,
                        "fly": "A" if j == 0 else "B",
                        "history": history[j],
                        "answers": answers,
                        "expected_sign": expected,
                        "predicted_sign": predicted,
                        "correct_sign": predicted == expected,
                    }
                )
        finally:
            for session in pair:
                session.close()
        print(
            json.dumps(
                {
                    "seed": seed,
                    "completed_replies": len(rows),
                    "correct": sum(r["correct_sign"] for r in rows),
                }
            ),
            flush=True,
        )
    write_json(a.out / "episodes.json", rows)
    write_json(
        a.out / "drives.json",
        {
            "texts": texts,
            "channel_rates_hz": [d["channel_rates_hz"].tolist() for d in drives],
            "writer_preprocessing_sha256": file_hash(a.bundle / "writer_preprocessing.npz"),
        },
    )
    report = {
        "status": "complete",
        "n": 8,
        "correct_sign": sum(r["correct_sign"] for r in rows),
        "gate_pass": sum(r["correct_sign"] for r in rows) >= 7,
        "same_initial_seed_and_baseline": True,
        "same_final_learned_drive": True,
        "exposure_history_text_to_reader": False,
        "selection": lock,
        "wall_s": time.monotonic() - start,
    }
    write_json(a.out / "report.json", report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
