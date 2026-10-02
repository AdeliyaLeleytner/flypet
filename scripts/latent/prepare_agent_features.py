#!/usr/bin/env python3
"""Cache frozen contextual LLM states for simulator-teacher demonstrations."""

import argparse, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet.latent_inference import LatentModels
from flypet.latent_agent_env import goal_text
from flypet.neural_records import file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = json.loads((a.teacher / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Incomplete teacher collection")
    for name, digest in manifest["files"].items():
        if file_hash(a.teacher / name) != digest:
            raise ValueError("Teacher artifact hash mismatch")
    models = LatentModels(a.bundle)
    rows = []
    features = []
    actions = []
    fine = []
    pop = []
    references = []
    for folder in sorted(a.teacher.glob("task-*")):
        episode = json.loads((folder / "episode.json").read_text())
        task = episode["task"]
        with np.load(folder / "features.npz", allow_pickle=False) as z:
            for example in episode["examples"]:
                turn = example["turn"]
                current = {k: z[f"{turn}_features_{k}"] for k in ("fine", "population")}
                reference = {k: z[f"{turn}_reference_{k}"] for k in ("fine", "population")}
                features.append(models.agent_features(current, reference, goal_text(task, turn)))
                fine.append(current["fine"])
                pop.append(current["population"])
                references.append(reference)
                actions.append(example["action"])
                rows.append({"task": task, "turn": turn, "action": example["action"]})
        print(json.dumps({"tasks": len(rows) // 2, "wall_s": time.monotonic() - start}), flush=True)
    train = np.array([r["task"]["split"] == "train" for r in rows])
    mean_state = {
        "fine": np.mean(np.stack(fine)[train], axis=0),
        "population": np.mean(np.stack(pop)[train], axis=0),
    }
    mean_ref = {
        k: np.mean(np.stack([r[k] for r in references])[train], axis=0)
        for k in ("fine", "population")
    }
    task_only = []
    cache = {}
    for row in rows:
        question = goal_text(row["task"], row["turn"])
        if question not in cache:
            cache[question] = models.agent_features(mean_state, mean_ref, question)
        task_only.append(cache[question])
    np.savez_compressed(
        a.out / "features.npz",
        x=np.stack(features),
        task_only_x=np.stack(task_only),
        actions=np.array(actions, np.float32),
        mean_fine=mean_state["fine"],
        mean_population=mean_state["population"],
        mean_reference_fine=mean_ref["fine"],
        mean_reference_population=mean_ref["population"],
    )
    write_json(a.out / "examples.json", rows)
    write_json(
        a.out / "manifest.json",
        {
            "status": "complete",
            "examples": len(rows),
            "wall_s": time.monotonic() - start,
            "teacher_sha256": file_hash(a.teacher / "manifest.json"),
            "bundle_sha256": file_hash(a.bundle / "manifest.json"),
            "files": {n: file_hash(a.out / n) for n in ["features.npz", "examples.json"]},
            "feature_inputs": "neural tokens and written target/remaining decisions only; frozen Qwen/reader",
            "task_only_control": "same goals and model with fixed training-mean neural observations",
        },
    )


if __name__ == "__main__":
    main()
