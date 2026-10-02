#!/usr/bin/env python3
"""Behavioral SFT of a continuous action head on frozen contextual LM states."""

import argparse, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from flypet.latent_policy import ContinuousPolicy
from flypet.neural_records import write_json, file_hash


def fit(x, actions, splits, out, epochs=500, seed=928, initial_directory=None):
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    torch.set_num_threads(4)
    train = np.flatnonzero(splits == "train")
    val = np.flatnonzero(splits == "validation")
    mean = x[train].mean(0)
    std = np.maximum(x[train].std(0), 0.001)
    if initial_directory:
        with np.load(Path(initial_directory) / "normalization.npz", allow_pickle=False) as z:
            mean = z["mean"]
            std = z["std"]
    features = torch.tensor(np.clip((x - mean) / std, -10, 10), dtype=torch.float32)
    targets = torch.tensor(actions, dtype=torch.float32)
    policy = ContinuousPolicy(x.shape[1], hidden=64)
    if initial_directory:
        policy.load_state_dict(load_file(str(Path(initial_directory) / "policy.safetensors")))
    optim = torch.optim.AdamW(policy.parameters(), lr=0.001, weight_decay=0.03)
    rng = np.random.default_rng(seed)
    best = float("inf")
    history = []
    for epoch in range(epochs):
        policy.train()
        order = rng.permutation(train)
        for k in range(0, len(order), 32):
            ids = order[k : k + 32]
            optim.zero_grad(set_to_none=True)
            loss = policy.imitation_loss(features[ids], targets[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)
            optim.step()
        policy.eval()
        with torch.no_grad():
            vl = float(policy.imitation_loss(features[val], targets[val]))
        if vl < best:
            best = vl
            selected = epoch + 1
            save_file(
                {k: v.contiguous() for k, v in policy.state_dict().items()},
                str(out / "policy.safetensors"),
            )
        if (epoch + 1) % 25 == 0:
            history.append({"epoch": epoch + 1, "validation_gaussian_nll_without_constant": vl})
    save_file(
        {k: v.contiguous() for k, v in policy.state_dict().items()},
        str(out / "policy_last.safetensors"),
    )
    policy.load_state_dict(load_file(str(out / "policy.safetensors")))
    metrics = {}
    with torch.no_grad():
        for name in ("train", "validation"):
            ids = np.flatnonzero(splits == name)
            pred, _, _ = policy.sample(features[ids], deterministic=True)
            metrics[name] = {
                "normalized_action_mae": float(
                    ((pred - targets[ids]).abs() / policy.action_high).mean()
                ),
                "gaussian_nll_without_constant": float(
                    policy.imitation_loss(features[ids], targets[ids])
                ),
            }
    np.savez_compressed(out / "normalization.npz", mean=mean, std=std)
    write_json(
        out / "config.json",
        {
            "feature_dim": x.shape[1],
            "hidden": 64,
            "action_high": policy.action_high.tolist(),
            "selected_epoch": selected,
        },
    )
    write_json(out / "training.json", history)
    write_json(
        out / "report.json",
        {
            "status": "complete",
            "selection": "validation Gaussian NLL; final rollout selection pending; no test access",
            "metrics": metrics,
        },
    )
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    manifest = json.loads((a.data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(a.data / name) != digest:
            raise ValueError("Feature cache hash mismatch")
    rows = json.loads((a.data / "examples.json").read_text())
    splits = np.array([r["task"]["split"] for r in rows])
    results = {}
    with np.load(a.data / "features.npz", allow_pickle=False) as z:
        for kind, key in [("sft", "x"), ("task_only_sft", "task_only_x")]:
            results[kind] = fit(z[key], z["actions"], splits, a.out / kind)
    write_json(
        a.out / "report.json",
        {
            "status": "complete",
            "metrics": results,
            "wall_s": time.monotonic() - start,
            "feature_manifest_sha256": file_hash(a.data / "manifest.json"),
        },
    )
    print(json.dumps(results), flush=True)


if __name__ == "__main__":
    main()
