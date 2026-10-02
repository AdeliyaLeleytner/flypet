#!/usr/bin/env python3
"""Locate reader information loss using frozen latent and neural-feature probes."""

import argparse, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from safetensors.torch import load_file
from flypet.latent_dialogue import DialogueReader
from flypet.neural_records import write_json, file_hash


def ridge_probe(x, y, split):
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "validation")
    test = np.flatnonzero(split == "development_test")
    mean = x[train].mean(0)
    std = np.maximum(x[train].std(0), 0.01)
    xx = np.clip((x - mean) / std, -20, 20).astype(np.float64)
    ym = y[train].mean(0)
    ys = np.maximum(y[train].std(0), 0.01)
    yy = (y - ym) / ys
    kernel = xx[train] @ xx[train].T
    best = None
    for alpha in (1.0, 10.0, 100.0, 1000.0, 10000.0):
        weights = xx[train].T @ np.linalg.solve(kernel + alpha * np.eye(len(train)), yy[train])
        pred = xx[val] @ weights * ys + ym
        error = float(abs(pred[:, -1] - y[val, -1]).mean())
        if best is None or error < best[0]:
            best = (error, alpha, weights)
    return {
        "alpha": best[1],
        "validation_mae": abs(xx[val] @ best[2] * ys + ym - y[val]).mean(0).tolist(),
        "development_test_mae": abs(xx[test] @ best[2] * ys + ym - y[test]).mean(0).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    start = time.monotonic()
    torch.set_num_threads(4)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    config = json.loads((a.bundle / "reader_config.json").read_text())
    fields = [
        "n_neurons",
        "channels",
        "metadata",
        "n_populations",
        "embedding_dim",
        "embedding_std",
        "tokens",
        "width",
        "auxiliary_dim",
    ]
    model = DialogueReader(**{k: config[k] for k in fields}).to(device).eval()
    model.load_state_dict(load_file(str(a.bundle / "reader.safetensors")))
    captured = []
    hook = model.latent_norm.register_forward_hook(
        lambda module, args, result: captured.append(
            result.detach().cpu().numpy().reshape(len(result), -1)
        )
    )
    rows = [json.loads(line) for line in (a.data / "states.jsonl").read_text().splitlines()]
    split = np.array([r["split"] for r in rows])
    reference = np.array([r["baseline_index"] for r in rows])
    with np.load(a.data / "features.npz", allow_pickle=False) as z:
        x = z["x"]
        pop = z["population_x"]
        targets = z["targets"]
        groups = z["population_ids"][z["neuron_indices"]]
    manifest = json.loads((a.data / "manifest.json").read_text())
    names = manifest["population_names"]
    mbon = np.flatnonzero(groups == names.index("MBON"))
    with torch.inference_mode():
        for i in range(0, len(x), 16):
            model(
                torch.tensor(x[i : i + 16], device=device),
                torch.tensor(pop[i : i + 16], device=device),
            )
    hook.remove()
    latent = np.concatenate(captured)
    result = {
        "scope": "frozen representation diagnostic, development splits only; no retraining of reader or LLM",
        "bundle_sha256": file_hash(a.bundle / "manifest.json"),
    }
    for name, feature in [("latent", latent), ("mbon_features", x[:, mbon, :].reshape(len(x), -1))]:
        result[name] = ridge_probe(feature, targets, split)
        change = np.array([r["phase"] == 4 for r in rows])
        difference = targets[:, 2:] - targets[reference, 2:]
        result[name + "_change"] = ridge_probe(
            np.concatenate([feature, feature[reference], feature - feature[reference]], axis=1)[
                change
            ],
            difference[change],
            split[change],
        )
    result["wall_s"] = time.monotonic() - start
    write_json(a.out, result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
