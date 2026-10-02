#!/usr/bin/env python3
"""Pretrain a neural-state bottleneck; no LLM, text or inference-time readout rules."""

import argparse, json, time, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from safetensors.torch import save_file, load_file
from flypet.neural_codec import NeuralCodec
from flypet.neural_records import write_json, file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=300)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(936)
    rng = np.random.default_rng(936)
    manifest = json.loads((a.data / "manifest.json").read_text())
    rows = [json.loads(s) for s in (a.data / "states.jsonl").read_text().splitlines()]
    with np.load(a.data / "features.npz", allow_pickle=False) as z:
        groups = z["population_ids"][z["neuron_indices"]]
        positions = np.flatnonzero(groups == manifest["population_names"].index("MBON"))
        x = z["x"][:, positions]
        target = z["targets"]
    split = np.array([r["split"] for r in rows])
    refs = np.array([r["baseline_index"] for r in rows])
    train = np.flatnonzero(split == "train")
    val = np.flatnonzero(split == "validation")
    y = np.c_[np.log1p(target[:, :2]), target[:, 2]]
    ym = y[train].mean(0)
    ys = np.maximum(y[train].std(0), 0.1)
    y = (y - ym) / ys
    model = NeuralCodec(len(positions))
    model.mean.copy_(torch.tensor(x[train].mean(0)))
    model.std.copy_(torch.tensor(np.maximum(x[train].std(0), 0.1)))
    xx = torch.tensor(x)
    yy = torch.tensor(y)
    ref = torch.tensor(refs)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    probabilities = np.zeros(len(rows))
    for mask in [target[train, 2] < 0, target[train, 2] >= 0]:
        ids = train[mask]
        if len(ids):
            probabilities[ids] = 1 / len(ids)
    probabilities /= probabilities.sum()
    best = float("inf")
    history = []
    for epoch in range(a.epochs):
        model.train()
        order = rng.choice(len(x), len(train), p=probabilities)
        total = []
        for start_index in range(0, len(order), 64):
            ids = order[start_index : start_index + 64]
            joined = np.r_[refs[ids], ids]
            n = len(ids)
            optimizer.zero_grad(set_to_none=True)
            _, pred, reconstruction, normalized = model(xx[joined])
            loss = (
                torch.nn.functional.mse_loss(pred, yy[joined])
                + 0.5
                * torch.nn.functional.mse_loss(
                    pred[n:, 2] - pred[:n, 2], yy[ids, 2] - yy[refs[ids], 2]
                )
                + 0.05 * torch.nn.functional.mse_loss(reconstruction, normalized)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            _, pred, _, _ = model(xx)
            prediction = pred.numpy() * ys + ym
        errors = abs(prediction[val, 2] - target[val, 2])
        neg = target[val, 2] < 0
        current = float(np.mean([errors[m].mean() for m in [neg, ~neg] if m.any()]))
        changes = np.array([i for i in val if rows[i]["phase"] == 4])
        delta_true = target[changes, 2] - target[refs[changes], 2]
        delta_pred = prediction[changes, 2] - prediction[refs[changes], 2]
        delta = float(abs(delta_pred - delta_true).mean())
        criterion = current + 0.5 * delta
        if criterion < best:
            best = criterion
            selected = epoch + 1
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(a.out / "codec.safetensors"),
            )
        if (epoch + 1) % 10 == 0:
            row = {
                "epoch": epoch + 1,
                "train_loss": float(np.mean(total)),
                "validation_balanced_valence_mae": current,
                "validation_delta_mae": delta,
                "wall_s": time.monotonic() - start,
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    model.load_state_dict(load_file(str(a.out / "codec.safetensors")))
    model.eval()
    metrics = {}
    with torch.no_grad():
        tokens, pred, reconstruction, normalized = model(xx)
        prediction = pred.numpy() * ys + ym
    for name in ("train", "validation", "development_test"):
        ids = np.flatnonzero(split == name)
        changes = np.array([i for i in ids if rows[i]["phase"] == 4])
        err = abs(prediction[ids, 2] - target[ids, 2])
        metrics[name] = {
            "n": len(ids),
            "valence_mae": float(err.mean()),
            "delta_mae": float(
                abs(
                    (prediction[changes, 2] - prediction[refs[changes], 2])
                    - (target[changes, 2] - target[refs[changes], 2])
                ).mean()
            ),
            "normalized_neural_reconstruction_mse": float(
                (reconstruction[ids] - normalized[ids]).square().mean()
            ),
            "sign_accuracy": float((np.sign(prediction[ids, 2]) == np.sign(target[ids, 2])).mean()),
        }
    np.savez_compressed(a.out / "targets.npz", mean=ym, std=ys)
    np.savez_compressed(a.out / "latents.npz", tokens=tokens.numpy())
    write_json(
        a.out / "config.json",
        {
            "neurons": len(positions),
            "channels": 9,
            "tokens": 4,
            "width": 128,
            "mbon_positions": positions.tolist(),
            "selected_epoch": selected,
        },
    )
    write_json(a.out / "training.json", history)
    write_json(
        a.out / "report.json",
        {
            "status": "complete",
            "metrics": metrics,
            "wall_s": time.monotonic() - start,
            "selection": "validation balanced valence MAE + 0.5 delta MAE",
            "dataset_manifest_sha256": file_hash(a.data / "manifest.json"),
        },
    )
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
