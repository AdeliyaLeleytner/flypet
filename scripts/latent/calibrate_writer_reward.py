#!/usr/bin/env python3
"""Learn a continuous dopamine output head with modality-balanced supervision.

Sensory rates stay with the validation-selected ridge map. This head also reads
only cached contextual LM features; no keyword or text-label inference rule.
"""

import argparse, json, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
from flypet.neural_records import write_json, file_hash
from scripts.latent.train_writer import metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=150)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(919)
    rng = np.random.default_rng(919)
    start = time.monotonic()
    rows = [json.loads(x) for x in (a.data / "samples.jsonl").read_text().splitlines()]
    with np.load(a.fit / "embedding_cache.npz", allow_pickle=False) as z:
        x = z["x"]
        target = z["targets"]
    with np.load(a.fit / "preprocessing.npz", allow_pickle=False) as z:
        basis = z["basis"]
    with np.load(a.fit / "ridge.npz", allow_pickle=False) as z:
        weight = z["weight"]
        bias = z["bias"]
    splits = {
        s: np.array([i for i, r in enumerate(rows) if r["split"] == s])
        for s in ["train", "validation", "development_test", "transfer_test"]
    }
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.SiLU(), nn.Linear(128, 2))
    nn.init.constant_(model[-1].bias, -2)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    xx = torch.tensor(x)
    yy = torch.tensor(target[:, -2:] / 90)
    best = float("inf")
    history = []
    for epoch in range(a.epochs):
        order = rng.permutation(splits["train"])
        model.train()
        for k in range(0, len(order), 64):
            ids = order[k : k + 64]
            optim.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xx[ids]), yy[ids])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optim.step()
        model.eval()
        with torch.no_grad():
            pred = model(xx[splits["validation"]]).sigmoid().numpy() * 90
        error = float(abs(pred - target[splits["validation"], -2:]).mean())
        if error < best:
            best = error
            selected = epoch + 1
            save_file(
                {k: v.contiguous() for k, v in model.state_dict().items()},
                str(a.out / "dopamine_head.safetensors"),
            )
        if (epoch + 1) % 25 == 0:
            history.append({"epoch": epoch + 1, "validation_dopamine_mae_hz": error})
            print(json.dumps(history[-1]), flush=True)
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(a.out / "dopamine_head.safetensors")))
    model.eval()
    results = {}
    predictions = {}
    for name, ids in splits.items():
        pred = np.clip(x[ids] @ weight + bias, 0, 1) * 300
        with torch.no_grad():
            pred[:, -2:] = model(xx[ids]).sigmoid().numpy() * 90
        results[name] = {
            **metrics(pred, target[ids], basis),
            "dopamine_mae_hz": float(abs(pred[:, -2:] - target[ids, -2:]).mean()),
            "no_external_reinforcement_mae_hz": float(
                abs(pred[target[ids, -2:].sum(axis=1) == 0, -2:]).mean()
            ),
        }
        predictions[name] = {
            "ids": ids.tolist(),
            "rates_hz": pred.tolist(),
            "targets_hz": target[ids].tolist(),
        }
    write_json(
        a.out / "config.json",
        {
            "dim": x.shape[1],
            "hidden": 128,
            "outputs": 2,
            "max_rate_hz": 90,
            "selected_epoch": selected,
        },
    )
    write_json(
        a.out / "report.json",
        {
            "status": "complete",
            "scope": "exploratory reinforcement-head calibration on existing splits",
            "metrics": results,
            "wall_s": time.monotonic() - start,
        },
    )
    write_json(a.out / "predictions.json", predictions)
    write_json(a.out / "training.json", history)
    write_json(
        a.out / "provenance.json",
        {
            "fit_files": {
                n: file_hash(a.fit / n)
                for n in ["embedding_cache.npz", "preprocessing.npz", "ridge.npz"]
            },
            "samples_sha256": file_hash(a.data / "samples.jsonl"),
            "head_files": {
                n: file_hash(a.out / n)
                for n in ["dopamine_head.safetensors", "config.json", "report.json"]
            },
        },
    )
    print(json.dumps({"status": "complete", "metrics": results}), flush=True)


if __name__ == "__main__":
    main()
