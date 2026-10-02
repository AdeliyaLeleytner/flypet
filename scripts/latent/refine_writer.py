#!/usr/bin/env python3
"""Fix the writer's dead-clamp optimization using its saved frozen-LM features.

Train an unconstrained residual around ridge; enforce physical bounds only at
inference. This reuses the cached representations and requires no GPU lease.
"""

import argparse, json, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from flypet.neural_records import write_json
from scripts.latent.train_writer import metrics


class ResidualWriter(nn.Module):
    def __init__(self, weight, bias, hidden=256):
        super().__init__()
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float32))
        self.net = nn.Sequential(
            nn.Linear(weight.shape[0], hidden), nn.SiLU(), nn.Linear(hidden, weight.shape[1])
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x @ self.weight + self.bias + self.net(x)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=250)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(915)
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
        name: np.array([i for i, r in enumerate(rows) if r["split"] == name])
        for name in ("train", "validation", "development_test")
    }
    model = ResidualWriter(weight, bias)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.01)
    xx = torch.tensor(x)
    yy = torch.tensor(target / 300)
    rng = np.random.default_rng(915)
    best = float("inf")
    history = []
    for epoch in range(a.epochs + 1):
        if epoch:
            model.train()
            order = rng.permutation(splits["train"])
            for k in range(0, len(order), 64):
                ids = order[k : k + 64]
                opt.zero_grad(set_to_none=True)
                pred = model(xx[ids])
                loss = (pred - yy[ids]).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(xx[splits["validation"]]).clamp(0, 1).numpy() * 300
        result = metrics(pred, target[splits["validation"]], basis)
        if result["port_mae_hz"] < best:
            best = result["port_mae_hz"]
            selected = epoch
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hidden": 256,
                    "dim": weight.shape[0],
                    "channels": weight.shape[1],
                    "epoch": epoch,
                },
                a.out / "writer.pt",
            )
        if epoch % 25 == 0:
            history.append({"epoch": epoch, **result})
            print(json.dumps(history[-1]), flush=True)
    saved = torch.load(a.out / "writer.pt", weights_only=True)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    results = {}
    predictions = {}
    for name, ids in splits.items():
        with torch.no_grad():
            pred = model(xx[ids]).clamp(0, 1).numpy() * 300
        results[name] = metrics(pred, target[ids], basis)
        predictions[name] = {
            "ids": ids.tolist(),
            "rates_hz": pred.tolist(),
            "targets_hz": target[ids].tolist(),
        }
    write_json(
        a.out / "report.json",
        {
            "status": "complete",
            "scope": "exploratory repair on the same wording split",
            "selected_epoch": selected,
            "metrics": results,
            "wall_s": time.monotonic() - start,
            "inference": "clip only after unconstrained residual prediction",
        },
    )
    write_json(a.out / "predictions.json", predictions)
    write_json(a.out / "training.json", history)
    print(
        json.dumps({"status": "complete", "selected_epoch": selected, "metrics": results}),
        flush=True,
    )


if __name__ == "__main__":
    main()
