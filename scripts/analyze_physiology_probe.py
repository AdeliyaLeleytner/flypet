"""Fixed CPU-only information probes on exported INITIAL states; no test-set access."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch
from torch import nn


def metrics(logits, y, last):
    prediction = logits.argmax(1)
    correct = prediction == y
    return {
        "accuracy": float(correct.float().mean()),
        "earlier_object_accuracy": float(correct[~last].float().mean()),
        "most_recent_object_accuracy": float(correct[last].float().mean()),
        "cross_entropy": float(nn.functional.cross_entropy(logits, y)),
    }


def main(a):
    torch.set_num_threads(4)
    torch.manual_seed(713)
    d = np.load(a.features)
    x = torch.from_numpy(d["train_x"])
    v = torch.from_numpy(d["validation_x"])
    mean = x.mean(0)
    std = x.std(0).clamp_min(0.01)
    x = ((x - mean) / std).clamp(-10, 10)
    v = ((v - mean) / std).clamp(-10, 10)
    y = torch.from_numpy(d["train_y"])
    vy = torch.from_numpy(d["validation_y"])
    q = torch.from_numpy(d["train_q"])
    vq = torch.from_numpy(d["validation_q"])
    last = torch.from_numpy(d["train_last"])
    vlast = torch.from_numpy(d["validation_last"])
    report = {
        "scope": "Train/validation diagnostic on initial physiological state, no test; three query-conditioned heads",
        "features_sha256": hashlib.sha256(Path(a.features).read_bytes()).hexdigest(),
        "n_train": len(x),
        "n_validation": len(v),
        "models": {},
    }
    train_logits = torch.zeros(len(x), 4)
    val_logits = torch.zeros(len(v), 4)
    for obj in range(3):
        xx = torch.cat([x[q == obj], torch.ones(int((q == obj).sum()), 1)], 1).double()
        target = nn.functional.one_hot(y[q == obj], 4).double()
        reg = torch.eye(xx.shape[1], dtype=torch.float64) * 10
        reg[-1, -1] = 0
        weights = torch.linalg.solve(xx.T @ xx + reg, xx.T @ target)
        train_logits[q == obj] = (xx @ weights).float()
        vv = torch.cat([v[vq == obj], torch.ones(int((vq == obj).sum()), 1)], 1).double()
        val_logits[vq == obj] = (vv @ weights).float()
    report["models"]["ridge_lambda10"] = {
        "train": metrics(train_logits, y, last),
        "validation": metrics(val_logits, vy, vlast),
    }
    model = nn.Sequential(nn.Linear(x.shape[1], 128), nn.SiLU(), nn.Linear(128, 12))
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    curve = []
    for step in range(1, 501):
        ids = torch.randint(len(x), (128,))
        logits = model(x[ids]).reshape(-1, 3, 4)[torch.arange(len(ids)), q[ids]]
        loss = nn.functional.cross_entropy(logits, y[ids])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 100 == 0:
            with torch.no_grad():
                vl = model(v).reshape(-1, 3, 4)[torch.arange(len(v)), vq]
                curve.append({"step": step, **metrics(vl, vy, vlast)})
    with torch.no_grad():
        tl = model(x).reshape(-1, 3, 4)[torch.arange(len(x)), q]
        vl = model(v).reshape(-1, 3, 4)[torch.arange(len(v)), vq]
    report["models"]["mlp128_500steps"] = {
        "train": metrics(tl, y, last),
        "validation": metrics(vl, vy, vlast),
        "curve": curve,
        "checkpoint_selection": "fixed500updates; no selection on test or validation",
    }
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
