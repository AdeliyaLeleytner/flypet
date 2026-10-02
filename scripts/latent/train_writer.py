#!/usr/bin/env python3
"""Frozen-LLM hidden states -> learned continuous sensory rates."""

import argparse, json, random, time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from flypet.neural_records import file_hash, write_json
from flypet.latent_writer import hidden_features

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class Writer(nn.Module):
    def __init__(self, dim, channels, hidden=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, channels))
        nn.init.constant_(self.net[-1].bias, 0.05)

    def forward(self, x):
        return self.net(x)


def metrics(pred, target, basis):
    predicted = pred @ basis.T
    truth = target @ basis.T
    norms = np.linalg.norm(truth, axis=1)
    active = norms > 1e-8
    quiet = ~active
    cosine = (predicted[active] * truth[active]).sum(axis=1) / (
        np.linalg.norm(predicted[active], axis=1) * norms[active] + 1e-9
    )
    return {
        "channel_mae_hz": float(np.abs(pred - target).mean()),
        "port_mae_hz": float(np.abs(predicted - truth).mean()),
        "relative_port_l2_active": float(
            (np.linalg.norm(predicted[active] - truth[active], axis=1) / norms[active]).mean()
        ),
        "cosine_active": float(cosine.mean()),
        "quiet_mean_port_hz": float(np.abs(predicted[quiet]).mean()) if quiet.any() else None,
        "n": len(target),
        "active": int(active.sum()),
        "quiet": int(quiet.sum()),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=914)
    p.add_argument("--max-seconds", type=int, default=3600)
    p.add_argument("--pooling", choices=["last", "content_mean"], default="content_mean")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    manifest = json.loads((a.data / "manifest.json").read_text())
    for name, h in manifest["files"].items():
        if file_hash(a.data / name) != h:
            raise ValueError("Dataset hash mismatch")
    rows = [json.loads(x) for x in (a.data / "samples.jsonl").read_text().splitlines()]
    with np.load(a.data / "targets.npz", allow_pickle=False) as z:
        target = z["rates_hz"]
        basis = z["basis"]
        roots = z["input_root_ids"]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    middle = model.config.num_hidden_layers // 2
    features = []
    for offset in range(0, len(rows), 32):
        if time.monotonic() - start > a.max_seconds:
            raise TimeoutError("Writer deadline")
        state = hidden_features(
            model, tok, [r["text"] for r in rows[offset : offset + 32]], pooling=a.pooling
        )
        features.append(state.float().cpu().numpy())
        if offset % 256 == 0:
            print(
                json.dumps({"embedded": min(offset + 32, len(rows)), "total": len(rows)}),
                flush=True,
            )
    x = np.concatenate(features)
    del model, state
    torch.cuda.empty_cache()
    splits = {
        name: np.array([i for i, r in enumerate(rows) if r["split"] == name])
        for name in ("train", "validation", "development_test", "transfer_test")
        if any(r["split"] == name for r in rows)
    }
    mu = x[splits["train"]].mean(axis=0)
    sd = np.maximum(x[splits["train"]].std(axis=0), 0.01)
    x = np.clip((x - mu) / sd, -10, 10).astype(np.float32)
    np.savez_compressed(a.out / "embedding_cache.npz", x=x, targets=target)
    np.savez_compressed(
        a.out / "preprocessing.npz", mean=mu, std=sd, basis=basis, input_root_ids=roots
    )
    train, val = splits["train"], splits["validation"]
    mean = (target[train] / 300).mean(axis=0)
    kernel = x[train] @ x[train].T
    best = None
    grid = []
    for alpha in (1, 10, 100, 1000):
        dual = np.linalg.solve(kernel + alpha * np.eye(len(train)), target[train] / 300 - mean)
        weight = (x[train].T @ dual).astype(np.float32)
        pred = np.clip(x[val] @ weight + mean, 0, 1) * 300
        score = metrics(pred, target[val], basis)
        grid.append({"alpha": alpha, **score})
        objective = score["port_mae_hz"]
        if best is None or objective < best[0]:
            best = (objective, alpha, weight)
    np.savez_compressed(a.out / "ridge.npz", weight=best[2], bias=mean)
    report = {
        "status": "training",
        "model": MODEL,
        "revision": REVISION,
        "middle_layer": middle,
        "feature": "concatenated middle and final contextual hidden states",
        "pooling": a.pooling,
        "dataset_manifest_sha256": file_hash(a.data / "manifest.json"),
        "scope": manifest["scope"],
        "channels": manifest["channels"],
        "ridge_grid": grid,
        "ridge_alpha": best[1],
        "counts": {k: len(v) for k, v in splits.items()},
    }
    writer = Writer(x.shape[1], target.shape[1]).cuda()
    opt = torch.optim.AdamW(writer.parameters(), lr=0.001, weight_decay=0.001)
    xx = torch.tensor(x, device="cuda")
    yy = torch.tensor(target / 300, device="cuda")
    rng = np.random.default_rng(a.seed)
    best_mse = float("inf")
    best_epoch = 0
    log = []
    for epoch in range(a.epochs):
        if time.monotonic() - start > a.max_seconds:
            raise TimeoutError("Writer deadline")
        writer.train()
        order = rng.permutation(train)
        for k in range(0, len(order), 64):
            ids = order[k : k + 64]
            opt.zero_grad(set_to_none=True)
            pred = writer(xx[ids])
            loss = (pred - yy[ids]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(writer.parameters(), 1)
            opt.step()
        writer.eval()
        with torch.no_grad():
            pred = writer(xx[val]).clamp(0, 1)
            mse = float((pred - yy[val]).square().mean())
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch + 1
            torch.save(
                {
                    "state_dict": writer.state_dict(),
                    "dim": x.shape[1],
                    "channels": target.shape[1],
                    "hidden": 256,
                    "epoch": best_epoch,
                },
                a.out / "writer.pt",
            )
        if (epoch + 1) % 20 == 0:
            row = {"epoch": epoch + 1, "validation_mse": mse}
            log.append(row)
            print(json.dumps(row), flush=True)
    checkpoint = torch.load(a.out / "writer.pt", map_location="cuda", weights_only=True)
    writer.load_state_dict(checkpoint["state_dict"])
    writer.eval()
    report["metrics"] = {}
    predictions = {}
    for name, ids in splits.items():
        with torch.no_grad():
            pred = (writer(xx[ids]).clamp(0, 1) * 300).cpu().numpy()
        ridge = np.clip(x[ids] @ best[2] + mean, 0, 1) * 300
        report["metrics"][name] = {
            "mlp": metrics(pred, target[ids], basis),
            "ridge": metrics(ridge, target[ids], basis),
            "zero": metrics(np.zeros_like(pred), target[ids], basis),
        }
        predictions[name] = {
            "ids": ids.tolist(),
            "mlp_rates_hz": pred.tolist(),
            "ridge_rates_hz": ridge.tolist(),
            "targets_hz": target[ids].tolist(),
        }
    report.update(status="complete", selected_epoch=best_epoch, wall_s=time.monotonic() - start)
    write_json(a.out / "report.json", report)
    write_json(a.out / "predictions.json", predictions)
    write_json(a.out / "training.json", log)
    print(json.dumps({"status": "complete", "metrics": report["metrics"]}), flush=True)


if __name__ == "__main__":
    main()
