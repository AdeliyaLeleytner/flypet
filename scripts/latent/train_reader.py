#!/usr/bin/env python3
"""Bounded frozen-Qwen reader training on downstream neural observations."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from flypet.latent_bridge import NeuralReader
from flypet.latent_reader_data import FIELDS, score
from flypet.neural_records import file_hash, write_json

MODEL = "Qwen/Qwen3-4B"
REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
QUESTION = (
    "Read the supplied neural state of a simulated fruit fly. Report one JSON object with exactly "
    'these numeric fields: "approach_hz", "avoid_hz", "valence". The first two are summed MBON '
    "population firing rates; valence is their normalized balance, between -1 and 1. "
    "Use one decimal place for rates and three for valence. Return JSON only."
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--tokens", type=int, default=16)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=913)
    p.add_argument("--max-seconds", type=int, default=5400)
    p.add_argument("--profile-only", action="store_true")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()

    def deadline():
        if time.monotonic() - start > a.max_seconds:
            raise TimeoutError("Reader exceeded its training/evaluation bound")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    manifest = json.loads((a.data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if file_hash(a.data / name) != digest:
            raise ValueError("Dataset hash mismatch: " + name)
    rows = [json.loads(x) for x in (a.data / "examples.jsonl").read_text().splitlines()]
    with np.load(a.data / "features.npz", allow_pickle=False) as z:
        x = z["x"]
        y = z["targets"]
        meta = z["metadata"]
        preprocessing = {
            key: z[key] for key in ("mean", "std", "metadata", "neuron_indices", "neuron_root_ids")
        }
    np.savez_compressed(a.out / "preprocessing.npz", **preprocessing)
    write_json(a.out / "dataset_manifest.json", manifest)
    splits = {
        key: np.array([i for i, r in enumerate(rows) if r["split"] == key])
        for key in ("train", "validation", "development_test")
    }
    if not all(len(v) for v in splits.values()):
        raise ValueError("Empty fitting/evaluation split")
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    load_start = time.monotonic()
    llm = (
        AutoModelForCausalLM.from_pretrained(
            a.model, revision=a.revision, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to("cuda")
        .eval()
    )
    llm.requires_grad_(False)
    embedding = llm.get_input_embeddings()
    dim = embedding.weight.shape[1]
    reader = NeuralReader(
        x.shape[1], x.shape[2], meta, dim, float(embedding.weight.std()), a.tokens, a.width
    ).cuda()

    def prompt(text):
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return list(tok(rendered, add_special_tokens=False)["input_ids"])

    prompt_ids = prompt(QUESTION)
    targets = [
        tok(r["caption"] + tok.eos_token, add_special_tokens=False)["input_ids"] for r in rows
    ]
    aux_y = np.c_[np.log1p(y[:, :2]), y[:, 2]]
    aux_mean = aux_y[splits["train"]].mean(axis=0)
    aux_std = np.maximum(aux_y[splits["train"]].std(axis=0), 0.05)
    aux_y = (aux_y - aux_mean) / aux_std

    def batch_loss(indices, auxiliary=True):
        n = len(indices)
        length = len(prompt_ids) + max(len(targets[i]) for i in indices)
        ids = torch.full((n, length), tok.pad_token_id, dtype=torch.long, device="cuda")
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for j, i in enumerate(indices):
            row = prompt_ids + targets[i]
            ids[j, : len(row)] = torch.tensor(row, device="cuda")
            mask[j, : len(row)] = 1
            labels[j, len(prompt_ids) : len(row)] = torch.tensor(targets[i], device="cuda")
        soft, aux = reader(torch.tensor(x[indices], device="cuda"))
        vectors = torch.cat((soft.to(embedding.weight.dtype), embedding(ids)), dim=1)
        mask = torch.cat((torch.ones((n, a.tokens), device="cuda", dtype=mask.dtype), mask), dim=1)
        labels = torch.cat(
            (torch.full((n, a.tokens), -100, device="cuda", dtype=labels.dtype), labels), dim=1
        )
        loss = llm(inputs_embeds=vectors, attention_mask=mask, labels=labels, use_cache=False).loss
        aux_loss = torch.nn.functional.mse_loss(aux, torch.tensor(aux_y[indices], device="cuda"))
        return (
            loss + (0.05 * aux_loss if auxiliary else 0),
            loss.detach(),
            aux_loss.detach(),
            int((labels != -100).sum()),
        )

    @torch.no_grad()
    def nll(indices):
        reader.eval()
        total = weight = 0
        for k in range(0, len(indices), a.batch):
            deadline()
            _, loss, _, count = batch_loss(indices[k : k + a.batch], False)
            total += float(loss) * count
            weight += count
        return total / weight

    @torch.no_grad()
    def generate(indices, mode="neural", donor=None, question=QUESTION):
        reader.eval()
        answers = []
        for k in range(0, len(indices), a.batch):
            deadline()
            group = indices[k : k + a.batch]
            if mode == "numeric":
                prompts = [
                    prompt(question + "\nSupplied measurements: " + rows[i]["caption"])
                    for i in group
                ]
                width = max(map(len, prompts))
                ids = torch.tensor(
                    [[tok.pad_token_id] * (width - len(z)) + z for z in prompts], device="cuda"
                )
                mask = torch.tensor(
                    [[0] * (width - len(z)) + [1] * len(z) for z in prompts], device="cuda"
                )
                vectors = embedding(ids)
            else:
                ids = torch.tensor([prompt(question)] * len(group), device="cuda")
                vectors = embedding(ids)
                if mode != "no_state":
                    actual = donor[k : k + len(group)] if donor is not None else group
                    xx = np.zeros_like(x[group]) if mode == "mean_state" else x[actual]
                    soft, _ = reader(torch.tensor(xx, device="cuda"))
                    vectors = torch.cat((soft.to(vectors.dtype), vectors), dim=1)
                mask = torch.ones(vectors.shape[:2], device="cuda", dtype=torch.long)
            reply = llm.generate(
                inputs_embeds=vectors,
                attention_mask=mask,
                max_new_tokens=80,
                do_sample=False,
                use_cache=True,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            answers.extend(tok.batch_decode(reply, skip_special_tokens=True))
        return answers

    report = {
        "status": "running",
        "model": a.model,
        "revision": a.revision,
        "dataset_manifest_sha256": file_hash(a.data / "manifest.json"),
        "counts": {k: len(v) for k, v in splits.items()},
        "scope": manifest["scope"],
        "load_seconds": time.monotonic() - load_start,
        "trainable_parameters": sum(p.numel() for p in reader.parameters()),
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
        "auxiliary_head": "training only; normalized log approach/avoid and valence, weight 0.05",
        "history_or_measured_values_in_neural_prompt": False,
    }
    positive = splits["validation"][:24]
    t = time.monotonic()
    texts = generate(positive, "numeric")
    report["positive_control"] = {
        **score(texts, y[positive]),
        "wall_s": time.monotonic() - t,
        "predictions": [{"id": rows[i]["id"], "text": s} for i, s in zip(positive, texts)],
    }
    write_json(a.out / "report.json", report)
    torch.cuda.reset_peak_memory_stats()
    reader.train()
    t = time.monotonic()
    loss, _, _, _ = batch_loss(splits["train"][: a.batch])
    loss.backward()
    torch.cuda.synchronize()
    report["profile"] = {
        "forward_backward_seconds": time.monotonic() - t,
        "allocated_peak_bytes": torch.cuda.max_memory_allocated(),
        "gradient_norm": float(torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)),
        "batch": a.batch,
    }
    reader.zero_grad(set_to_none=True)
    write_json(a.out / "report.json", report)
    print(
        json.dumps(
            {
                "positive_control": report["positive_control"]["mae_valid"],
                "profile": report["profile"],
            }
        ),
        flush=True,
    )
    if a.profile_only:
        report["status"] = "profile_complete"
        write_json(a.out / "report.json", report)
        return
    if report["positive_control"]["valid_fraction"] < 0.9:
        raise RuntimeError(
            "The supplied-value language control failed; inspect before fitting the bridge"
        )
    initial = {k: v.detach().cpu().clone() for k, v in reader.state_dict().items()}
    torch.save(initial, a.out / "initial_reader.pt")
    optim = torch.optim.AdamW(reader.parameters(), lr=a.lr, weight_decay=0.01)
    best = float("inf")
    best_epoch = None
    log = []
    rng = np.random.default_rng(a.seed)
    for epoch in range(a.epochs):
        reader.train()
        losses = []
        order = rng.permutation(splits["train"])
        t = time.monotonic()
        for k in range(0, len(order), a.batch):
            deadline()
            indices = order[k : k + a.batch]
            optim.zero_grad(set_to_none=True)
            loss, numerical, aux, _ = batch_loss(indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reader.parameters(), 1.0)
            optim.step()
            losses.append(float(numerical))
        val = nll(splits["validation"])
        record = {
            "epoch": epoch + 1,
            "train_nll": float(np.mean(losses)),
            "validation_nll": val,
            "wall_s": time.monotonic() - t,
        }
        log.append(record)
        if val < best:
            best, best_epoch = val, epoch + 1
            torch.save(
                {
                    "state_dict": reader.state_dict(),
                    "epoch": best_epoch,
                    "validation_nll": best,
                    "config": report["config"],
                    "embedding_dim": dim,
                    "embedding_std": reader.embedding_std,
                    "metadata": meta,
                    "n_neurons": x.shape[1],
                    "channels": x.shape[2],
                    "dataset_manifest_sha256": report["dataset_manifest_sha256"],
                },
                a.out / "reader.pt",
            )
        write_json(a.out / "training.json", log)
        print(json.dumps(record), flush=True)
    checkpoint = torch.load(a.out / "reader.pt", map_location="cuda", weights_only=False)
    reader.load_state_dict(checkpoint["state_dict"])
    report["selected_epoch"] = best_epoch
    report["validation_nll"] = best
    predictions = {}
    metrics = {}
    for split in ("validation", "development_test"):
        selection = splits[split]
        donors = np.roll(selection, 1)
        for mode in ("neural", "donor_state", "mean_state", "no_state", "numeric"):
            t = time.monotonic()
            texts = generate(selection, mode, donor=donors if mode == "donor_state" else None)
            key = split + "/" + mode
            predictions[key] = [
                {
                    "id": rows[i]["id"],
                    "text": text,
                    "target": y[i].tolist(),
                    "donor_id": rows[donors[j]]["id"] if mode == "donor_state" else None,
                }
                for j, (i, text) in enumerate(zip(selection, texts))
            ]
            metrics[key] = {**score(texts, y[selection]), "wall_s": time.monotonic() - t}
    report.update(status="complete", metrics=metrics, total_seconds=time.monotonic() - start)
    write_json(a.out / "predictions.json", predictions)
    write_json(a.out / "report.json", report)
    print(
        json.dumps({"status": "complete", "selected_epoch": best_epoch, "metrics": metrics}),
        flush=True,
    )


if __name__ == "__main__":
    main()
