#!/usr/bin/env python3
"""Train frozen-Qwen forecasts of a different probe's memory-induced valence change.

Inputs are fixed diagnostic MBON responses plus the requested odor profile,
complete input history plus profiles, or a same-capacity constant adapter.
Only validation NLL selects checkpoints; the supplied-value control is an oracle.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.memory_forecast_data import (
    digest,
    load_dataset,
    prepare_dataset,
    score_outputs,
    swap_pairs,
    target_values,
)

QUESTION = (
    "Forecast the requested fly simulation probe from the supplied memory context "
    'as one JSON object with exactly the numeric field "delta_valence". '
    "This is the requested-probe MBON valence minus its matched naive-state valence, "
    "in [-2,2], not a body action. The requested response is not supplied. "
    "Use three decimal places. Output JSON only."
)


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--revision", help="Optional pinned Hugging Face revision")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--generation-batch", type=int, default=24)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    ap.add_argument("--device")
    ap.add_argument(
        "--modes",
        nargs="+",
        choices=("neural", "history", "null"),
        default=["neural", "history", "null"],
    )
    ap.add_argument(
        "--generation-per-split",
        type=int,
        default=0,
        help="0 evaluates every held-out row; positive cap sampled with fixed seed 2718",
    )
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--max-seconds", type=int, default=3600)
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Technical check: two training/validation rows and one matched test pair only",
    )
    ap.add_argument("--skip-numeric", action="store_true")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    a = ap.parse_args()
    if (
        min(
            a.epochs,
            a.batch,
            a.generation_batch,
            a.tokens,
            a.hidden,
            a.max_new_tokens,
            a.max_seconds,
        )
        < 1
    ):
        ap.error("Training sizes, token limits and deadline must be positive")
    if a.generation_per_split < 0 or a.lr <= 0:
        ap.error("Generation cap must be nonnegative and learning rate positive")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def check_deadline():
        if time.monotonic() - started > a.max_seconds:
            raise TimeoutError("Configured forecast training/evaluation deadline exceeded")

    rows, neural_raw, history_raw, data_manifest, source_hashes = load_dataset(a.data)
    prepared = prepare_dataset(rows, neural_raw, history_raw, data_manifest)
    manifest = prepared["manifest"]
    manifest["derived_dataset_sources"] = source_hashes
    manifest["code_sha256"] = {
        str(p.relative_to(ROOT)): digest(p)
        for p in (
            Path(__file__).resolve(),
            ROOT / "flypet/memory_forecast_data.py",
            ROOT / "flypet/memory_reader_data.py",
        )
    }
    manifest["config"] = vars(a)
    save_json(out / "manifest.json", manifest)
    np.savez_compressed(
        out / "preprocessing.npz",
        **{k: prepared[k] for k in ("neural_mu", "neural_sd", "history_mu", "history_sd")},
    )
    print(
        json.dumps({"counts": manifest["counts"], "preprocessing": manifest["preprocessing"]}),
        flush=True,
    )
    if a.prepare_only:
        return

    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = torch.device(
        a.device
        or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    dtype = getattr(torch, a.dtype)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model_kwargs = {"revision": a.revision} if a.revision else {}
    tok = AutoTokenizer.from_pretrained(a.model, **model_kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    llm = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, **model_kwargs).to(dev).eval()
    llm.requires_grad_(False)
    if a.gradient_checkpointing:
        llm.gradient_checkpointing_enable()
        llm.config.use_cache = False
    embedding = llm.get_input_embeddings()
    dim = embedding.weight.shape[1]
    emb_std = float(embedding.weight.std().detach())

    class Projector(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, a.hidden), nn.GELU(), nn.Linear(a.hidden, a.tokens * dim)
            )
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return (self.norm(self.net(x).reshape(len(x), a.tokens, dim)) * emb_std).to(dtype)

    def prompt_ids(text):
        text = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return tok(text, add_special_tokens=False)["input_ids"]

    fixed_prompt = prompt_ids(QUESTION)
    prompts = [fixed_prompt] * len(rows)
    numeric_prompts = [
        prompt_ids(
            QUESTION
            + "\nORACLE CONTROL: the exact target is supplied; copy this value: "
            + json.dumps(target_values(row), separators=(",", ":"))
        )
        for row in rows
    ]
    targets = [
        tok(text + tok.eos_token, add_special_tokens=False)["input_ids"]
        for text in prepared["captions"]
    ]

    def loss_for(proj, matrix, selection, prompt_set=prompts):
        selection = list(map(int, selection))
        length = max(len(prompt_set[i]) + len(targets[i]) for i in selection)
        ids = torch.full((len(selection), length), tok.pad_token_id, dtype=torch.long, device=dev)
        mask = torch.zeros_like(ids)
        labels = torch.full_like(ids, -100)
        for j, i in enumerate(selection):
            p, t = prompt_set[i], targets[i]
            ids[j, : len(p) + len(t)] = torch.tensor(p + t, device=dev)
            mask[j, : len(p) + len(t)] = 1
            labels[j, len(p) : len(p) + len(t)] = torch.tensor(t, device=dev)
        inputs = embedding(ids)
        if proj is not None:
            soft = proj(torch.as_tensor(matrix[selection], device=dev))
            inputs = torch.cat((soft, inputs), dim=1)
            mask = torch.cat(
                (torch.ones((len(selection), a.tokens), dtype=torch.long, device=dev), mask), dim=1
            )
            labels = torch.cat(
                (
                    torch.full((len(selection), a.tokens), -100, dtype=torch.long, device=dev),
                    labels,
                ),
                dim=1,
            )
        return llm(inputs_embeds=inputs, attention_mask=mask, labels=labels).loss

    @torch.no_grad()
    def nll(proj, matrix, selection, prompt_set=prompts):
        total = tokens = 0
        for start in range(0, len(selection), a.batch):
            check_deadline()
            batch = selection[start : start + a.batch]
            count = sum(len(targets[int(i)]) for i in batch)
            total += float(loss_for(proj, matrix, batch, prompt_set)) * count
            tokens += count
        return total / tokens if tokens else None

    @torch.no_grad()
    def generate(proj, matrix, selection, prompt_set=prompts):
        result = []
        for start in range(0, len(selection), a.generation_batch):
            check_deadline()
            batch = list(map(int, selection[start : start + a.generation_batch]))
            length = max(len(prompt_set[i]) for i in batch)
            ids = torch.full((len(batch), length), tok.pad_token_id, dtype=torch.long, device=dev)
            mask = torch.zeros_like(ids)
            for j, i in enumerate(batch):
                p = prompt_set[i]
                ids[j, -len(p) :] = torch.tensor(p, device=dev)
                mask[j, -len(p) :] = 1
            inputs = embedding(ids)
            if proj is not None:
                soft = proj(torch.as_tensor(matrix[batch], device=dev))
                inputs = torch.cat((soft, inputs), dim=1)
                mask = torch.cat(
                    (torch.ones((len(batch), a.tokens), dtype=torch.long, device=dev), mask), dim=1
                )
            generated = llm.generate(
                inputs_embeds=inputs,
                attention_mask=mask,
                do_sample=False,
                max_new_tokens=a.max_new_tokens,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
            result.extend(tok.batch_decode(generated, skip_special_tokens=True))
        return result

    train_rows, val_rows = prepared["splits"]["train"], prepared["splits"]["validation"]
    evaluation = {
        name: ids
        for name, ids in prepared["splits"].items()
        if name not in ("train", "validation", "excluded")
    }
    if "query_chemical_test" in evaluation:
        query_rows = evaluation.pop("query_chemical_test")
        for history_split in sorted({rows[i]["history_split"] for i in query_rows}):
            evaluation[f"query_chemical_test_from_{history_split}"] = np.asarray(
                [i for i in query_rows if rows[i]["history_split"] == history_split], dtype=np.int64
            )
    if a.smoke:
        train_rows, val_rows = train_rows[:2], val_rows[:2]
        smoke_pairs = swap_pairs(rows, evaluation["test"])
        if not smoke_pairs:
            raise ValueError("Smoke requires a matched test assignment pair")
        evaluation = {"test": np.asarray(smoke_pairs[0], dtype=np.int64)}
    generation = {
        name: (
            np.random.default_rng(2718).permutation(ids)[: a.generation_per_split]
            if a.generation_per_split
            else ids
        )
        for name, ids in evaluation.items()
    }
    report = {
        "config": vars(a),
        "device": str(dev),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "model_revision": getattr(llm.config, "_commit_hash", None),
        "versions": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "numpy": np.__version__,
        },
        "training": {},
        "evaluation": {},
        "generation_rows": {
            name: [rows[i]["example_id"] for i in ids] for name, ids in generation.items()
        },
        "numeric_control_semantics": "ORACLE supplied-target copying control; not a forecast or predictor",
        "primary_metrics": "delta_valence generation MAE with parse coverage, matched assignment contrasts, no-change baseline",
        "smoke_only": a.smoke,
        "target_response_excluded": True,
        "llm_trainable_parameters": sum(p.numel() for p in llm.parameters() if p.requires_grad),
        "no_change_baseline": {
            name: score_outputs(['{"delta_valence":0.000}'] * len(ids), [rows[int(i)] for i in ids])
            for name, ids in generation.items()
        },
    }
    predictions = []

    def persist():
        report["elapsed_seconds"] = time.monotonic() - started
        save_json(out / "report.json", report)
        save_json(out / "predictions.json", predictions)

    def record(mode, split, selection, texts, **extra):
        for i, text in zip(selection, texts):
            predictions.append(
                {
                    "mode": mode,
                    "split": split,
                    "example_id": rows[int(i)]["example_id"],
                    "split_group_id": rows[int(i)]["split_group_id"],
                    "history_id": rows[int(i)]["history_id"],
                    "diagnostic_example_ids": rows[int(i)]["diagnostic_example_ids"],
                    "truth": target_values(rows[int(i)]),
                    "generated": text,
                    **extra,
                }
            )

    trained = {}
    for mode in a.modes:
        check_deadline()
        # Each mode gets the same RNG start; the null has exactly the same input
        # dimension and initialized parameters as neural, but no state signal.
        torch.manual_seed(a.seed)
        rng = np.random.default_rng(a.seed)
        matrix = prepared[mode]
        proj = Projector(matrix.shape[1]).to(dev)
        opt = torch.optim.AdamW(proj.parameters(), lr=a.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, a.epochs * ((len(train_rows) + a.batch - 1) // a.batch)
        )
        history, best_nll, best_state, best_epoch = [], float("inf"), None, None
        gradient_norms = []
        mode_start = time.monotonic()
        for epoch in range(a.epochs):
            proj.train()
            shuffled = rng.permutation(train_rows)
            total = tokens = 0
            for start in range(0, len(shuffled), a.batch):
                check_deadline()
                batch = shuffled[start : start + a.batch]
                loss = loss_for(proj, matrix, batch)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Nonfinite training loss in {mode}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = float(nn.utils.clip_grad_norm_(proj.parameters(), 1))
                if not np.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise RuntimeError("Adapter gradient must be finite and nonzero")
                gradient_norms.append(gradient_norm)
                opt.step()
                scheduler.step()
                count = sum(len(targets[int(i)]) for i in batch)
                total += float(loss.detach()) * count
                tokens += count
            proj.eval()
            validation = nll(proj, matrix, val_rows)
            if validation is None or not np.isfinite(validation):
                raise RuntimeError("Nonfinite validation NLL")
            history.append(
                {
                    "epoch": epoch + 1,
                    "train_nll": total / tokens,
                    "validation_nll": validation,
                    "seconds": time.monotonic() - mode_start,
                }
            )
            if validation < best_nll:
                best_nll, best_epoch = validation, epoch + 1
                best_state = {
                    key: value.detach().cpu().clone() for key, value in proj.state_dict().items()
                }
            print(json.dumps({"mode": mode, **history[-1]}), flush=True)
        proj.load_state_dict(best_state)
        proj.eval()
        torch.save(
            {
                "projector": best_state,
                "mode": mode,
                "input_dim": matrix.shape[1],
                "hidden": a.hidden,
                "tokens": a.tokens,
                "model": a.model,
                "model_revision": report["model_revision"],
                "seed": a.seed,
                "embedding_std": emb_std,
                "dtype": a.dtype,
                "selection": "minimum validation NLL",
                "selected_epoch": best_epoch,
                "manifest_sha256": digest(out / "manifest.json"),
                "preprocessing_sha256": digest(out / "preprocessing.npz"),
                "question": QUESTION,
            },
            out / f"{mode}_projector.pt",
        )
        report["training"][mode] = {
            "history": history,
            "selected_epoch": best_epoch,
            "selection": "minimum validation NLL",
            "parameters": sum(p.numel() for p in proj.parameters()),
            "gradient_norm_min": min(gradient_norms),
            "gradient_norm_max": max(gradient_norms),
        }
        trained[mode] = proj
        persist()

    # No held-out target has influenced adapter fitting or selection above.
    for mode, proj in trained.items():
        report["evaluation"][mode] = {}
        for split, selection in evaluation.items():
            text = generate(proj, prepared[mode], generation[split])
            metric = score_outputs(text, [rows[int(i)] for i in generation[split]])
            metric["nll"] = nll(proj, prepared[mode], selection)
            report["evaluation"][mode][split] = metric
            record(mode, split, generation[split], text)
            print(json.dumps({"mode": mode, "split": split, **metric}), flush=True)
            persist()

    if "neural" in trained:
        report["state_swap"] = {}
        for split, selection in generation.items():
            # Build counterparts from all rows in this held-out partition even
            # when generation is capped. Recipient selection stays prespecified.
            selected_set = set(map(int, selection))
            pairs = [(r, d) for r, d in swap_pairs(rows, evaluation[split]) if r in selected_set]
            if not pairs:
                report["state_swap"][split] = {"n": 0}
                continue
            recipients, donors = np.asarray(pairs, dtype=np.int64).T
            # Query prompt is identical across rows; inputs come from matched
            # donor states. Compare against both donor and recipient truths.
            texts = generate(trained["neural"], prepared["neural"], donors)
            report["state_swap"][split] = {
                "pairing": "opposite reinforcement assignment; same conditioning pair/dose/history seed/query/probe seed",
                "donor_scored": score_outputs(texts, [rows[i] for i in donors]),
                "recipient_scored": score_outputs(texts, [rows[i] for i in recipients]),
                "mean_truth_delta_valence_gap": float(
                    np.mean(
                        [
                            abs(
                                rows[s]["target"]["delta_valence"]
                                - rows[t]["target"]["delta_valence"]
                            )
                            for t, s in pairs
                        ]
                    )
                ),
            }
            for recipient, donor, text in zip(recipients, donors, texts):
                predictions.append(
                    {
                        "mode": "neural_state_swap",
                        "split": split,
                        "example_id": rows[recipient]["example_id"],
                        "donor_example_id": rows[donor]["example_id"],
                        "truth": target_values(rows[donor]),
                        "recipient_truth": target_values(rows[recipient]),
                        "generated": text,
                    }
                )
            persist()

    if not a.skip_numeric:
        report["evaluation"]["oracle_numeric_text_positive_control"] = {}
        for split, selection in generation.items():
            texts = generate(None, None, selection, numeric_prompts)
            metric = score_outputs(texts, [rows[int(i)] for i in selection])
            metric["nll"] = nll(None, None, evaluation[split], numeric_prompts)
            report["evaluation"]["oracle_numeric_text_positive_control"][split] = metric
            record("oracle_numeric_text_positive_control", split, selection, texts)
            print(
                json.dumps(
                    {"mode": "oracle_numeric_text_positive_control", "split": split, **metric}
                ),
                flush=True,
            )
            persist()

    report["complete"] = True
    report["experiment_complete"] = not a.smoke
    report["artifact_hashes"] = {
        p.name: digest(p)
        for p in out.iterdir()
        if p.is_file() and p.name not in ("report.json", "report.json.tmp")
    }
    persist()
    print("COMPLETE", out, flush=True)


if __name__ == "__main__":
    main()
