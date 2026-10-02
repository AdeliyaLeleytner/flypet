#!/usr/bin/env python3
"""Read saved experimental memory-reader checkpoints in a fresh process.

Applies the checkpoint's saved feature selection, normalization, prompt and model
revision to named dataset examples. Does not fit preprocessing, change the pet's
runtime reader, load mutable pet memory, or run a simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.memory_reader_data import (
    digest,
    encode_history,
    load_dataset,
    load_profiles,
    parse_output,
    target_values,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--example-id", nargs="+", required=True)
    ap.add_argument("--device")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.out and a.out.exists():
        ap.error("Output path already exists")
    directory = a.checkpoint.resolve().parent
    manifest_path, preprocessing_path = directory / "manifest.json", directory / "preprocessing.npz"
    manifest = json.loads(manifest_path.read_text())
    rows, rates, neuron_indices, sources = load_dataset(a.data)
    for filename, expected in manifest["sources"].items():
        path = a.data / filename
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Dataset source differs from saved training manifest: {filename}")
    by_id = {row["example_id"]: i for i, row in enumerate(rows)}
    try:
        selected = [by_id[name] for name in a.example_id]
    except KeyError as exc:
        raise ValueError(f"Unknown example ID: {exc.args[0]}") from exc
    selected_rows = [rows[i] for i in selected]

    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if digest(manifest_path) != checkpoint["manifest_sha256"]:
        raise ValueError("Checkpoint and training manifest disagree")
    if (
        checkpoint.get("preprocessing_sha256")
        and digest(preprocessing_path) != checkpoint["preprocessing_sha256"]
    ):
        raise ValueError("Saved preprocessing differs from checkpoint hash")
    mode = checkpoint["mode"]
    with np.load(preprocessing_path, allow_pickle=False) as preprocessing:
        if mode == "neural":
            positions = {int(index): i for i, index in enumerate(neuron_indices)}
            columns = [positions[int(index)] for index in preprocessing["neuron_indices"]]
            matrix = (
                np.log1p(rates[selected][:, columns]) - preprocessing["neural_mu"]
            ) / preprocessing["neural_sd"]
        elif mode == "null":
            matrix = np.zeros((len(selected), checkpoint["input_dim"]), dtype=np.float32)
        elif mode == "history":
            profile_name = (
                "input_profiles.json"
                if "input_profiles.json" in manifest["sources"]
                else "chemical_profiles.json"
            )
            profiles, _ = load_profiles(a.data / profile_name)
            config = manifest["preprocessing"]
            raw = encode_history(
                selected_rows, config["history_vocabulary"], config["max_history_events"], profiles
            )
            matrix = (raw - preprocessing["history_mu"]) / preprocessing["history_sd"]
        else:
            raise ValueError(f"Unrecognized checkpoint mode: {mode}")
    del rates
    if matrix.shape != (len(selected), checkpoint["input_dim"]) or not np.isfinite(matrix).all():
        raise ValueError("Saved feature schema produced invalid query input")

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
    dtype_name = checkpoint.get("dtype", manifest["config"]["dtype"])
    dtype = getattr(torch, dtype_name)
    kwargs = {"revision": checkpoint["model_revision"]} if checkpoint.get("model_revision") else {}
    tokenizer = AutoTokenizer.from_pretrained(checkpoint["model"], **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = (
        AutoModelForCausalLM.from_pretrained(checkpoint["model"], dtype=dtype, **kwargs)
        .to(dev)
        .eval()
    )
    llm.requires_grad_(False)
    embedding = llm.get_input_embeddings()
    dim, tokens = embedding.weight.shape[1], checkpoint["tokens"]
    embedding_std = checkpoint.get("embedding_std", float(embedding.weight.std().detach()))

    class Projector(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(checkpoint["input_dim"], checkpoint["hidden"]),
                nn.GELU(),
                nn.Linear(checkpoint["hidden"], tokens * dim),
            )
            self.norm = nn.LayerNorm(dim)

        def forward(self, x):
            return (self.norm(self.net(x).reshape(len(x), tokens, dim)) * embedding_std).to(dtype)

    projector = Projector().to(dev).eval()
    projector.load_state_dict(checkpoint["projector"], strict=True)
    text_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": checkpoint["question"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompt = tokenizer(text_prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt] * len(selected), dtype=torch.long, device=dev)
    with torch.no_grad():
        soft = projector(torch.as_tensor(matrix, dtype=torch.float32, device=dev))
        embeddings = torch.cat((soft, embedding(input_ids)), dim=1)
        attention = torch.ones(embeddings.shape[:2], dtype=torch.long, device=dev)
        output_ids = llm.generate(
            inputs_embeds=embeddings,
            attention_mask=attention,
            do_sample=False,
            max_new_tokens=manifest["config"]["max_new_tokens"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    saved_path = directory / "predictions.json"
    saved_predictions = json.loads(saved_path.read_text()) if saved_path.exists() else []
    previous = {r["example_id"]: r["generated"] for r in saved_predictions if r["mode"] == mode}
    report = {
        "checkpoint": str(a.checkpoint.resolve()),
        "checkpoint_sha256": digest(a.checkpoint),
        "manifest_sha256": digest(manifest_path),
        "preprocessing_sha256": digest(preprocessing_path),
        "model": checkpoint["model"],
        "model_revision": checkpoint.get("model_revision"),
        "mode": mode,
        "device": str(dev),
        "dtype": dtype_name,
        "fitted_on_query": False,
        "uses_saved_prompt": True,
        "dataset_hashes_verified": True,
        "query_script_sha256": digest(Path(__file__)),
        "examples": [],
    }
    for row, text in zip(selected_rows, generated):
        report["examples"].append(
            {
                "example_id": row["example_id"],
                "split": row["split"],
                "generated": text,
                "parsed": parse_output(text),
                "truth": target_values(row),
                "matches_saved_generation": (
                    text == previous[row["example_id"]] if row["example_id"] in previous else None
                ),
            }
        )
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
