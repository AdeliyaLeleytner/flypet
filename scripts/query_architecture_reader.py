#!/usr/bin/env python3
"""Standalone inference from a saved experimental adapter, with hash checks."""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.memory_reader_data import digest, load_dataset, parse_output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--graph", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--example-id", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from flypet.graph_reader import make_projector

    ck_path = Path(a.checkpoint)
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    base = ck_path.parent
    if (
        digest(base / "manifest.json") != ck["manifest_sha256"]
        or digest(base / "preprocessing.npz") != ck["preprocessing_sha256"]
    ):
        raise ValueError("Checkpoint preprocessing/manifest mismatch")
    graph_path = Path(a.graph) / "graph.npz"
    if digest(graph_path) != ck["graph_sha256"]:
        raise ValueError("Graph checksum mismatch")
    with np.load(graph_path, allow_pickle=False) as z:
        graph = {k: z[k].copy() for k in z.files}
    with np.load(base / "preprocessing.npz", allow_pickle=False) as z:
        features, mu, sd = z["neuron_indices"], z["neural_mu"], z["neural_sd"]
    if not np.array_equal(features, graph["neuron_indices"]):
        raise ValueError("Neuron order mismatch")
    rows, rates, indices, hashes = load_dataset(a.data)
    manifest = json.loads((base / "manifest.json").read_text())
    for name in ("flypet/graph_reader.py", "flypet/memory_reader_data.py"):
        if digest(ROOT / name) != manifest["code_sha256"].get(name):
            raise ValueError(f"Inference code differs from training snapshot: {name}")
    if any(manifest["sources"].get(k) != value for k, value in hashes.items()):
        raise ValueError("Inference dataset differs from the recorded training snapshot")
    positions = {int(v): i for i, v in enumerate(indices)}
    i = next(i for i, r in enumerate(rows) if r["example_id"] == a.example_id)
    x = (np.log1p(rates[i, [positions[int(v)] for v in features]]) - mu) / sd
    dev = torch.device(a.device)
    dtype = getattr(torch, ck["dtype"]) if dev.type == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(ck["model"], revision=ck["model_revision"])
    llm = (
        AutoModelForCausalLM.from_pretrained(
            ck["model"], revision=ck["model_revision"], dtype=dtype
        )
        .to(dev)
        .eval()
    )
    llm.requires_grad_(False)
    model = (
        make_projector(
            ck["mode"],
            len(features),
            ck["tokens"],
            llm.get_input_embeddings().weight.shape[1],
            graph,
            hidden=ck["hidden"],
            width=ck["width"],
        )
        .to(dev)
        .eval()
    )
    model.load_state_dict({k.removeprefix("reader."): v for k, v in ck["projector"].items()})
    prompt = tok.apply_chat_template(
        [{"role": "user", "content": ck["question"]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = tok(prompt, add_special_tokens=False, return_tensors="pt").input_ids.to(dev)
    started = time.monotonic()
    with torch.no_grad():
        prefix = (model(torch.tensor(x[None], device=dev)) * ck["embedding_std"]).to(dtype)
        embeddings = torch.cat((prefix, llm.get_input_embeddings()(ids)), dim=1)
        output = llm.generate(
            inputs_embeds=embeddings,
            attention_mask=torch.ones(embeddings.shape[:2], dtype=torch.long, device=dev),
            do_sample=False,
            max_new_tokens=80,
            pad_token_id=tok.pad_token_id,
            eos_token_id=tok.eos_token_id,
        )
    text = tok.decode(output[0], skip_special_tokens=True)
    result = {
        "example_id": a.example_id,
        "mode": ck["mode"],
        "device": str(dev),
        "dtype": str(dtype),
        "checkpoint_sha256": digest(ck_path),
        "generated": text,
        "parsed": parse_output(text),
        "truth": rows[i]["target"],
        "seconds": time.monotonic() - started,
        "scope": "fresh-process checkpoint loading/inference; not an additional independent test",
    }
    with Path(a.out).open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
