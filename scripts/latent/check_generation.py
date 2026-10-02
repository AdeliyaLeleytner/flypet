#!/usr/bin/env python3
"""Diagnose supplied-value generation separately from neural reader fitting."""

import argparse, json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from flypet.latent_reader_data import parse_reply
from scripts.latent.train_reader import MODEL, REVISION, QUESTION


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = [json.loads(x) for x in (a.data / "examples.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["split"] == "validation"][:8]
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .cuda()
        .eval()
    )
    torch.set_num_threads(4)
    results = {}
    for variant in ("original", "explicit_copy"):
        prompts = []
        for row in rows:
            q = (
                QUESTION
                if variant == "original"
                else "Copy the supplied JSON exactly. Preserve every number and its sign. Do not recompute, negate, or reinterpret any value. Return JSON only."
            )
            text = q + "\nSupplied measurements: " + row["caption"]
            prompts.append(
                tok.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            )
        batch = tok(prompts, add_special_tokens=False, padding=True, return_tensors="pt").to("cuda")
        for method in ("input_ids", "inputs_embeds"):
            kw = (
                dict(batch)
                if method == "input_ids"
                else {
                    "inputs_embeds": model.get_input_embeddings()(batch["input_ids"]),
                    "attention_mask": batch["attention_mask"],
                }
            )
            with torch.inference_mode():
                output = model.generate(
                    **kw,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                )
            if method == "input_ids":
                output = output[:, batch["input_ids"].shape[1] :]
            texts = tok.batch_decode(output, skip_special_tokens=True)
            results[variant + "/" + method] = [
                {"id": r["id"], "target": r["target"], "text": t, "parsed": parse_reply(t)}
                for r, t in zip(rows, texts)
            ]
    a.out.write_text(
        json.dumps(
            {
                "scope": "posthoc prompt and inputs_embeds diagnostic; no checkpoint changes",
                "results": results,
            },
            indent=2,
        )
        + "\n"
    )
    print("diagnostic complete", flush=True)


if __name__ == "__main__":
    main()
