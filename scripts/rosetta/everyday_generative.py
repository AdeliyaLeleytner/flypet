#!/usr/bin/env python3
"""Exploratory follow-up to the pre-registered everyday-smell test: ask the model instead of reading its embedding.

The same 25 words; Qwen3-32B (no thinking, greedy) picks the one block odorant most characteristic of the smell.
The fly then smells the picked odorant (its recorded block response). Metric as before: rank of the key odorant
among the 111 by correlation of Kenyon-cell (and lateral-horn) code deviations with the picked odorant's code.
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/rosetta"))
from everyday import PAIRS
from analyze_block import _Order, corr_rows
from flypet import rosetta as R
from transformers import AutoModelForCausalLM, AutoTokenizer

OLD, OUT = ROOT / "data/rosetta_20260925", ROOT / "data/rosetta_scaling_20260926"
MODEL = "/workspace/models/Qwen3-32B"


def main():
    P = np.load(OLD / "block_predictions.npz")
    block = [str(b) for b in P["block"]]
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to("cuda").eval()
    listing = "\n".join(block)
    picks = {}
    for word, key in PAIRS:
        msg = [
            {
                "role": "user",
                "content": f"Here is a list of odorant chemicals, one per line:\n{listing}\n\n"
                f"Which single chemical from this list is the most characteristic of the smell of {word}? "
                "Answer with the exact name from the list and nothing else.",
            }
        ]
        text = tok.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        ids = tok(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=24, do_sample=False)
        ans = (
            tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
            .strip()
            .splitlines()[0]
            .strip()
            .strip(".`'\"")
        )
        match = next((b for b in block if b.lower() == ans.lower()), None) or next(
            (b for b in block if b.lower() in ans.lower()), None
        )
        picks[word] = {"answer": ans, "picked": match, "key": key}
        print(f"{word:20s} -> {ans!r:30s} key {key}", flush=True)
    sims = R.load_sims(OLD / "block_sims.npz")
    pops = R.populations(_Order())
    res = {"picks": picks}
    for L in ("KC", "LH"):
        A = np.log1p(
            np.stack(
                [
                    np.mean([R.rates(sims[f"actual|{b}|{s}"], pops[L]) for s in R.SEEDS], 0)
                    for b in block
                ]
            )
        )
        A = A[:, A.std(0) > 0]
        mu = A.mean(0, keepdims=True)
        S = corr_rows(A - mu, A - mu)
        ranks = []
        for word, key in PAIRS:
            p = picks[word]["picked"]
            if p is None:
                ranks.append(len(block))
                continue
            i, k = block.index(p), block.index(key)
            ranks.append(int((S[i] > S[i, k]).sum()) + 1)
        res[L] = {
            "ranks": ranks,
            "median_rank": float(np.median(ranks)),
            "top5": float(np.mean(np.array(ranks) <= 5)),
            "exact_key": float(np.mean([picks[w]["picked"] == k for w, k in PAIRS])),
        }
        print(
            f"{L}: median rank {res[L]['median_rank']:.0f}, top-5 {res[L]['top5'] * 100:.0f}%, picked the key odorant {res[L]['exact_key'] * 100:.0f}%"
        )
    json.dump(res, open(OUT / "everyday_generative.json", "w"), indent=1)


if __name__ == "__main__":
    main()
