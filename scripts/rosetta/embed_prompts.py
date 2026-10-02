#!/usr/bin/env python3
"""Qwen3-4B-Base hidden states for every DoOR odorant under a prompt template.

--pool name: mean over the odorant-name tokens; --pool last: the final prompt token (e.g. after "smells like").
"""

import argparse, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer, AutoModelForCausalLM
from flypet.odor import DoorOdor

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--template", default="{name} smells like")
ap.add_argument("--pool", choices=["name", "last"], default="last")
ap.add_argument("--output", type=Path, required=True)
ap.add_argument("--model", default="Qwen/Qwen3-4B-Base")
a = ap.parse_args()
door = DoorOdor()
keys = [k for k in door.resp.index if k in door.key2name]
names = [door.key2name[k] for k in keys]
tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
model = (
    AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, local_files_only=True)
    .to("mps")
    .eval()
)
layers = [9, 18, 27, 36]
out = {l: [] for l in layers}
t0 = time.time()
with torch.no_grad():
    for nm in names:
        text = a.template.format(name=nm)
        ids = tok(text, return_tensors="pt").to("mps")
        hs = model(**ids, output_hidden_states=True).hidden_states
        if a.pool == "last":
            sl = slice(-1, None)
        else:
            pre = a.template.split("{name}")[0].rstrip()
            n_pre = len(tok(pre)["input_ids"]) if pre else 0
            n_post = (
                len(tok(a.template.split("{name}")[1])["input_ids"])
                if a.template.split("{name}")[1].strip()
                else 0
            )
            sl = slice(n_pre, ids["input_ids"].shape[1] - n_post)
        for l in layers:
            out[l].append(hs[l][0, sl].float().mean(0).cpu().numpy())
np.savez(
    a.output,
    keys=np.array(keys),
    names=np.array(names),
    template=a.template,
    pool=a.pool,
    **{f"L{l}": np.stack(v) for l, v in out.items()},
)
print(f"{len(names)} prompts in {time.time() - t0:.0f} s -> {a.output}")
