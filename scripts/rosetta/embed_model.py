#!/usr/bin/env python3
"""Hidden states of any causal LM for every DoOR odorant name, at four relative depths (1/4, 1/2, 3/4, last).

--pool name: mean over the name tokens of "The odorant {name}"; --pool last: last token of "{name} smells like".
Output keys: L1..L4 for the four depths, plus the absolute layer numbers.
"""

import argparse, sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers import AutoModelForCausalLM, AutoTokenizer
from flypet.odor import DoorOdor

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--model", required=True)
ap.add_argument("--pool", choices=["name", "last"], required=True)
ap.add_argument("--output", type=Path, required=True)
a = ap.parse_args()
template = "The odorant {name}" if a.pool == "name" else "{name} smells like"
device = "cuda" if torch.cuda.is_available() else "mps"
door = DoorOdor()
keys = [k for k in door.resp.index if k in door.key2name]
names = [door.key2name[k] for k in keys]
tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device).eval()
n_layers = model.config.num_hidden_layers
layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
n_pre = len(tok("The odorant")["input_ids"])
out = {l: [] for l in layers}
t0 = time.time()
with torch.no_grad():
    for nm in names:
        ids = tok(template.format(name=nm), return_tensors="pt").to(device)
        hs = model(**ids, output_hidden_states=True).hidden_states
        sl = slice(n_pre, None) if a.pool == "name" else slice(-1, None)
        for l in layers:
            out[l].append(hs[l][0, sl].float().mean(0).cpu().numpy())
np.savez(
    a.output,
    keys=np.array(keys),
    names=np.array(names),
    layers=np.array(layers),
    template=template,
    **{f"L{i + 1}": np.stack(out[l]) for i, l in enumerate(layers)},
)
print(f"{a.model}: {len(names)} names, layers {layers}, {time.time() - t0:.0f} s -> {a.output}")
