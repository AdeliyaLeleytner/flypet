"""Qwen3-4B-Base hidden states for DoOR odorant names (mean over the name's tokens)."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])  # repository root
import sys, time, numpy as np, torch

sys.path.insert(0, _ROOT + "")
from transformers import AutoTokenizer, AutoModelForCausalLM
from flypet.odor import DoorOdor

door = DoorOdor()
keys = [k for k in door.resp.index if k in door.key2name]
names = [door.key2name[k] for k in keys]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")
model = (
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-4B-Base", dtype=torch.bfloat16)
    .to("mps")
    .eval()
)
prefix = "The odorant"
n_pre = len(tok(prefix)["input_ids"])
layers = [9, 18, 27, 36]
out = {l: [] for l in layers}
t0 = time.time()
with torch.no_grad():
    for nm in names:
        ids = tok(prefix + " " + nm, return_tensors="pt").to("mps")
        assert ids["input_ids"].shape[1] > n_pre
        hs = model(**ids, output_hidden_states=True).hidden_states
        for l in layers:
            out[l].append(hs[l][0, n_pre:].float().mean(0).cpu().numpy())
np.savez(
    sys.argv[1],
    keys=np.array(keys),
    names=np.array(names),
    **{f"L{l}": np.stack(v) for l, v in out.items()},
)
print(f"embedded {len(names)} names in {time.time() - t0:.0f} s, dim {out[36][0].shape[0]}")
