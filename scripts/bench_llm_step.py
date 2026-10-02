import time, sys, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

dev = sys.argv[1]
attn = sys.argv[2] if len(sys.argv) > 2 else "sdpa"
dtype = getattr(torch, sys.argv[3]) if len(sys.argv) > 3 else torch.float32
B = int(sys.argv[4]) if len(sys.argv) > 4 else 8
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
m = (
    AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", dtype=dtype, attn_implementation=attn)
    .to(dev)
    .eval()
)
m.requires_grad_(False)
E = m.get_input_embeddings()
d = E.weight.shape[1]
soft = torch.zeros((B, 16, d), device=dev, dtype=dtype, requires_grad=True)
ids = torch.randint(0, 1000, (B, 100), device=dev)
emb = E(ids)


def step():
    inp = torch.cat([soft, emb], 1)
    lab = torch.cat([torch.full((B, 16), -100, device=dev, dtype=torch.long), ids], 1)
    loss = m(inputs_embeds=inp, labels=lab).loss
    loss.backward()
    return float(loss)


t0 = time.time()
step()
t1 = time.time()
step()
t2 = time.time()
print(
    f"{dev} {attn} {dtype} B={B}: first step {t1 - t0:.1f}s, second step {t2 - t1:.1f}s", flush=True
)
