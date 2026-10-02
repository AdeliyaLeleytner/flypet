"""Training-only four-example overfit check, using saved INITIAL physiological features."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from flypet.physiology_language import MODEL_ID, MODEL_REVISION
from flypet.physiology_memory_task import OBJECTS, LOCATIONS, question_text


def main(a):
    root = Path(a.results)
    torch.set_num_threads(4)
    torch.manual_seed(713)
    report = json.loads((root / "report.json").read_text())
    c = report["config"]
    data = np.load(root / "initial_state_probe.npz")
    rows = [json.loads(x) for x in (Path(a.data) / "task/train.jsonl").read_text().splitlines()]
    ids = [
        next(i for i, r in enumerate(rows) if r["query_object"] == "key" and r["answer"] == color)
        for color in LOCATIONS
    ]
    chosen = [rows[i] for i in ids]
    features = torch.tensor(data["train_x"][ids], device="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    lm = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
        .requires_grad_(False)
    )
    embedding = lm.get_input_embeddings()
    dimension = embedding.weight.shape[1]
    tokens = c["prefix_tokens"]
    decoder = nn.Sequential(
        nn.Linear(features.shape[-1], c["hidden"]),
        nn.SiLU(),
        nn.Linear(c["hidden"], tokens * dimension),
    ).cuda()
    norm = nn.LayerNorm(dimension).cuda()
    state = torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    decoder.load_state_dict(
        {k[len("decoder.") :]: v for k, v in state.items() if k.startswith("decoder.")}
    )
    norm.load_state_dict(
        {k[len("output_norm.") :]: v for k, v in state.items() if k.startswith("output_norm.")}
    )
    q = tok.apply_chat_template(
        [{"role": "user", "content": question_text(chosen[0])}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    targets = [
        tok(color + tok.eos_token, add_special_tokens=False)["input_ids"] for color in LOCATIONS
    ]
    all_ids = torch.tensor([q + t for t in targets], device="cuda")
    tail = embedding(all_ids).detach()
    question = embedding(torch.tensor([q], device="cuda")).detach()
    labels = torch.full(
        (4, tokens + len(q) + len(targets[0])), -100, device="cuda", dtype=torch.long
    )
    labels[:, tokens + len(q) :] = torch.tensor(targets, device="cuda")
    params = list(decoder.parameters()) + list(norm.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    history = []
    start = time.monotonic()
    for step in range(301):
        if time.monotonic() - start > a.max_seconds:
            break
        prefix = norm(decoder(features).reshape(4, tokens, dimension)) * report["embedding_std"]
        inputs = torch.cat([prefix.to(tail.dtype), tail], 1)
        loss = lm(
            inputs_embeds=inputs,
            attention_mask=torch.ones_like(labels),
            labels=labels,
            use_cache=False,
        ).loss
        if step % 25 == 0:
            with torch.no_grad():
                pure = torch.cat([prefix.to(tail.dtype), question.expand(4, -1, -1)], 1)
                output = lm.generate(
                    inputs_embeds=pure,
                    attention_mask=torch.ones(pure.shape[:2], device="cuda", dtype=torch.long),
                    max_new_tokens=8,
                    do_sample=False,
                    pad_token_id=tok.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                text = [
                    tok.decode(t, skip_special_tokens=True).strip().lower().rstrip(".!").strip()
                    for t in output.sequences[:, -len(output.scores) :]
                ]
            history.append(
                {
                    "step": step,
                    "loss": float(loss),
                    "generated": text,
                    "accuracy": sum(x == y for x, y in zip(text, LOCATIONS)) / 4,
                }
            )
            payload = {
                "scope": "Train-only memorization check; four fixed initial physiological states, fixedkey query; no generalization claim",
                "ids": [r["id"] for r in chosen],
                "history": history,
                "seconds": time.monotonic() - start,
            }
            (root / "decoder_overfit_check.json").write_text(json.dumps(payload, indent=2))
            if history[-1]["accuracy"] == 1.0 and float(loss) < 0.05:
                break
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
    print(json.dumps(history[-1]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--max-seconds", type=int, default=120)
    main(p.parse_args())
