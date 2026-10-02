#!/usr/bin/env python3
"""GRPO for a small language model that writes fly-training protocols.

Reward = success + 0.2 * graded score - 0.05 * invalid lines (capped at 4), computed by the surrogate fly
(flypet.teach_tasks.FastSurrogate); a protocol with no valid trial gets -0.5. Training tasks are every target
vector with 1-3 non-zero entries that is not one of the 76 evaluation tasks; 30 of them are held out as a
validation set for checkpoint selection. The evaluation tasks are only decoded (greedy) at the start and end.

One gradient step per sampled batch, so the PPO ratio is 1 and GRPO reduces to REINFORCE with a
group-normalised advantage; a k3 KL penalty to the initial model is optional (--beta). The policy is a LoRA
adapter on a bf16 base (all attention and MLP projections); the reference is the same model with the adapter off.
"""

from __future__ import annotations
import argparse, json, sys, time
from itertools import combinations, product
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.teach import Task, parse_protocol, score
from flypet.teach_prompt import SYSTEM, messages, user_prompt
from flypet.teach_tasks import FastSurrogate, make_tasks

DATA = ROOT / "data/teach_20260925"


def target_name(tg, panel):
    return " ".join(f"{'+' if tg[p] > 0 else '-'}{p}" for p in panel if tg[p])


def task_pool(panel, budget, exclude):
    pool = []
    for k in (1, 2, 3):
        for idx in combinations(range(len(panel)), k):
            for signs in product((-1, 1), repeat=k):
                tg = {p: 0 for p in panel}
                for i, s in zip(idx, signs):
                    tg[panel[i]] = s
                if tuple(tg[p] for p in panel) not in exclude:
                    pool.append(Task(panel, tg, budget, name=target_name(tg, panel)))
    return pool


def reward(sur, task, text):
    acts, invalid = parse_protocol(text, task.panel, task.budget)
    if not acts:
        return -0.5, 0.0, -1.0, acts, invalid
    s, r, _ = score(task, sur.run(acts), sur.naive)
    return s + 0.2 * r - 0.05 * min(invalid, 4), s, r, acts, invalid


class RealFly:
    """Reward from the whole-brain fly: each protocol runs with its own random training seed and one probe seed
    that differs from the evaluation seeds (101-103), so the policy cannot fit particular Poisson draws."""

    def __init__(self, panel, workers, probe_seed, rng):
        from flypet.teach import FlyPool

        self.panel, self.probe_seed, self.rng = panel, probe_seed, rng
        self.pool = FlyPool(workers)
        base = self.pool.episodes([("naive", [], panel, 5000, (probe_seed,))])
        self.naive = base["naive"]["valence"]

    def rewards(self, pairs):
        parsed = [parse_protocol(x, t.panel, t.budget) for t, x in pairs]
        jobs = [
            (str(i), acts, self.panel, int(self.rng.integers(6000, 9000)), (self.probe_seed,))
            for i, (acts, _) in enumerate(parsed)
            if acts
        ]
        res = self.pool.episodes(jobs) if jobs else {}
        out = []
        for i, ((t, _), (acts, inv)) in enumerate(zip(pairs, parsed)):
            if not acts:
                out.append((-0.5, 0.0, -1.0, acts, inv))
                continue
            s, r, _ = score(t, res[str(i)]["valence"], self.naive)
            out.append((s + 0.2 * r - 0.05 * min(inv, 4), s, r, acts, inv))
        return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--group", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--gen-chunk", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=80)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--tag", default="qwen3-0.6b")
    ap.add_argument("--reward", choices=["surrogate", "real"], default="surrogate")
    ap.add_argument(
        "--init-adapter",
        default=None,
        help="start from a saved LoRA adapter (e.g. the surrogate-trained one)",
    )
    ap.add_argument("--real-workers", type=int, default=96)
    ap.add_argument("--real-probe-seed", type=int, default=201)
    ap.add_argument(
        "--plain",
        action="store_true",
        help="base model: plain-text prompt instead of the chat template",
    )
    a = ap.parse_args(argv)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    out = DATA / f"grpo_{a.tag}"
    out.mkdir(parents=True, exist_ok=True)
    sur = FastSurrogate(DATA / "surrogate.pkl")
    test = make_tasks(sur.panel, budget=sur.budget)
    pool = task_pool(sur.panel, sur.budget, {tuple(t.targets[p] for p in sur.panel) for t in test})
    perm = rng.permutation(len(pool))
    val, train = [pool[i] for i in perm[:30]], [pool[i] for i in perm[30:]]
    print(f"train tasks {len(train)}, validation {len(val)}, test {len(test)}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
    policy = get_peft_model(
        base,
        LoraConfig(
            r=a.lora_r,
            lora_alpha=2 * a.lora_r,
            lora_dropout=0.0,
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.enable_input_require_grads()
    if a.init_adapter:
        from peft import load_peft_weights

        set_peft_model_state_dict(policy, load_peft_weights(a.init_adapter))
        print("initialised from", a.init_adapter, flush=True)
    policy.print_trainable_parameters()
    real = (
        RealFly(sur.panel, a.real_workers, a.real_probe_seed, np.random.default_rng(1))
        if a.reward == "real"
        else None
    )
    score_pairs = (
        (lambda pairs: real.rewards(pairs))
        if real
        else (lambda pairs: [reward(sur, t, x) for t, x in pairs])
    )
    opt = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=a.lr,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    snapshot = lambda: {k: v.detach().clone() for k, v in get_peft_model_state_dict(policy).items()}
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    if a.plain:
        render = lambda t: f"{SYSTEM}\n\n{user_prompt(t)}\n\nProtocol:\n"
    else:
        render = lambda t: tok.apply_chat_template(
            messages(t), tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def generate(tasks, n, greedy=False):
        enc = tok(
            [render(t) for t in tasks for _ in range(n)], return_tensors="pt", padding=True
        ).to("cuda")
        policy.eval()
        kw = (
            dict(do_sample=False)
            if greedy
            else dict(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
        )
        parts = []
        with torch.no_grad():
            for c0 in range(0, enc["input_ids"].shape[0], a.gen_chunk):
                sl = slice(c0, c0 + a.gen_chunk)
                seq = policy.generate(
                    input_ids=enc["input_ids"][sl],
                    attention_mask=enc["attention_mask"][sl],
                    max_new_tokens=a.max_new,
                    pad_token_id=tok.pad_token_id,
                    **kw,
                )
                parts.append(seq[:, enc["input_ids"].shape[1] :])
        width = max(p.shape[1] for p in parts)
        comp = torch.cat(
            [
                torch.nn.functional.pad(p, (0, width - p.shape[1]), value=tok.pad_token_id)
                for p in parts
            ],
            0,
        )
        mask = torch.ones_like(comp)
        for i in range(comp.shape[0]):
            hit = [j for j, x in enumerate(comp[i].tolist()) if x in eos]
            if hit:
                mask[i, hit[0] + 1 :] = 0
        return enc, comp, mask, tok.batch_decode(comp, skip_special_tokens=True)

    def logps(model, enc, comp, cmask, rows):
        ids = torch.cat([enc["input_ids"][rows], comp[rows]], 1)
        attn = torch.cat([enc["attention_mask"][rows], cmask[rows]], 1)
        pos = (attn.cumsum(-1) - 1).clamp(min=0)
        k = comp.shape[1]
        logits = (
            model(input_ids=ids, attention_mask=attn, position_ids=pos, logits_to_keep=k + 1)
            .logits[:, :-1]
            .float()
        )
        return torch.log_softmax(logits, -1).gather(-1, comp[rows].unsqueeze(-1)).squeeze(-1)

    def evaluate(tasks, label):
        _, _, _, texts = generate(tasks, 1, greedy=True)
        res = score_pairs(list(zip(tasks, texts)))
        return (
            {
                "label": label,
                "success": float(np.mean([r[1] for r in res])),
                "graded": float(np.mean([r[2] for r in res])),
                "valid": float(np.mean([len(r[3]) > 0 for r in res])),
                "invalid_lines": float(np.mean([r[4] for r in res])),
            },
            texts,
            res,
        )

    log = open(out / "log.jsonl", "w")
    if real is None:
        first, first_texts, first_res = evaluate(test, "test@0")
    else:  # save the untrained model's test protocols; eval_teachers scores them on the test seeds
        _, _, _, first_texts = generate(test, 1, greedy=True)
        first_res = [
            (None, None, None, *parse_protocol(x, t.panel, t.budget))
            for t, x in zip(test, first_texts)
        ]
        first = {"label": "test@0", "note": "scored by eval_teachers"}
    json.dump(
        {
            t.name: {"text": x, "actions": r[3], "invalid": r[4]}
            for t, x, r in zip(test, first_texts, first_res)
        },
        open(out / "test_protocols_step0.json", "w"),
        indent=1,
    )
    v0, _, _ = evaluate(val, "val@0")
    print(json.dumps(first), json.dumps(v0), flush=True)
    log.write(json.dumps({"step": 0, "test": first, "val": v0}) + "\n")
    log.flush()
    best_val, t0 = v0["success"], time.time()
    best_state = snapshot()
    for step in range(1, a.steps + 1):
        batch = [train[i] for i in rng.choice(len(train), size=a.prompts, replace=False)]
        enc, comp, cmask, texts = generate(batch, a.group)
        res = score_pairs([(batch[i // a.group], x) for i, x in enumerate(texts)])
        R = torch.tensor([r[0] for r in res], dtype=torch.float32).view(a.prompts, a.group)
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).view(-1).cuda()
        policy.train()
        opt.zero_grad(set_to_none=True)
        ntok = cmask.sum().clamp(min=1)
        kl_sum = 0.0
        for s0 in range(0, comp.shape[0], a.micro):
            rows = torch.arange(s0, min(s0 + a.micro, comp.shape[0]), device="cuda")
            lp = logps(policy, enc, comp, cmask, rows)
            m = cmask[rows].float()
            loss_tok = -adv[rows, None] * lp
            if a.beta > 0:
                with torch.no_grad(), policy.disable_adapter():
                    rlp = logps(policy, enc, comp, cmask, rows)
                d = rlp - lp
                kl = torch.exp(d) - d - 1
                loss_tok = loss_tok + a.beta * kl
                kl_sum += float((kl * m).sum())
            ((loss_tok * m).sum() / ntok).backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()
        rec = {
            "step": step,
            "reward": float(R.mean()),
            "success": float(np.mean([r[1] for r in res])),
            "valid": float(np.mean([len(r[3]) > 0 for r in res])),
            "kl": kl_sum / float(ntok),
            "zero_std_groups": int((R.std(1) < 1e-6).sum()),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if step % a.eval_every == 0 or step == a.steps:
            v, _, _ = evaluate(val, f"val@{step}")
            rec["val"] = v
            if v["success"] > best_val:
                best_val = v["success"]
                rec["saved"] = True
                best_state = snapshot()
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)
    # final: best-by-validation checkpoint on the test tasks
    set_peft_model_state_dict(policy, best_state)
    policy.save_pretrained(out / "best_adapter")
    final, texts, res = evaluate(test, "test@best")
    json.dump(
        {
            t.name: {"text": x, "actions": r[3], "invalid": r[4]}
            for t, x, r in zip(test, texts, res)
        },
        open(out / "test_protocols_best.json", "w"),
        indent=1,
    )
    log.write(json.dumps({"final_test": final, "best_val_success": best_val}) + "\n")
    log.close()
    print("final test (surrogate):", json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
