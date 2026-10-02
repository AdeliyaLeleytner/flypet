#!/usr/bin/env python3
"""Multi-turn GRPO for a language model that talks to the FlyTalk fly (flypet.flytalk).

Each episode is one chat: the task description, then alternating model actions and the fly's observations. Token ids
are stitched by hand (prompt, generated completion, "<|im_start|>user ... <|im_start|>assistant" block) so that the
sequence the model generated from is exactly the sequence it is trained on; the loss covers generated tokens only.
Groups are several rollouts of the same fly (same hidden history and task); the advantage is the episode score
normalised within its group and applied to every generated token of the episode. One gradient step per batch.

--eval-only runs greedy episodes and reports scores (the zero-shot baseline).
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.flytalk import PANEL, Env, describe
from flypet.teach import FlyPool

OUT = ROOT / "data/flytalk_rl_20260926"


class Agent:
    def __init__(self, model_path, lora_r, device="cuda"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).to(
            device
        )
        self.end = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.end
        self.peft = False
        if lora_r:
            from peft import LoraConfig, get_peft_model

            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            self.model.enable_input_require_grads()
            self.model = get_peft_model(
                self.model,
                LoraConfig(
                    r=lora_r,
                    lora_alpha=2 * lora_r,
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
            self.peft = True
        self.turn_tail = self.tok(
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False
        )["input_ids"]

    def first_ids(self, text):
        rendered = self.tok.apply_chat_template(
            [{"role": "user", "content": text}],
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=False,
        )
        return list(self.tok(rendered, add_special_tokens=False)["input_ids"])

    def obs_ids(self, obs):
        return (
            self.tok("\n<|im_start|>user\n" + obs, add_special_tokens=False)["input_ids"]
            + self.turn_tail
        )

    @torch.no_grad()
    def generate(self, seqs, greedy, max_new, chunk=16, temperature=1.0):
        self.model.eval()
        outs = []
        for c0 in range(0, len(seqs), chunk):
            part = seqs[c0 : c0 + chunk]
            L = max(len(s) for s in part)
            ids = torch.tensor([[self.pad] * (L - len(s)) + s for s in part], device="cuda")
            att = torch.tensor([[0] * (L - len(s)) + [1] * len(s) for s in part], device="cuda")
            kw = (
                dict(do_sample=False)
                if greedy
                else dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
            )
            g = self.model.generate(
                input_ids=ids,
                attention_mask=att,
                max_new_tokens=max_new,
                pad_token_id=self.pad,
                eos_token_id=self.end,
                **kw,
            )[:, L:].tolist()
            for row in g:
                cut = row.index(self.end) + 1 if self.end in row else len(row)
                comp = row[:cut]
                if not comp or comp[-1] != self.end:
                    comp = comp + [self.end]
                outs.append(comp)
        return outs


def with_brief(prompt, brief):
    """Background inserted before the task instructions, exactly as scripts/flytalk/claude_agent.py does."""
    if not brief:
        return prompt
    head, sep, tail = prompt.partition("Your task:")
    return f"{head}{brief.strip()}\n\n{sep}{tail}" if sep else f"{prompt}\n\n{brief.strip()}"


def rollout(agent, env, specs, greedy, max_turns, max_new, temperature=1.0, brief=""):
    """specs: [(task, seed)] -> episodes with token ids, generated-token mask and result."""
    eps = env.reset_many(
        [(t, s, f"{t}-{s}-{k}-{time.time_ns()}") for k, (t, s) in enumerate(specs)]
    )
    runs = [
        {
            "eid": e.eid,
            "ids": agent.first_ids(with_brief(describe(e, env), brief)),
            "mask": None,
            "texts": [],
        }
        for e in eps
    ]
    for r in runs:
        r["mask"] = [0] * len(r["ids"])
    for turn in range(max_turns):
        live = [r for r in runs if not env.episodes[r["eid"]].done]
        if not live:
            break
        last = turn == max_turns - 1
        comps = agent.generate([r["ids"] for r in live], greedy, max_new, temperature=temperature)
        items = []
        for r, c in zip(live, comps):
            text = agent.tok.decode(c, skip_special_tokens=True).strip()
            if last and not text.upper().startswith(("ANSWER", "FINISH")):
                text = (
                    "ANSWER\n" if env.episodes[r["eid"]].task == "diagnose" else "FINISH\n"
                ) + text
            r["ids"] += c
            r["mask"] += [1] * len(c)
            r["texts"].append(text)
            items.append((r["eid"], text))
        obs = env.step_many(items)
        for r in live:
            if not env.episodes[r["eid"]].done:
                o = agent.obs_ids(obs[r["eid"]])
                r["ids"] += o
                r["mask"] += [0] * len(o)
    for r in runs:
        ep = env.episodes[r["eid"]]
        if not ep.done:  # should not happen: the last turn always finishes
            env.step_many([(r["eid"], "ANSWER" if ep.task == "diagnose" else "FINISH")])
        r["result"] = ep.result
        r["task"] = ep.task
        r["history"] = ep.history
    return runs


THINK_END = "</think>"


def rollout_chat(agent, env, specs, max_turns, max_new, brief="", thinking=True, seed=0):
    """Evaluation with the official chat template: each turn the whole conversation is re-rendered, earlier turns keep
    only the action (Qwen3 drops old reasoning from history), and thinking mode samples with Qwen's recommended settings
    (temperature 0.6, top-p 0.95, top-k 20). Returns runs like rollout(), plus the raw completions."""
    torch.manual_seed(seed)
    eps = env.reset_many(
        [(t, s, f"{t}-{s}-{k}-{time.time_ns()}") for k, (t, s) in enumerate(specs)]
    )
    runs = [
        {
            "eid": e.eid,
            "messages": [{"role": "user", "content": with_brief(describe(e, env), brief)}],
            "texts": [],
            "raw": [],
        }
        for e in eps
    ]
    kw = (
        dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
        if thinking
        else dict(do_sample=False)
    )
    for turn in range(max_turns):
        live = [r for r in runs if not env.episodes[r["eid"]].done]
        if not live:
            break
        prompts = [
            agent.tok.apply_chat_template(
                r["messages"], add_generation_prompt=True, enable_thinking=thinking, tokenize=False
            )
            for r in live
        ]
        seqs = [list(agent.tok(p, add_special_tokens=False)["input_ids"]) for p in prompts]
        outs = []
        agent.model.eval()
        with torch.no_grad():
            for c0 in range(0, len(seqs), 16):
                part = seqs[c0 : c0 + 16]
                L = max(len(x) for x in part)
                ids = torch.tensor([[agent.pad] * (L - len(x)) + x for x in part], device="cuda")
                att = torch.tensor([[0] * (L - len(x)) + [1] * len(x) for x in part], device="cuda")
                g = agent.model.generate(
                    input_ids=ids,
                    attention_mask=att,
                    max_new_tokens=max_new,
                    pad_token_id=agent.pad,
                    eos_token_id=agent.end,
                    **kw,
                )[:, L:].tolist()
                outs += [
                    __import__("re").sub(
                        r"<\|[^|]*\|>",
                        "",
                        agent.tok.decode(
                            row[: row.index(agent.end)] if agent.end in row else row,
                            skip_special_tokens=False,
                        ),
                    )
                    for row in g
                ]
        items = []
        last = turn == max_turns - 1
        for r, raw in zip(live, outs):
            action = (
                raw.split(THINK_END, 1)[1].strip()
                if THINK_END in raw
                else (raw.strip() if not thinking else "")
            )
            if last and not action.upper().startswith(("ANSWER", "FINISH")):
                action = (
                    "ANSWER\n" if env.episodes[r["eid"]].task == "diagnose" else "FINISH\n"
                ) + action
            r["raw"].append(raw)
            r["texts"].append(action)
            r["messages"].append({"role": "assistant", "content": action or "(no action)"})
            items.append((r["eid"], action or "(no action)"))
        obs = env.step_many(items)
        for r in live:
            if not env.episodes[r["eid"]].done:
                r["messages"].append({"role": "user", "content": obs[r["eid"]]})
    for r in runs:
        ep = env.episodes[r["eid"]]
        if not ep.done:
            env.step_many([(r["eid"], "ANSWER" if ep.task == "diagnose" else "FINISH")])
        r["result"] = ep.result
        r["task"] = ep.task
        r["history"] = ep.history
    return runs


def token_logps(model, ids, mask):
    x = torch.tensor([ids], device="cuda")
    logits = model(input_ids=x).logits[0, :-1].float()
    tgt = x[0, 1:]
    m = torch.tensor(mask[1:], device="cuda", dtype=torch.bool)
    return torch.log_softmax(logits[m], -1).gather(-1, tgt[m].unsqueeze(-1)).squeeze(-1)


def summarize(runs):
    out = {}
    for t in ("diagnose", "erase"):
        sel = [r["result"] for r in runs if r["task"] == t]
        if sel:
            out[t] = {
                "n": len(sel),
                "score": float(np.mean([s["score"] for s in sel])),
                "exact": float(np.mean([s["exact"] for s in sel])),
                "actions": float(np.mean([s["n_actions"] for s in sel])),
            }
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=40)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument(
        "--eval-seeds",
        type=int,
        default=60,
        help="evaluation flies per task (seeds 0..n-1, the shared test set)",
    )
    ap.add_argument(
        "--tasks", default="diagnose", help="comma-separated tasks for training and evaluation"
    )
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--flies", type=int, default=8, help="distinct flies (groups) per step")
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--max-turns", type=int, default=14)
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--skip-eval0", action="store_true", help="reuse an earlier step-0 evaluation")
    ap.add_argument(
        "--brief",
        default=None,
        help="text file inserted before 'Your task:' in every prompt, training and evaluation",
    )
    ap.add_argument("--adapter", default=None, help="LoRA adapter to evaluate (with --eval-only)")
    ap.add_argument(
        "--thinking",
        action="store_true",
        help="evaluation with Qwen3 thinking on (chat-template rollouts)",
    )
    a = ap.parse_args(argv)
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    env = Env(FlyPool(a.workers), PANEL)
    agent = Agent(a.model, 0 if a.eval_only else a.lora_r)
    if a.adapter:
        from peft import PeftModel

        agent.model = PeftModel.from_pretrained(agent.model, a.adapter).eval()
    brief = open(a.brief).read() if a.brief else ""
    tasks = a.tasks.split(",")
    eval_specs = [(t, s) for t in tasks for s in range(a.eval_seeds)]
    t0 = time.time()
    if a.thinking:
        ev = rollout_chat(
            agent, env, eval_specs, a.max_turns, a.max_new, brief=brief, thinking=True
        )
        json.dump(
            [{k: r[k] for k in ("task", "history", "texts", "raw", "result")} for r in ev],
            open(out / "eval_step0.json", "w"),
            indent=1,
            default=str,
        )
        print("step 0 eval:", json.dumps(summarize(ev)), f"{time.time() - t0:.0f} s", flush=True)
        return
    if not a.skip_eval0:
        ev = rollout(agent, env, eval_specs, True, a.max_turns, a.max_new, brief=brief)
        json.dump(
            [{k: r[k] for k in ("task", "history", "texts", "result")} for r in ev],
            open(out / "eval_step0.json", "w"),
            indent=1,
            default=str,
        )
        print("step 0 eval:", json.dumps(summarize(ev)), f"{time.time() - t0:.0f} s", flush=True)
    if a.eval_only:
        return
    opt = torch.optim.AdamW(
        [p for p in agent.model.parameters() if p.requires_grad],
        lr=a.lr,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    rng = np.random.default_rng(1)
    log = open(out / "log.jsonl", "w")
    for step in range(1, a.steps + 1):
        seeds = rng.integers(
            1000, 10**6, size=a.flies
        )  # training flies never overlap evaluation seeds 0..eval_seeds-1
        picked = rng.choice(tasks, size=a.flies)
        specs = [(str(t), int(s)) for t, s in zip(picked, seeds) for _ in range(a.group)]
        runs = rollout(agent, env, specs, False, a.max_turns, a.max_new, a.temperature, brief=brief)
        R = torch.tensor([r["result"]["score"] for r in runs], dtype=torch.float32).view(
            a.flies, a.group
        )
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).view(-1)
        agent.model.train()
        opt.zero_grad(set_to_none=True)
        ntok = sum(sum(r["mask"][1:]) for r in runs)
        kl_sum = 0.0
        for r, A in zip(runs, adv.tolist()):
            if abs(A) < 1e-8:
                continue
            lp = token_logps(agent.model, r["ids"], r["mask"])
            loss = -A * lp.sum()
            if a.beta > 0:
                with torch.no_grad(), agent.model.disable_adapter():
                    rlp = token_logps(agent.model, r["ids"], r["mask"])
                dlt = rlp - lp
                kl = (torch.exp(dlt) - dlt - 1).sum()
                loss = loss + a.beta * kl
                kl_sum += float(kl)
            (loss / ntok).backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in agent.model.parameters() if p.requires_grad], 1.0
        )
        opt.step()
        rec = {
            "step": step,
            "score": float(R.mean()),
            "by_task": summarize(runs),
            "kl": kl_sum / max(ntok, 1),
            "zero_std_groups": int((R.std(1) < 1e-6).sum()),
            "elapsed_s": round(time.time() - t0, 1),
        }
        if step % 10 == 0 or step == a.steps:
            ev = rollout(agent, env, eval_specs, True, a.max_turns, a.max_new, brief=brief)
            rec["eval"] = summarize(ev)
            json.dump(
                [{k: r[k] for k in ("task", "history", "texts", "result")} for r in ev],
                open(out / f"eval_step{step}.json", "w"),
                indent=1,
                default=str,
            )
            agent.model.save_pretrained(out / f"adapter_step{step}")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
