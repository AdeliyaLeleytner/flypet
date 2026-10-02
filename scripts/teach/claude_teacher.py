#!/usr/bin/env python3
"""Zero-shot frontier teacher: one Claude CLI call per evaluation task, same prompt as every other teacher."""

from __future__ import annotations
import argparse, json, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.llm import get_llm
from flypet.teach import parse_protocol
from flypet.teach_prompt import SYSTEM, user_prompt
from flypet.teach_tasks import FastSurrogate, make_tasks

DATA = ROOT / "data/teach_20260925"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(argv)
    sur = FastSurrogate(DATA / "surrogate.pkl")
    tasks = make_tasks(sur.panel, budget=sur.budget)
    llm = get_llm("claude-cli", model=a.model)
    out_path = DATA / f"teacher_claude_{a.model}.json"
    done = json.load(open(out_path)) if out_path.exists() else {}

    def ask(task):
        if task.name in done:
            return task.name, done[task.name]
        t0 = time.time()
        text = llm.complete(SYSTEM, user_prompt(task), max_tokens=400)
        text = text if isinstance(text, str) else json.dumps(text)
        acts, invalid = parse_protocol(text, task.panel, task.budget)
        return task.name, {
            "text": text,
            "actions": acts,
            "invalid": invalid,
            "wall_s": round(time.time() - t0, 1),
        }

    with ThreadPoolExecutor(a.threads) as ex:
        for i, (name, rec) in enumerate(ex.map(ask, tasks), 1):
            done[name] = rec
            if i % 10 == 0 or i == len(tasks):
                json.dump(done, open(out_path, "w"), indent=1)
                print(f"  {i}/{len(tasks)}", flush=True)
    ok = sum(1 for r in done.values() if r["actions"])
    s = [sur.evaluate(t, done[t.name]["actions"])[0] for t in tasks]
    print(
        f"{ok}/{len(tasks)} parsed protocols; surrogate success mean {sum(s) / len(s):.3f}; invalid lines "
        f"{sum(r['invalid'] for r in done.values())}"
    )


if __name__ == "__main__":
    main()
