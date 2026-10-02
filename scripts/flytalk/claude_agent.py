#!/usr/bin/env python3
"""Claude as the experimenter in FlyTalk, through the local `claude` CLI, against a FlyTalk server.

Each turn the model sees the task description and the experiment log so far and writes one line (or the final
ANSWER / FINISH). The server runs the whole-brain fly; episodes use the same seeds as scripts/flytalk/baselines.py,
so every player meets the same flies. Results: data/flytalk_20260926/claude_<model>.json (resumable).
"""

from __future__ import annotations
import argparse, json, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.llm import get_llm

OUT = ROOT / "data/flytalk_20260926"
SYSTEM = (
    "You are an experimental neuroscientist running a behavioural experiment on a fruit fly. "
    "Reply with only the line(s) the instructions ask for, no commentary."
)


def post(url, path, obj):
    req = urllib.request.Request(
        url + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def next_line(reply: str) -> str:
    lines = [l.rstrip() for l in reply.strip().strip("`").splitlines() if l.strip()]
    if not lines:
        return ""
    for i, l in enumerate(lines):  # a final answer spans several lines
        if l.strip().strip("*").upper().startswith(("ANSWER", "FINISH", "DONE")):
            return "\n".join(lines[i:])
    return lines[0]


def play(llm, url, task, seed, max_turns, brief=""):
    ep = post(url, "/reset", {"task": task, "seed": seed})
    eid, prompt, transcript, t0 = ep["eid"], ep["prompt"], "", time.time()
    if brief:  # extra background, inserted before the task instructions
        head, sep, tail = prompt.partition("Your task:")
        prompt = f"{head}{brief.strip()}\n\n{sep}{tail}" if sep else f"{prompt}\n\n{brief.strip()}"
    for turn in range(max_turns):
        end = "ANSWER" if task == "diagnose" else "FINISH"
        user = (
            f"{prompt}\n\nExperiment log so far:\n{transcript or '(nothing yet)'}\n\n"
            f"Write your next line (one action, or {end})."
        )
        if turn == max_turns - 1:
            user += f"\nThis is your last turn: write {end} now."
        reply = llm.complete(SYSTEM, user, max_tokens=600)
        line = next_line(reply if isinstance(reply, str) else json.dumps(reply))
        r = post(url, "/step", {"eid": eid, "action": line or "(empty)"})
        transcript = r["transcript"]
        if r["done"]:
            break
    res = post(url, "/result", {"eid": eid})
    if not res["done"]:  # ran out of turns without finishing
        post(url, "/step", {"eid": eid, "action": "ANSWER" if task == "diagnose" else "FINISH"})
        res = post(url, "/result", {"eid": eid})
    return {
        "task": task,
        "seed": seed,
        **(res["result"] or {}),
        "history": res["history"],
        "transcript": res["transcript"],
        "wall_s": round(time.time() - t0, 1),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--model", default="sonnet")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--tasks", nargs="+", default=["diagnose", "erase"])
    ap.add_argument(
        "--brief",
        default=None,
        help="text file with background inserted before the task instructions",
    )
    ap.add_argument("--tag", default=None, help="suffix of the results file, e.g. informed")
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / (f"claude_{a.model}_{a.tag}.json" if a.tag else f"claude_{a.model}.json")
    brief = open(a.brief).read() if a.brief else ""
    done = {(r["task"], r["seed"]): r for r in json.load(open(path))} if path.exists() else {}
    info = json.loads(urllib.request.urlopen(a.url + "/info", timeout=60).read())
    llm = get_llm("claude-cli", model=a.model)
    jobs = [(t, s) for t in a.tasks for s in range(a.n) if (t, s) not in done]
    print(
        f"{len(jobs)} episodes to play; budget {info['budget']}, delta {info['delta']}", flush=True
    )

    def run(job):
        try:
            return play(llm, a.url, job[0], job[1], info["budget"] + 4, brief)
        except Exception as exc:
            return {"task": job[0], "seed": job[1], "error": f"{type(exc).__name__}: {exc}"}

    with ThreadPoolExecutor(a.threads) as ex:
        for i, r in enumerate(ex.map(run, jobs), 1):
            if "error" not in r:
                done[(r["task"], r["seed"])] = r
            else:
                print("  error", r["task"], r["seed"], r["error"][:200], flush=True)
            if i % 5 == 0 or i == len(jobs):
                json.dump(list(done.values()), open(path, "w"), indent=1)
                print(f"  {i}/{len(jobs)}", flush=True)
    import numpy as np

    for t in a.tasks:
        sel = [r for r in done.values() if r["task"] == t]
        if sel:
            extra = (
                "conditioned right %.3f untouched right %.3f"
                % (
                    np.nanmean([r["conditioned_right"] for r in sel]),
                    np.mean([r["untouched_right"] for r in sel]),
                )
                if t == "diagnose"
                else "restored %.3f kept %.3f"
                % (np.nanmean([r["restored"] for r in sel]), np.mean([r["kept"] for r in sel]))
            )
            print(
                f"{t}: n={len(sel)} score {np.mean([r['score'] for r in sel]):.3f} exact {np.mean([r['exact'] for r in sel]):.3f} "
                f"{extra} actions {np.mean([r['n_actions'] for r in sel]):.1f}"
            )


if __name__ == "__main__":
    main()
