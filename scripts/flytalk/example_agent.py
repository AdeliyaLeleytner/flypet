#!/usr/bin/env python3
"""A minimal FlyTalk agent: standard library only, talks to the HTTP server.

Two built-in policies and a hook for any language model:

    --policy rule        measure every odour, re-measure the most shifted ones, then label at most two shifts
                         beyond 0.10 (a prior-aware threshold rule; 0.852 on seeds 0-59)
    --policy cmd --cmd "claude -p --model sonnet"
                         each turn, pipe the prompt and the log so far to this shell command on stdin and send
                         the first line it prints back as the next action (works with any CLI that reads stdin)

    python scripts/flytalk/example_agent.py --policy rule --seeds 0 5
    python scripts/flytalk/example_agent.py --policy cmd --cmd "ollama run qwen3:32b" --seeds 0 60 --out my_agent.json

The server must be running (scripts/flytalk/server.py). Results are written in the same row format as the other
players, so scripts/flytalk/leaderboard.py can rank them next to the paper's agents.
"""

from __future__ import annotations
import argparse, json, shlex, subprocess, sys, urllib.request
from pathlib import Path


def call(url, path, obj=None):
    data = None if obj is None else json.dumps(obj).encode()
    req = urllib.request.Request(
        url + path, data=data, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def rule_policy(info):
    """Returns act(state) -> next action, using only the measured values."""
    panel, naive, budget = info["panel"], info["naive"], info["budget"]

    def act(state):
        values = state["values"]  # {odour: [measured valences]}
        used = sum(len(v) for v in values.values())
        todo = [o for o in panel if not values.get(o)]
        if todo and used < budget:
            return f"SMELL {todo[0]}"
        shift = {o: sum(v) / len(v) - naive[o] for o, v in values.items() if v}
        if used < budget:  # re-measure the odour closest to the 0.10 threshold
            return f"SMELL {min(shift, key=lambda o: abs(abs(shift[o]) - 0.10))}"
        keep = sorted([o for o in shift if abs(shift[o]) > 0.10], key=lambda o: -abs(shift[o]))[:2]
        return "ANSWER\n" + "\n".join(
            f"{o}: {('+' if shift[o] > 0 else '-') if o in keep else '0'}" for o in panel
        )

    return act


def cmd_policy(cmd):
    def act(state):
        text = f"{state['prompt']}\n\nExperiment log so far:\n{state['log'] or '(nothing yet)'}\n\nWrite your next line."
        if state["last_turn"]:
            text += "\nThis is your last turn: write ANSWER now."
        out = subprocess.run(
            shlex.split(cmd), input=text, capture_output=True, text=True, timeout=900
        ).stdout
        lines = [l.strip() for l in out.strip().strip("`").splitlines() if l.strip()]
        for i, l in enumerate(lines):  # an answer spans several lines
            if l.strip("*").upper().startswith(("ANSWER", "FINISH")):
                return "\n".join(lines[i:])
        return lines[0] if lines else ""

    return act


def play(url, seed, policy, info):
    ep = call(url, "/reset", {"task": "diagnose", "seed": seed})
    values, log = {}, ""
    for turn in range(info["budget"] + 4):
        line = policy(
            {
                "prompt": ep["prompt"],
                "log": log,
                "values": values,
                "last_turn": turn == info["budget"] + 3,
            }
        )
        r = call(url, "/step", {"eid": ep["eid"], "action": line or "(empty)"})
        log = r["transcript"]
        if line.startswith("SMELL ") and "valence" in r["observation"]:
            odour = line[6:].strip()
            values.setdefault(odour, []).append(float(r["observation"].split(":")[1].split("(")[0]))
        if r["done"]:
            break
    res = call(url, "/result", {"eid": ep["eid"]})
    if not res["done"]:
        call(url, "/step", {"eid": ep["eid"], "action": "ANSWER"})
        res = call(url, "/result", {"eid": ep["eid"]})
    return {
        "task": "diagnose",
        "seed": seed,
        **res["result"],
        "history": res["history"],
        "transcript": res["transcript"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--policy", choices=["rule", "cmd"], default="rule")
    ap.add_argument(
        "--cmd", help="shell command for --policy cmd; reads the prompt on stdin, prints the action"
    )
    ap.add_argument(
        "--seeds",
        type=int,
        nargs=2,
        default=[0, 60],
        metavar=("FIRST", "STOP"),
        help="test flies are 0-59",
    )
    ap.add_argument("--out", type=Path, help="write the per-fly rows here (JSON)")
    a = ap.parse_args(argv)
    info = call(a.url, "/info")
    policy = rule_policy(info) if a.policy == "rule" else cmd_policy(a.cmd)
    rows = []
    for seed in range(*a.seeds):
        r = play(a.url, seed, policy, info)
        rows.append(r)
        print(
            f"fly {seed}: score {r['score']:.3f}, all seven right {r['exact']}, measurements {r['n_actions']}",
            flush=True,
        )
    print(f"mean score {sum(r['score'] for r in rows) / len(rows):.3f} over {len(rows)} flies")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    sys.exit(main())
