#!/usr/bin/env python3
"""FlyTalk diagnose leaderboard on the 60 test flies (seeds 0-59).

Collects every player's per-fly rows, ranks them by balanced score and adds the error analysis of the paper:
false alarms on odours whose valence moved through generalisation (expected shift above 0.10 in the Bayes table)
versus odours that did not move, trained odours missed, the ester twin, and flies with more than two odours named.

    python scripts/flytalk/leaderboard.py                      # all recorded players
    python scripts/flytalk/leaderboard.py --extra my_agent.json --name my_agent

Writes data/flytalk_20260926/leaderboard.csv and output/flytalk/leaderboard.md.
"""

from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D1, D2 = ROOT / "data/flytalk_20260926", ROOT / "data/flytalk_rl_20260926"
PANEL = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
]

DESCRIBE = {  # player id -> (label, what it is); rows added after the first results are marked post hoc
    "random": ("Random", "random measurements, random answer"),
    "nothing": ("Answer all 0", "no measurements"),
    "lookup": ("Threshold rule", "measure all, re-measure ambiguous, label shifts beyond 0.10"),
    "lookup_prior": (
        "Threshold rule, at most two",
        "same measurements, two largest shifts only (post hoc)",
    ),
    "example_rule": (
        "example_agent.py --policy rule",
        "reference client shipped with the benchmark",
    ),
    "qwen32b_zero": ("Qwen3-32B zero-shot", "greedy, thinking off"),
    "qwen32b_grpo_step10": ("Qwen3-32B + GRPO, 10 steps", "multi-turn GRPO, LoRA r16"),
    "qwen32b_grpo_step20": ("Qwen3-32B + GRPO, 20 steps", "multi-turn GRPO, LoRA r16"),
    "qwen32b_grpo_step30": ("Qwen3-32B + GRPO, 30 steps", "multi-turn GRPO, LoRA r16"),
    "qwen32b_zero_brief": (
        "Qwen3-32B zero-shot + brief",
        "brief_informed.txt in the prompt (post hoc)",
    ),
    "qwen32b_grpo_step30_brief": (
        "Qwen3-32B GRPO 30 + brief",
        "brief_informed.txt in the prompt (post hoc)",
    ),
    "claude_sonnet": (
        "Claude Sonnet 5 zero-shot",
        "claude-sonnet-5 via the CLI, one line per turn",
    ),
    "claude_opus": ("Claude Opus 5.5 zero-shot", "claude-opus-5-5 via the CLI, one line per turn"),
    "claude_opus_informed_scored": (
        "Claude Opus 5.5 + brief + scoring rule",
        "brief_informed_scored.txt",
    ),
    "qwen32b_think": ("Qwen3-32B zero-shot, thinking on", "reasoning mode enabled"),
    **{
        f"qwen32b_grpo_brief_step{n}": (
            f"Qwen3-32B + GRPO with brief, {n} steps",
            "brief_informed.txt in every training and test prompt",
        )
        for n in (10, 20, 30)
    },
    "claude_sonnet_scored": ("Claude Sonnet 5 + scoring rule", "brief_scored_only.txt (post hoc)"),
    "claude_sonnet_informed": ("Claude Sonnet 5 + brief", "brief_informed.txt (post hoc)"),
    "claude_sonnet_informed_scored": (
        "Claude Sonnet 5 + brief + scoring rule",
        "brief_informed_scored.txt (post hoc)",
    ),
    "bayes": ("Bayes player, marginal labels", "knows the response table of all 799 histories"),
    "bayes_balanced": (
        "Bayes player, balanced decision",
        "same measurements, answer maximises expected score (post hoc)",
    ),
}


def load(extra=()):
    players = {}
    if (D1 / "baselines.json").exists():
        for r in json.load(open(D1 / "baselines.json")):
            players.setdefault(r["player"], []).append(r)
    named = {
        "claude_sonnet*.json": None,
        "claude_opus*.json": None,
        "lookup_prior.json": "lookup_prior",
        "bayes_player.json": "bayes",
        "bayes_balanced.json": "bayes_balanced",
        "example_rule.json": "example_rule",
    }
    for pattern, name in named.items():
        for p in sorted(D1.glob(pattern)):
            players[name or p.stem] = json.load(open(p))
    for p in sorted(D2.glob("rows_*.json")) if D2.exists() else []:
        for r in json.load(open(p)):
            players.setdefault(r.get("player", p.stem), []).append(r)
    for path, name in extra:
        players[name] = json.load(open(path))
    return {
        k: {
            r["seed"]: r
            for r in v
            if r.get("task", "diagnose") == "diagnose" and 0 <= r["seed"] < 60
        }
        for k, v in players.items()
    }


def generalised_pairs():
    """(seed, odour) of untrained odours whose expected shift under the fly's true history exceeds 0.10."""
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "scripts/flytalk"))
    import bayes_player as B

    P = B.Player(D1 / "bayes_table.npz")
    key = lambda h: tuple(sorted((o, r, int(n)) for o, r, n in h))
    idx = {key(h): i for i, h in enumerate(P.hists)}
    from flypet.flytalk import sample_history

    gen, unshifted = set(), set()
    for seed in range(60):
        hist = sample_history(PANEL, np.random.default_rng(seed))
        hi = idx[key(hist)]
        trained = {o for o, _, _ in hist}
        for j, o in enumerate(PANEL):
            if o not in trained:
                (gen if abs(P.mu[hi, j] - P.mu[idx[()], j]) > 0.10 else unshifted).add((seed, o))
    return gen, unshifted


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)])
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def row(name, R, gen, unshifted):
    lab = lambda r, o: r["pred"].get(o, "0") if isinstance(r.get("pred"), dict) else "0"
    m, lo, hi = boot([r["score"] for r in R.values()])
    twin = [
        r
        for r in R.values()
        if (r["truth"]["ethyl acetate"] == "0") != (r["truth"]["methyl acetate"] == "0")
    ]
    twin_err = sum(
        lab(r, "methyl acetate" if r["truth"]["ethyl acetate"] != "0" else "ethyl acetate") != "0"
        for r in twin
    )
    label, what = DESCRIBE.get(name, (name, ""))
    return {
        "player": name,
        "label": label,
        "what": what,
        "flies": len(R),
        "score": round(m, 4),
        "ci_low": round(lo, 4),
        "ci_high": round(hi, 4),
        "all_seven_right": round(float(np.mean([r["exact"] for r in R.values()])), 4),
        "trained_right": round(float(np.nanmean([r["conditioned_right"] for r in R.values()])), 4),
        "untrained_right": round(float(np.mean([r["untouched_right"] for r in R.values()])), 4),
        "measurements": round(float(np.mean([r["n_actions"] for r in R.values()])), 1),
        "more_than_two": sum(sum(lab(r, o) != "0" for o in PANEL) > 2 for r in R.values()),
        "fa_generalised": sum(lab(R[s], o) != "0" for s, o in gen if s in R),
        "fa_unshifted": sum(lab(R[s], o) != "0" for s, o in unshifted if s in R),
        "missed": sum(lab(r, o) == "0" for r in R.values() for o in PANEL if r["truth"][o] != "0"),
        "twin": f"{twin_err}/{len(twin)}",
    }


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="rows JSON from example_agent.py --out",
    )
    ap.add_argument("--name", action="append", default=[], help="name for each --extra file")
    a = ap.parse_args(argv)
    extra = list(zip(a.extra, a.name or [p.stem for p in a.extra]))
    P = load(extra)
    gen, unshifted = generalised_pairs()
    rows = sorted(
        (row(k, v, gen, unshifted) for k, v in P.items() if len(v) == 60), key=lambda r: -r["score"]
    )
    skipped = {k: len(v) for k, v in P.items() if len(v) != 60}
    with open(D1 / "leaderboard.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = ROOT / "output/flytalk/leaderboard.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FlyTalk diagnose leaderboard",
        "",
        "60 test flies (seeds 0-59). Score: mean of trained odours labelled with the right sign and untrained odours "
        "labelled 0; 95% bootstrap interval over flies. False alarms are split into untrained odours whose valence "
        f"moved by more than 0.10 through generalisation ({len(gen)} odour-fly pairs) and the rest ({len(unshifted)}). "
        "Generated by `scripts/flytalk/leaderboard.py`.",
        "",
        "| Player | Score [95% CI] | All 7 right | Trained right | Untrained right | Measurements | >2 named | "
        f"False alarms, generalised (of {len(gen)}) | False alarms, unshifted (of {len(unshifted)}) | Trained missed | Ester twin |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['score']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}] | {r['all_seven_right']:.2f} | "
            f"{r['trained_right']:.2f} | {r['untrained_right']:.2f} | {r['measurements']} | {r['more_than_two']} | "
            f"{r['fa_generalised']} | {r['fa_unshifted']} | {r['missed']} | {r['twin']} |"
        )
    lines += ["", "Player notes:", ""] + [
        f"- **{r['label']}**: {r['what']}" for r in rows if r["what"]
    ]
    out.write_text("\n".join(lines) + "\n")
    for r in rows:
        print(f"{r['score']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]  {r['label']}")
    if skipped:
        print("skipped (not 60 flies):", skipped)


if __name__ == "__main__":
    main()
