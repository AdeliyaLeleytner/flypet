#!/usr/bin/env python3
"""Figures for the ICLR 2027 submission (paper_iclr27/figures/). Vector PDF at the ICLR text width (5.5 in).

fig2_substrate  KC->DAN ablation, learning-dependent odour dopamine, erase feasibility
fig3_teaching   open-loop teachers on the real fly vs their surrogate prediction; target/untouched trade-off
fig4_diagnose   closed-loop diagnosis players with bootstrap intervals; trained vs untrained accuracy; GRPO curve
"""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper_iclr27/figures"
FIG.mkdir(parents=True, exist_ok=True)
W = 5.5
OK = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "grey": "#8C8C8C",
}
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def panel(ax, s, title):
    ax.set_title(f"$\\bf{{{s}}}$  {title}", loc="left", fontsize=7, pad=4)


def boot(x, n=5000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = np.array([x[rng.integers(len(x), size=len(x))].mean() for _ in range(n)])
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


# ---------------------------------------------------------------- Figure 2
def fig2():
    fig, axs = plt.subplots(
        1, 3, figsize=(W, 1.75), gridspec_kw={"width_ratios": [1.15, 1.1, 1.0], "wspace": 0.55}
    )
    # (a) ethyl acetate valence after each history, intact vs KC->DAN synapses cut
    ea = {
        "untrained": ([0.184, 0.192, 0.197], [0.184, 0.192, 0.197]),
        "sugar\npaired": ([0.654, 0.701, 0.663], [0.784, 0.808, 0.798]),
        "shock\npaired": ([-0.429, -0.453, -0.419], [-0.429, -0.437, -0.430]),
        "unpaired": ([-0.044, -0.002, -0.078], [0.193, 0.186, 0.183]),
    }
    ax = axs[0]
    for i, (k, (a, b)) in enumerate(ea.items()):
        for off, vals, c in ((-0.17, a, OK["black"]), (0.17, b, OK["orange"])):
            ax.bar(
                i + off, np.mean(vals), 0.3, color=c, alpha=0.25 if c == OK["black"] else 0.35, lw=0
            )
            ax.scatter([i + off] * 3, vals, s=5, color=c, zorder=3, lw=0)
    ax.axhline(0.191, color=OK["grey"], lw=0.5, ls=":")
    ax.set_xticks(range(4))
    ax.set_xticklabels(
        [k.replace("\n", " ") for k in ea],
        fontsize=5.8,
        rotation=30,
        ha="right",
        rotation_mode="anchor",
    )
    ax.set_ylabel("ethyl acetate valence")
    ax.set_ylim(-0.6, 0.9)
    ax.scatter([], [], s=5, color=OK["black"], label="intact")
    ax.scatter([], [], s=5, color=OK["orange"], label="KC$\\rightarrow$DAN cut")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.1, fontsize=5.8)
    panel(ax, "a", "self-recruited dopamine")
    # (b) active DANs in an odour-alone trial, before and after learning
    d = json.load(open(ROOT / "data/teach_20260925/dopamine_memory_check.json"))
    odours = ["ethyl acetate", "methyl acetate", "acetic acid", "2-heptanone", "linalool"]
    ax = axs[1]
    for j, (h, c) in enumerate(
        (("naive", OK["grey"]), ("reward", OK["blue"]), ("punish", OK["red"]))
    ):
        ax.bar(
            np.arange(len(odours)) + (j - 1) * 0.26,
            [d[f"{o}|{h}"]["dan_active"] for o in odours],
            0.25,
            color=c,
            lw=0,
            label={"naive": "untrained", "reward": "after 3 sugar", "punish": "after 3 shock"}[h],
        )
    ax.set_xticks(range(len(odours)))
    ax.set_xticklabels(odours, fontsize=5.6, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_ylabel("active DANs, odour alone")
    ax.legend(frameon=False, loc="upper right", handlelength=0.8, borderaxespad=0.1)
    panel(ax, "b", "learning reshapes it")
    # (c) erase feasibility: deviation of the trained odour after m corrective trials
    e = json.load(open(ROOT / "data/flytalk_20260926/erase_feasibility.json"))
    ax = axs[2]
    for kind, c, ms in (
        ("opposite", OK["purple"], range(0, 7)),
        ("expose", OK["green"], range(1, 7)),
    ):
        med, lo, hi = [], [], []
        for m in ms:
            v = [abs(r["dev"]) for r in e if r["strategy"] == f"{kind} x{m}"]
            med.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
        xs = np.array(list(ms))
        ax.fill_between(xs, lo, hi, color=c, alpha=0.15, lw=0)
        ax.plot(
            xs,
            med,
            color=c,
            marker="o",
            ms=2.5,
            lw=0.9,
            label={"opposite": "opposite pairing", "expose": "odour alone"}[kind],
        )
    ax.axhspan(0, 0.15, color=OK["grey"], alpha=0.15, lw=0)
    ax.text(6.2, 0.07, "tolerance", fontsize=5.8, ha="right", va="center", color="#555555")
    ax.set_xlabel("corrective trials")
    ax.set_ylabel("|valence $-$ untrained|")
    ax.set_ylim(0, 1.3)
    ax.set_xticks(range(7))
    ax.legend(frameon=False, loc="upper left", handlelength=1.2, borderaxespad=0.1, fontsize=5.5)
    panel(ax, "c", "erasure fails")
    fig.savefig(FIG / "fig2_substrate.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 3
def fig3():
    rows = json.load(open(ROOT / "data/teach_20260925/teachers_real.json"))
    base = json.load(open(ROOT / "data/teach_20260925/oracle_baselines.json"))
    naive = base["naive"]
    tg = {r["task"]: r["targets"] for r in base["rows"]}
    show = [
        ("random", "random protocol", OK["grey"]),
        ("textbook", "6 alternating pairings", OK["yellow"]),
        ("oracle", "surrogate beam search", OK["purple"]),
        ("qwen0.6b_grpo_sur", "GRPO on surrogate (Qwen3-0.6B)", OK["orange"]),
        ("oracle_rerank", "surrogate + fly re-ranking", OK["green"]),
        ("claude_sonnet", "Claude Sonnet, zero-shot", OK["blue"]),
        ("textbook_min", "one pairing per target", OK["black"]),
    ]
    fig, axs = plt.subplots(
        1, 2, figsize=(W, 1.9), gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.3}
    )
    ax = axs[0]
    for i, (k, lab, c) in enumerate(show):
        sel = [r for r in rows if r["teacher"] == k]
        m, lo, hi = boot([r["success_true"] for r in sel])
        ax.barh(i, m, 0.62, color=c, alpha=0.85, lw=0)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color="black", lw=0.7, capsize=0)
        ax.scatter(
            np.mean([r["success_sur"] for r in sel]),
            i,
            marker="D",
            s=10,
            facecolor="white",
            edgecolor=OK["red"],
            lw=0.8,
            zorder=4,
        )
    ax.scatter(
        [],
        [],
        marker="D",
        s=10,
        facecolor="white",
        edgecolor=OK["red"],
        lw=0.8,
        label="surrogate prediction",
    )
    ax.set_yticks(range(len(show)))
    ax.set_yticklabels([x[1] for x in show], fontsize=6)
    tg_all = list(tg.values())
    nothing = np.mean([sum(v == 0 for v in t.values()) / len(t) for t in tg_all])
    ax.axvline(nothing, color=OK["grey"], lw=0.7, ls="--", zorder=0)
    ax.set_ylim(-0.6, len(show) + 0.1)
    ax.text(
        nothing - 0.01,
        len(show) - 0.3,
        "do nothing",
        fontsize=5.6,
        ha="right",
        va="center",
        color="#555555",
    )
    ax.set_xlabel("targets met on the whole-brain fly (74 tasks)")
    ax.set_xlim(0, 1.02)
    ax.legend(frameon=False, loc="lower right", fontsize=5.8, borderaxespad=0.1, handletextpad=0.2)
    panel(ax, "a", "open-loop teaching")
    ax = axs[1]
    for i, (k, lab, c) in enumerate(show):
        sel = [r for r in rows if r["teacher"] == k]
        moved = [
            t * (r["valence_true"][o] - naive[o]) >= 0.15
            for r in sel
            for o, t in tg[r["task"]].items()
            if t
        ]
        kept = [
            abs(r["valence_true"][o] - naive[o]) < 0.15
            for r in sel
            for o, t in tg[r["task"]].items()
            if not t
        ]
        ax.scatter(
            np.mean(kept), np.mean(moved), s=18, color=c, edgecolor="black", lw=0.3, zorder=3
        )
        lab3 = {
            "random": ("random", 0.04, 0.0, "left"),
            "textbook": ("6 pairings", -0.03, 0.0, "right"),
            "qwen0.6b_grpo_sur": ("GRPO on\nsurrogate", -0.02, 0.07, "center"),
            "oracle": ("surrogate\nsearch", -0.1, -0.02, "center"),
            "textbook_min": ("one pairing,\nClaude", 0.035, -0.005, "left"),
        }
        if k in lab3:
            txt, dx, dy, ha = lab3[k]
            ax.text(
                np.mean(kept) + dx,
                np.mean(moved) + dy,
                txt,
                fontsize=5.3,
                ha=ha,
                va="center",
                color="#333333",
            )
    ax.set_xlabel("untouched odours kept")
    ax.set_ylabel("targeted odours moved")
    ax.set_xlim(0.25, 0.95)
    ax.set_ylim(0.25, 0.95)
    panel(ax, "b", "side effects decide the score")
    fig.savefig(FIG / "fig3_teaching.pdf")
    plt.close(fig)


# ---------------------------------------------------------------- Figure 4
def load_diagnose():
    D1, D2 = ROOT / "data/flytalk_20260926", ROOT / "data/flytalk_rl_20260926"
    players = {}
    for r in json.load(open(D1 / "baselines.json")):
        if r["task"] == "diagnose":
            players.setdefault(r["player"], []).append(r)
    for tag in ("", "_informed", "_informed_scored"):
        if (D1 / f"claude_sonnet{tag}.json").exists():
            players[f"claude_sonnet{tag}"] = [
                r
                for r in json.load(open(D1 / f"claude_sonnet{tag}.json"))
                if r["task"] == "diagnose"
            ]
    if (D1 / "lookup_prior.json").exists():
        players["lookup_prior"] = json.load(open(D1 / "lookup_prior.json"))
    if (D1 / "bayes_player.json").exists():
        players["bayes"] = json.load(open(D1 / "bayes_player.json"))
    if (D1 / "bayes_balanced.json").exists():
        players["bayes_balanced"] = json.load(open(D1 / "bayes_balanced.json"))
    for p in sorted(D2.glob("rows_*.json")):
        for r in json.load(open(p)):
            if r.get("task", "diagnose") == "diagnose":
                players.setdefault(r["player"], []).append(r)
    return {k: [r for r in v if 0 <= r["seed"] < 60] for k, v in players.items()}


def fig4():
    P = load_diagnose()
    order = [
        ("random", "random"),
        ("nothing", "answer all 0"),
        ("qwen32b_zero", "Qwen3-32B zero-shot"),
        ("lookup", "smell all, threshold"),
    ]
    grpo = sorted(
        [k for k in P if k.startswith("qwen32b_grpo") and not k.endswith("_brief")],
        key=lambda k: int(k.split("step")[-1]),
    )
    if grpo:
        order.append((grpo[-1], f"Qwen3-32B + GRPO ({grpo[-1].split('step')[-1]} steps)"))
    order += [
        ("claude_sonnet", "Claude Sonnet zero-shot"),
        ("lookup_prior", "threshold, at most two"),
        ("claude_sonnet_informed", "Claude Sonnet + brief"),
        ("claude_sonnet_informed_scored", "Claude + brief + scoring"),
        ("bayes_balanced", "Bayes player"),
    ]
    order = [o for o in order if o[0] in P]
    col = {
        "random": OK["grey"],
        "nothing": "#C8C8C8",
        "qwen32b_zero": OK["sky"],
        "lookup": OK["black"],
        "claude_sonnet": OK["blue"],
        "bayes_balanced": OK["purple"],
        "lookup_prior": "#777777",
        "claude_sonnet_informed": "#003f7f",
        "claude_sonnet_informed_scored": "#001f40",
    }
    colour = lambda k: OK["orange"] if "grpo" in k else col.get(k, OK["green"])
    fig, axs = plt.subplots(
        1, 3, figsize=(W, 1.9), gridspec_kw={"width_ratios": [1.35, 1.0, 1.0], "wspace": 0.55}
    )
    ax = axs[0]
    for i, (k, lab) in enumerate(order):
        m, lo, hi = boot([r["score"] for r in P[k]])
        c = colour(k)
        ax.barh(i, m, 0.6, color=c, alpha=0.85, lw=0)
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], color="black", lw=0.7, capsize=0)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([lab for _, lab in order], fontsize=6)
    ax.set_xlim(0, 1)
    ax.set_xlabel("balanced score (60 flies)")
    panel(ax, "a", "closed-loop diagnosis")
    ax = axs[1]
    for i, (k, lab) in enumerate(order):
        tr = np.nanmean([r["conditioned_right"] for r in P[k]])
        ur = np.mean([r["untouched_right"] for r in P[k]])
        ax.scatter(ur, tr, s=18, color=colour(k), edgecolor="black", lw=0.3, zorder=3)
        short = {
            "random": ("random", 0.05, 0.0),
            "qwen32b_zero": ("Qwen", -0.1, -0.03),
            "lookup": ("threshold", -0.03, 0.04),
            "claude_sonnet": ("Claude,\nthreshold $\\leq$2", -0.17, -0.09),
            "bayes_balanced": ("Bayes", 0.0, 0.035),
            "claude_sonnet_informed": ("Claude\n+ brief", 0.0, -0.12),
            "claude_sonnet_informed_scored": ("+ scoring", 0.075, 0.03),
        }
        if k in short or "grpo" in k:
            txt, dx, dy = short.get(k, ("GRPO", -0.08, 0.05))
            ax.text(
                ur + dx,
                tr + dy,
                txt,
                fontsize=5.3,
                ha="center" if k != "random" else "left",
                va="center",
                color="#333333",
            )
    ax.set_xlabel("untrained odours labelled 0")
    ax.set_ylabel("trained odours labelled right")
    ax.set_xlim(0.2, 1.1)
    ax.set_ylim(0.2, 1.02)
    panel(ax, "b", "where the errors are")
    ax = axs[2]
    steps = [(0, "qwen32b_zero")] + [(int(k.split("step")[-1]), k) for k in grpo if "step" in k]
    if len(steps) > 1:
        xs = [s for s, _ in steps]
        ms = [boot([r["score"] for r in P[k]]) for _, k in steps]
        ax.errorbar(
            xs,
            [m[0] for m in ms],
            yerr=[[m[0] - m[1] for m in ms], [m[2] - m[0] for m in ms]],
            color=OK["orange"],
            marker="o",
            ms=2.5,
            lw=0.9,
            elinewidth=0.6,
            capsize=0,
            label="Qwen3-32B + GRPO",
        )
    for k, c, lab in (
        ("lookup", OK["grey"], "smell all, threshold"),
        ("claude_sonnet", OK["blue"], "Claude Sonnet"),
        ("bayes_balanced", OK["purple"], "Bayes player"),
    ):
        if k in P:
            ax.axhline(np.mean([r["score"] for r in P[k]]), color=c, lw=0.7, ls="--", label=lab)
    ax.set_xlabel("GRPO step")
    ax.set_ylabel("test score")
    ax.set_ylim(0.45, 1.0)
    ax.legend(frameon=False, fontsize=5.3, loc="lower right", handlelength=1.4, borderaxespad=0.1)
    panel(ax, "c", "learning to interrogate")
    fig.savefig(FIG / "fig4_diagnose.pdf")
    plt.close(fig)
    order = order + [("bayes", "Bayes, marginal labels")] if "bayes" in P else order
    return {
        k: {
            "n": len(P[k]),
            "score": boot([r["score"] for r in P[k]]),
            "exact": float(np.mean([r["exact"] for r in P[k]])),
            "trained": float(np.nanmean([r["conditioned_right"] for r in P[k]])),
            "untrained": float(np.mean([r["untouched_right"] for r in P[k]])),
            "smells": float(np.mean([r["n_actions"] for r in P[k]])),
        }
        for k, _ in order
    }


if __name__ == "__main__":
    fig2()
    fig3()
    summary = fig4()
    json.dump(summary, open(FIG / "fig4_numbers.json", "w"), indent=1)
    for k, v in summary.items():
        print(
            f"{k:24s} n={v['n']:2d} score {v['score'][0]:.3f} [{v['score'][1]:.3f},{v['score'][2]:.3f}] exact {v['exact']:.2f} trained {v['trained']:.2f} untrained {v['untrained']:.2f} smells {v['smells']:.1f}"
        )
