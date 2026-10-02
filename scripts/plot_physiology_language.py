"""Plots from a completed pilot; no smoothed or fabricated measurements."""

import argparse, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(a):
    root = Path(a.results)
    r = json.loads((root / "report.json").read_text())
    if not r.get("complete"):
        raise ValueError("Complete pilot required")
    colors = ["#2378b5", "#d4782d"]
    names = ["trainable_core", "frozen_core"]
    labels = ["Trainable core", "Frozen core"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for name, label, color in zip(names, labels, colors):
        arm = r["arms"][name]
        curve = arm["validation"]
        xx = [p["step"] for p in curve]
        axes[0].plot(
            xx,
            [p["metrics"]["all"]["accuracy"] for p in curve],
            marker="o",
            ms=3,
            color=color,
            label=label,
        )
        axes[1].plot(
            xx,
            [p["first_answer_token_nll"] for p in curve],
            marker="o",
            ms=3,
            color=color,
            label=label,
        )
    for ax in axes[:2]:
        warmup = r["config"].get("decoder_warmup", 0)
        if warmup:
            ax.axvspan(0, warmup, color="grey", alpha=0.12, label="Decoder warm-up")
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=0.15)
    axes[0].axhline(0.25, color="grey", ls=":", lw=1, label="Chance")
    axes[0].axhline(0.625, color="grey", ls="--", lw=1, label="Last-location heuristic")
    axes[0].set(title="Validation accuracy (48 examples)", ylabel="Fraction correct", ylim=(0, 1))
    axes[0].legend(fontsize=9)
    axes[1].axhline(np.log(4), color="grey", ls=":", lw=1)
    axes[1].set(title="Validation answer-token NLL", ylabel="Nats")
    x = np.arange(3)
    width = 0.36
    for i, (name, label, color) in enumerate(zip(names, labels, colors)):
        arm = r["arms"][name]
        v = [
            arm["test"]["metrics"]["all"]["accuracy"],
            arm["test_zero_state"]["metrics"]["all"]["accuracy"],
            arm["test_state_swap"]["all"]["accuracy"],
        ]
        bars = axes[2].bar(x + (i - 0.5) * width, v, width, color=color, label=label)
        axes[2].bar_label(bars, labels=[f"{z:.1%}" for z in v], fontsize=9, padding=3)
    axes[2].set_xticks(x, ["Normal", "Zero state", "State swap"])
    axes[2].set(ylim=(0, 1), ylabel="Fraction correct", title="Test (288 examples / 72 groups)")
    axes[2].axhline(0.25, color="grey", ls=":", lw=1)
    axes[2].axhline(0.625, color="grey", ls="--", lw=1)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Physiological connectome → frozen Qwen3-4B: one-initialization pilot", fontsize=14
    )
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
