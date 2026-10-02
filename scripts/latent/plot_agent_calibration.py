#!/usr/bin/env python3
"""Plot the recorded training-only dose calibration (requires matplotlib)."""

import argparse, json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    rows = json.loads(a.report.read_text())["rows"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    colors = ["#34745b", "#bf7044"]
    for task in (0, 1):
        group = [x for x in rows if x["task"] == task]
        for j, channel in enumerate(["PAM", "PPL1"]):
            index = j + 1
            other = 2 - j
            doses = sorted(
                [x for x in group if x["action"][0] == 1 and x["action"][other] == 0],
                key=lambda x: x["action"][index],
            )
            axes[0, task].errorbar(
                [x["action"][index] for x in doses],
                [x["pooled_mean"] for x in doses],
                yerr=[x["pooled_std"] for x in doses],
                fmt="o-",
                lw=1.5,
                ms=4,
                capsize=3,
                color=colors[j],
                label=channel,
            )
            strengths = sorted(
                [x for x in group if x["action"][index] == 12 and x["action"][other] == 0],
                key=lambda x: x["action"][0],
            )
            axes[1, task].errorbar(
                [x["action"][0] for x in strengths],
                [x["pooled_mean"] for x in strengths],
                yerr=[x["pooled_std"] for x in strengths],
                fmt="o-",
                lw=1.5,
                ms=4,
                capsize=3,
                color=colors[j],
                label=channel + " at 12 Hz",
            )
        axes[0, task].set(
            title=f"Training recipe {task} · dopamine dose",
            xlabel="External input rate (Hz)",
            ylabel="Pooled MBON balance",
        )
        axes[1, task].set(
            title=f"Training recipe {task} · sensory strength",
            xlabel="Odor exposure multiplier",
            ylabel="Pooled MBON balance",
        )
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(-0.75, 0.65)
    fig.suptitle("Graded control in the fixed neural model", fontsize=16)
    fig.supxlabel(
        "Three seeds per condition; bars show SD. Rates pooled over 750 ms. Two training recipes, exploratory calibration.",
        fontsize=9,
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=160)


if __name__ == "__main__":
    main()
