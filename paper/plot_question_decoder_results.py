"""Render an anonymous, source-verified figure for the completed six-fit study."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from paper.generate_question_decoder_results import ARMS, SEEDS, digest, load_verified, seed_values
from scripts.analyze_question_decoder import require

COLORS = {"plain": "#2878A5", "conditioned": "#C47727"}
LABELS = {"plain": "State only", "conditioned": "State + question"}
MARKERS = ("o", "s", "^")


def plot(analysis_path: Path, paper_dir: Path) -> dict:
    analysis, reports, receipt = load_verified(analysis_path)
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 4.7), layout="constrained")
    loss_ax, paired_ax, position_ax, control_ax = axes.ravel()
    for arm in ARMS:
        curves = [reports[seed]["arms"][arm]["validation"] for seed in SEEDS]
        steps = [[entry["step"] for entry in curve] for curve in curves]
        require(
            all(steps[0] == values for values in steps[1:]),
            "validation schedules differ across seeds",
        )
        values = np.array([[entry["nll"] for entry in curve] for curve in curves])
        require(np.isfinite(values).all(), "nonfinite validation NLL")
        for value in values:
            loss_ax.plot(steps[0], value, color=COLORS[arm], alpha=0.25, linewidth=0.7)
        loss_ax.plot(steps[0], values.mean(0), color=COLORS[arm], linewidth=1.6, label=LABELS[arm])
    loss_ax.axhline(math.log(4), color=".45", linestyle=":", linewidth=0.9)
    loss_ax.set(
        title="a  Validation answer-token loss", xlabel="Optimizer updates", ylabel="NLL (nats)"
    )
    loss_ax.legend(frameon=False, loc="upper right")
    for i, seed in enumerate(SEEDS):
        values = [analysis["per_seed"][str(seed)][arm]["test"]["accuracy"] * 100 for arm in ARMS]
        offset = (i - 1) * 0.055
        paired_ax.plot(np.array([0, 1]) + offset, values, color=".65", linewidth=0.7, zorder=1)
        for x, arm, value in zip((0, 1), ARMS, values):
            paired_ax.scatter(
                x + offset,
                value,
                color=COLORS[arm],
                marker=MARKERS[i],
                s=27,
                linewidth=0.5,
                edgecolor="white",
                zorder=3,
            )
    for x, arm in enumerate(ARMS):
        mean = seed_values(analysis, arm, "test", "accuracy").mean() * 100
        paired_ax.plot([x - 0.13, x + 0.13], [mean, mean], color="black", linewidth=1.5, zorder=4)
    paired_ax.set(
        title="b  Paired test accuracy",
        ylabel="Correct answers",
        xticks=[0, 1],
        xticklabels=[LABELS[arm] for arm in ARMS],
        xlim=(-0.35, 1.35),
    )
    for x, metric in enumerate(("earlier_object_accuracy", "most_recent_object_accuracy")):
        for j, arm in enumerate(ARMS):
            center = x + (-0.13 if j == 0 else 0.13)
            values = seed_values(analysis, arm, "test", metric) * 100
            position_ax.plot(
                [center - 0.08, center + 0.08], [values.mean()] * 2, color=COLORS[arm], linewidth=2
            )
            for i, value in enumerate(values):
                position_ax.scatter(
                    center + (i - 1) * 0.025,
                    value,
                    s=20,
                    color=COLORS[arm],
                    marker=MARKERS[i],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=3,
                )
    position_ax.set(
        title="c  Which object's location is queried?",
        ylabel="Correct answers",
        xticks=[0, 1],
        xticklabels=["Earlier object", "Latest object"],
        xlim=(-0.4, 1.4),
    )
    for arm in ARMS:
        values = np.array(
            [
                seed_values(analysis, arm, condition, "accuracy") * 100
                for condition in ("test", "mean_state", "swapped_state")
            ]
        )
        offset = -0.06 if arm == "plain" else 0.06
        control_ax.plot(np.arange(3) + offset, values.mean(1), color=COLORS[arm], linewidth=1.3)
        for i in range(len(SEEDS)):
            control_ax.scatter(
                np.arange(3) + offset + (i - 1) * 0.018,
                values[:, i],
                color=COLORS[arm],
                marker=MARKERS[i],
                s=21,
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
    control_ax.set(
        title="d  Replacing the supplied state",
        ylabel="Correct answers",
        xticks=[0, 1, 2],
        xticklabels=["Intact", "Train mean", "Paired swap"],
        xlim=(-0.3, 2.3),
    )
    for ax in (paired_ax, position_ax, control_ax):
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.axhline(25, color=".55", linestyle=":", linewidth=0.8, zorder=0)
    for ax in axes.ravel():
        ax.grid(axis="y", color=".9", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
    output = paper_dir / "figures"
    output.mkdir(parents=True, exist_ok=True)
    pdf = output / "question_decoder_results.pdf"
    png = output / "question_decoder_results.png"
    fig.savefig(
        pdf,
        metadata={
            "Author": "",
            "Title": "Decoding fixed physiological states",
            "Subject": "Exploratory matched question-access comparison",
            "Creator": "Scientific results rendering",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(png, dpi=240, metadata={"Software": "Scientific results rendering"})
    plt.close(fig)
    receipt["plot_script_sha256"] = digest(Path(__file__))
    receipt["verification_script_sha256"] = digest(
        Path(__file__).with_name("generate_question_decoder_results.py")
    )
    receipt["outputs"] = {"figures/" + path.name: digest(path) for path in (pdf, png)}
    path = paper_dir / "analysis/question_decoder_plot_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--paper-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    receipt = plot(args.analysis, args.paper_dir)
    print(
        json.dumps(
            {"outputs": receipt["outputs"], "analysis_sha256": receipt["analysis_sha256"]}, indent=2
        )
    )
