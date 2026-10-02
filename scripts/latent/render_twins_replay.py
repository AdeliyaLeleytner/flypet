#!/usr/bin/env python3
"""Create a labeled animated plot of an exported, actually executed twins run."""

import argparse, json, textwrap
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--first", type=Path, required=True)
    p.add_argument("--second", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--reader-label", default="neural reader")
    a = p.parse_args()
    records = [json.loads(path.read_text()) for path in (a.first, a.second)]
    if records[0]["seed"] != records[1]["seed"]:
        raise ValueError("Twin seeds differ")
    events = [[e for e in r["events"] if e["mode"] == "interact"] for r in records]
    if any(len(e) != 6 for e in events):
        raise ValueError("Expected the six-window twins protocol")
    if events[0][0]["replay"]["channel_rates_hz"] != events[1][0]["replay"]["channel_rates_hz"]:
        raise ValueError("Baseline drives differ")
    if events[0][-1]["replay"]["channel_rates_hz"] != events[1][-1]["replay"]["channel_rates_hz"]:
        raise ValueError("Final drives differ")
    replies = [next(e["reply"] for e in reversed(r["events"]) if e.get("reply")) for r in records]
    values = [[e["telemetry"]["valence"] for e in stream] for stream in events]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6))
    fig.patch.set_facecolor("#f6f5ee")
    fig.subplots_adjust(top=0.73, bottom=0.30, wspace=0.22)
    fig.suptitle("Same starting brain. Different experience.", fontsize=20, y=0.97, color="#1c332c")
    fig.text(
        0.5,
        0.89,
        f"Recorded simulation · seed {records[0]['seed']} · {a.reader_label}",
        ha="center",
        color="#65766a",
    )
    phase = fig.text(0.5, 0.81, "", ha="center", fontsize=13, color="#1c332c")
    colors = ["#34745b", "#bf7044"]
    lines = []
    points = []
    labels = []
    captions = []
    stages = [
        "Baseline",
        "Experience 1",
        "Experience 2",
        "Experience 3",
        "Rest",
        "Same scent probe",
    ]
    for i, ax in enumerate(axes):
        ax.set_facecolor("#fffef9")
        ax.set_xlim(-0.15, 5.2)
        ax.set_ylim(-1, 1)
        ax.set_xticks([0, 1, 2, 3, 4, 5], ["Start", "1", "2", "3", "Rest", "Probe"])
        ax.set_ylabel("Measured MBON balance" if i == 0 else "")
        ax.axhline(0, color="#abb6ac", lw=0.8)
        ax.grid(alpha=0.12)
        ax.set_title(
            "Fly A · reward history" if i == 0 else "Fly B · punishment history",
            color=colors[i],
            fontsize=12,
        )
        (line,) = ax.plot([], [], color=colors[i], lw=2.5)
        (point,) = ax.plot([], [], marker="o", color=colors[i], ms=7)
        lines.append(line)
        points.append(point)
        labels.append(
            ax.text(
                0.98,
                0.95,
                "",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=18,
                color=colors[i],
            )
        )
        fig.text(0.135 + i * 0.422, 0.225, "GENERATED REPLY", fontsize=8, color="#65766a")
        captions.append(
            fig.text(0.135 + i * 0.422, 0.19, "", fontsize=9.5, color="#1c332c", va="top")
        )
    fig.text(
        0.5,
        0.045,
        "Recorded data, not a live simulation. Language is generated from neural tokens; exposure histories are not supplied to the reader.",
        ha="center",
        fontsize=8,
        color="#65766a",
    )

    def animate(frame):
        index = min(frame // 5, 5)
        phase.set_text(stages[index])
        for i in range(2):
            lines[i].set_data(np.arange(index + 1), values[i][: index + 1])
            points[i].set_data([index], [values[i][index]])
            labels[i].set_text(f"{values[i][index]:+.2f}")
            captions[i].set_text(textwrap.fill(replies[i], width=40) if index == 5 else "")
        return [phase, *lines, *points, *labels, *captions]

    a.out.parent.mkdir(parents=True, exist_ok=True)
    animation = FuncAnimation(fig, animate, frames=50, interval=200, blit=False)
    animation.save(a.out, writer=PillowWriter(fps=5), dpi=100)
    animate(49)
    fig.savefig(a.out.with_suffix(".png"), dpi=150)


if __name__ == "__main__":
    main()
