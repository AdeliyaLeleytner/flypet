"""Readable appendix figure, with raw validation measurements only."""

import argparse, json, math
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--report", required=True)
p.add_argument("--out", required=True)
a = p.parse_args()
r = json.loads(Path(a.report).read_text())
assert r["complete"]
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "legend.fontsize": 7})
f, axs = plt.subplots(1, 2, figsize=(6.5, 2.75), layout="constrained")
for name, label, color in [
    ("trainable_core", "Trainable core", "#2378b5"),
    ("frozen_core", "Frozen core", "#d4782d"),
]:
    v = r["arms"][name]["validation"]
    xx = [x["step"] for x in v]
    axs[0].plot(
        xx,
        [x["metrics"]["all"]["accuracy"] for x in v],
        "-o",
        ms=2.5,
        lw=1,
        color=color,
        label=label,
    )
    axs[1].plot(xx, [x["first_answer_token_nll"] for x in v], "-o", ms=2.5, lw=1, color=color)
for ax in axs:
    ax.axvspan(0, 100, color="grey", alpha=0.12)
    ax.set_xlabel("Optimizer updates")
    ax.grid(alpha=0.12)
    ax.spines[["top", "right"]].set_visible(False)
axs[0].axhline(0.25, color="grey", ls=":", lw=1, label="Chance")
axs[0].axhline(0.625, color="grey", ls="--", lw=1, label="Last-location heuristic")
axs[0].set(title="a  Validation accuracy", ylim=(0, 1), ylabel="Fraction correct")
axs[0].legend(loc="upper right", frameon=False)
axs[1].axhline(math.log(4), color="grey", ls=":", lw=1)
axs[1].set(title="b  Answer-token loss", ylabel="NLL (nats)")
f.savefig(a.out, dpi=240)
