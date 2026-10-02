"""Publication figures from recorded cases; never initializes a simulator."""

from pathlib import Path
import argparse, json, itertools
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
COL = {
    "blue": "#326B91",
    "teal": "#218477",
    "orange": "#CC7440",
    "purple": "#8264A4",
    "gray": "#78838B",
}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
    }
)


def save(fig, out, name):
    fig.savefig(out / (name + ".pdf"), facecolor="white")
    fig.savefig(out / (name + ".png"), dpi=190, facecolor="white")
    plt.close(fig)


def architecture(out):
    fig = plt.figure(figsize=(6.8, 3.0))
    ax = fig.add_axes([0.01, 0.40, 0.98, 0.56])
    ax.set(xlim=(-0.02, 1.02), ylim=(0, 1))
    ax.axis("off")
    boxes = [
        (0.00, 0.57, 0.18, 0.34, "Language request\nand chemical name"),
        (0.235, 0.57, 0.19, 0.34, "Planner / recipe\n+ sensory encoding"),
        (0.48, 0.57, 0.23, 0.34, "FlyWire LIF\n+ KC–MBON memory"),
        (0.77, 0.57, 0.22, 0.34, "Brain readouts\n+ MaleCNS VNC"),
    ]
    for x, y, w, h, t in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.009,rounding_size=0.025",
                facecolor="#F3F6F8",
                edgecolor=COL["blue"],
                linewidth=0.8,
            )
        )
        ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=8)
    for i in range(3):
        x, y, w, h, _ = boxes[i]
        nx, ny, nw, nh, _ = boxes[i + 1]
        ax.add_patch(
            FancyArrowPatch(
                (x + w + 0.005, y + h / 2),
                (nx - 0.01, ny + nh / 2),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.8,
                color=COL["gray"],
            )
        )
    ax.text(
        0.335,
        0.40,
        "DoOR profiles; other channels\nwith parameter provenance",
        ha="center",
        va="center",
        fontsize=7.5,
        color="#43515B",
    )
    ax.text(
        0.595,
        0.16,
        "Neural rates\n↓\nsoft-token Qwen caption",
        ha="center",
        va="center",
        fontsize=8,
        color=COL["teal"],
    )
    ax.annotate(
        "",
        xy=(0.595, 0.32),
        xytext=(0.595, 0.56),
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": COL["teal"]},
    )
    ax.text(
        0.875,
        0.16,
        "Numerical summaries\n↓\nlanguage explanation",
        ha="center",
        va="center",
        fontsize=8,
        color=COL["blue"],
    )
    ax.annotate(
        "",
        xy=(0.875, 0.32),
        xytext=(0.875, 0.56),
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": COL["blue"]},
    )
    tabax = fig.add_axes([0.03, 0.015, 0.95, 0.31])
    tabax.axis("off")
    table = tabax.table(
        cellText=[
            ["Planner", "Request + catalogue", "Stimulus recipe"],
            ["Narrator", "Readouts + recipe + history + reader caption", "Explanation"],
            ["Qwen reader", "Neural rates only", "State caption"],
        ],
        colLabels=["Language component", "Available information", "Output"],
        cellLoc="left",
        colLoc="left",
        colWidths=[0.24, 0.45, 0.31],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1, 1.24)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCD4DA")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#E9EEF2")
            cell.set_text_props(weight="bold")
    save(fig, out, "fig1_system")


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 0.0


def figures(run, out):
    records = [json.loads(l) for l in (run / "records.jsonl").read_text().splitlines() if l]
    protocol = json.loads((run / "protocol.json").read_text())
    validation = json.loads((run / "validation.json").read_text())
    odors = protocol["odors"]
    seeds = protocol["seeds"]
    ar = [r for r in records if r["case"] == "A"]
    patterns = {}
    fractions = {}
    for r in ar:
        k = r["populations"]["KC"]
        patterns[r["compound"], r["seed"]] = {
            i for i, v in zip(k["model_indices"], k["rates_hz"]) if v > 0
        }
        fractions[r["compound"], r["seed"]] = k["fraction_active"]
    J = np.zeros((len(odors), len(odors)))
    for i, a in enumerate(odors):
        for j, b in enumerate(odors):
            pairs = (
                list(itertools.combinations(seeds, 2))
                if i == j
                else list(itertools.product(seeds, seeds))
            )
            J[i, j] = np.mean([jaccard(patterns[a, s], patterns[b, t]) for s, t in pairs])
    C = np.array(validation["input_cosine_similarity"], dtype=float)
    ii, jj = np.triu_indices(len(odors), 1)
    valid = np.isfinite(C[ii, jj])
    rho = float(np.corrcoef(C[ii, jj][valid], J[ii, jj][valid])[0, 1])
    fig, axs = plt.subplots(
        1, 2, figsize=(6.8, 2.95), gridspec_kw={"width_ratios": [1.12, 1]}, layout="constrained"
    )
    short = [
        "ethyl\nacetate",
        "methyl\nacetate",
        "1-hexanol",
        "Z3-hexenol",
        "acetic\nacid",
        "pyrrolidine",
    ]
    im = axs[0].imshow(J, vmin=0, vmax=1, cmap="Blues")
    axs[0].set_xticks(range(6), short, rotation=45, ha="right")
    axs[0].set_yticks(range(6), short)
    axs[0].set_title("a  Kenyon-cell code overlap", loc="left", weight="bold")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.04, pad=0.025)
    cb.set_label("Mean Jaccard overlap")
    axs[1].scatter(C[ii, jj], J[ii, jj], s=26, color=COL["blue"], edgecolor="white", linewidth=0.4)
    axs[1].set(
        xlabel="Receptor-profile cosine similarity",
        ylabel="Mean KC-code Jaccard overlap",
        xlim=(-0.04, 1.04),
        ylim=(-0.015, max(0.2, float(J[ii, jj].max()) * 1.2)),
    )
    axs[1].set_title("b  Input and internal geometry", loc="left", weight="bold")
    axs[1].text(
        0.03,
        0.96,
        f"15 chemical pairs; r = {rho:.2f}",
        transform=axs[1].transAxes,
        va="top",
        fontsize=8,
    )
    save(fig, out, "fig2_olfaction")
    bnames = [x["name"] for x in protocol["sensory"]]
    readouts = ["proboscis_extension", "escape_takeoff", "antennal_grooming"]
    H = np.array(
        [
            [
                np.mean(
                    [
                        r["summary"]["behaviours"][k]["max_rate_hz"]
                        for r in records
                        if r["case"] == "B" and r["condition"] == c
                    ]
                )
                for k in readouts
            ]
            for c in bnames
        ]
    )
    fig, axs = plt.subplots(
        1, 2, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.2, 1]}, layout="constrained"
    )
    im = axs[0].imshow(H, aspect="auto", cmap="YlGnBu", vmin=0)
    axs[0].set_xticks(range(3), ["Proboscis", "Escape", "Antennal\ngrooming"])
    axs[0].set_yticks(
        range(len(bnames)),
        ["Zero input", "Sugar", "Bitter", "Sugar + bitter", "Antenna touch", "Looming"],
    )
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            axs[0].text(
                j,
                i,
                f"{H[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if H[i, j] > H.max() * 0.65 else "#182A39",
            )
    axs[0].set_title("a  Brain output channels", loc="left", weight="bold")
    cb = fig.colorbar(im, ax=axs[0], fraction=0.04, pad=0.025)
    cb.set_label("Mean peak rate (Hz)")
    gf = [r for r in records if r["case"] == "GF"]
    groups = [
        [r["vnc"]["target"]["max_rate_hz"] for r in gf if r["condition"] == "GF"],
        [r["vnc"]["target"]["max_rate_hz"] for r in gf if r["condition"] != "GF"],
    ]
    for j, ys in enumerate(groups):
        jitter = np.linspace(-0.1, 0.1, len(ys))
        axs[1].scatter(
            j + jitter, ys, s=24, color=COL["teal"] if j == 0 else COL["gray"], alpha=0.85
        )
        axs[1].hlines(np.median(ys), j - 0.18, j + 0.18, color="#172C37", lw=1.2)
    axs[1].set(
        xticks=[0, 1],
        xticklabels=["Giant fibre\n3 seeds", "5 other DN pairs\n3 seeds each"],
        ylabel="TTMn firing rate (Hz)",
        xlim=(-0.5, 1.5),
        ylim=(-3, max(max(x) for x in groups) * 1.16),
    )
    axs[1].set_title("b  Descending-to-motor bridge", loc="left", weight="bold")
    save(fig, out, "fig3_sensorimotor")
    mr = [r for r in records if r["case"] == "C" and r["phase"] in ("probe", "post", "reload")]
    comps = protocol["memory_compounds"]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 2.8), sharey=True, layout="constrained")
    mem = {}
    for panel, comp in enumerate(comps):
        conditions = [
            "naive",
            "A" if panel == 0 else "B",
            "B" if panel == 0 else "A",
            "unpaired",
            "A_reloaded",
        ]
        mem[comp] = {}
        for x, h in enumerate(conditions):
            ys = [r["valence"]["score"] for r in mr if r["compound"] == comp and r["history"] == h]
            mem[comp][h] = ys
            color = ["#304653", COL["teal"], COL["orange"], COL["gray"], COL["purple"]][x]
            axs[panel].scatter(
                x + np.linspace(-0.08, 0.08, len(ys)), ys, s=26, color=color, zorder=3
            )
            axs[panel].hlines(np.mean(ys), x - 0.21, x + 0.21, color=color, lw=1.6)
        axs[panel].axhline(0, color="#BAC3CA", lw=0.7, zorder=0)
        axs[panel].set(
            xticks=range(5),
            xticklabels=["Naive", "Rewarded", "Punished", "Unpaired", "Reload A"],
            ylim=(-1.08, 1.08),
        )
        axs[panel].tick_params(axis="x", rotation=30)
        axs[panel].set_title(("a  " if panel == 0 else "b  ") + comp, loc="left", weight="bold")
    axs[0].set_ylabel("Model MBON valence score")
    save(fig, out, "fig4_memory")
    summary = json.loads((run / "summary.json").read_text())
    reloads = [r for r in summary["memory_comparisons"] if r["history"] == "A_reloaded"]
    obs = {
        "protocol_id": protocol["protocol_id"],
        "seeds": seeds,
        "n_records": len(records),
        "odor_compounds": odors,
        "KC_fraction_active": {c: [fractions[c, s] for s in seeds] for c in odors},
        "KC_jaccard_mean": J.tolist(),
        "input_cosine": C.tolist(),
        "pairwise_input_KC_pearson_descriptive": rho,
        "sensory_conditions": bnames,
        "brain_readout_names": readouts,
        "brain_mean_peak_rates_hz": H.tolist(),
        "GF_TTMn_Hz": groups[0],
        "random_TTMn_Hz": groups[1],
        "memory_valence": mem,
        "all_reload_counts_equal": all(r["reload_equal_counts"] for r in reloads),
        "reload_probe_count": len(reloads),
    }
    (out / "observations.json").write_text(
        json.dumps(obs, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in obs.items()
                if k not in ("KC_jaccard_mean", "input_cosine", "memory_valence")
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "paper/figures")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    architecture(a.out)
    if a.run:
        figures(a.run, a.out)
