#!/usr/bin/env python3
"""Report completed language forecasts alongside fixed CPU controls."""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "mean": float(np.mean(values)),
        "std_over_initializations": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "initializations": len(values),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    reports = [
        json.loads(p.read_text()) for p in sorted(a.reader.glob("seed_*/results/report.json"))
    ]
    if len(reports) != 3 or sorted(r["config"]["seed"] for r in reports) != [0, 1, 2]:
        raise ValueError("All three prespecified forecast runs are required")
    if not all(r.get("complete") and r.get("experiment_complete") for r in reports):
        raise ValueError("Incomplete or smoke result cannot enter the scientific summary")
    cpu = json.loads((a.analysis / "report.json").read_text())
    grouped, flat = defaultdict(list), []
    for r in reports:
        for mode, partitions in r["evaluation"].items():
            for split, metric in partitions.items():
                pair = metric["matched_assignment_contrast"]
                row = {
                    "seed": r["config"]["seed"],
                    "mode": mode,
                    "split": split,
                    "n": metric["n"],
                    "parse_rate": metric["parse_rate"],
                    "delta_mae_valid": metric["mae_valid"]["delta_valence"],
                    "contrast_mae_valid": pair["mae_valid"],
                    "contrast_coverage": pair["coverage"],
                    "no_change_mae": metric["no_change_mae"],
                    "nll": metric["nll"],
                }
                grouped[(mode, split)].append(row)
                flat.append(row)
    result = {
        "complete": True,
        "task": "forecast_delta_valence",
        "seeds": [0, 1, 2],
        "models": {},
        "target_range": [-2, 2],
        "target_response_in_input": False,
        "fixed_diagnostics": ["methyl acetate", "1-hexanol"],
        "scope": "Two held-out conditioning families with three familiar query chemicals; one excluded chemical. Errors conditional on valid JSON, accompanied by coverage. SD is across training initializations, not a confidence interval.",
        "oracle_note": "Oracle text supplies the true target. It checks output formatting, not prediction.",
    }
    for (mode, split), rows in grouped.items():
        result["models"].setdefault(mode, {})[split] = {
            key: stats([r[key] for r in rows])
            for key in (
                "parse_rate",
                "delta_mae_valid",
                "contrast_mae_valid",
                "contrast_coverage",
                "nll",
            )
        }
        result["models"][mode][split].update(
            {"n_per_seed": rows[0]["n"], "no_change_mae": rows[0]["no_change_mae"]}
        )
    (a.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (a.out / "metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), constrained_layout=True)
    choices = [
        ("neural", "Diagnostic probes → LLM"),
        ("history", "Full history → LLM"),
        ("null", "Constant prefix → LLM"),
    ]
    for ax, split, title in zip(
        axes,
        ["test", "query_chemical_test_from_test"],
        ["New conditioning pairs", "Excluded chemical: acetic acid"],
    ):
        labels, means, sds, coverage, points = [], [], [], [], []
        for mode, label in choices:
            v = result["models"][mode][split]
            error = v["delta_mae_valid"]
            labels.append(label)
            means.append(error["mean"] if error else np.nan)
            sds.append((error["std_over_initializations"] or 0) if error else 0)
            coverage.append(v["parse_rate"]["mean"])
            points.append(
                [
                    r["evaluation"][mode][split]["mae_valid"]["delta_valence"]
                    for r in reports
                    if r["evaluation"][mode][split]["mae_valid"]["delta_valence"] is not None
                ]
            )
        cpu_split = "test" if split == "test" else "query_chemical_test__test"
        for prefix, label in [
            ("diagnostic_mbon_and_query", "Diagnostic probes → MLP"),
            ("full_history", "Full history → MLP"),
        ]:
            values = [
                cpu["prospective_delta"][f"{prefix}__mlp_seed{s}"]["metrics"][cpu_split]["mae"][
                    "valence"
                ]
                for s in (0, 1, 2)
            ]
            labels.append(label)
            means.append(np.mean(values))
            sds.append(np.std(values, ddof=1))
            points.append(values)
        ax.barh(
            range(len(labels)),
            means,
            xerr=sds,
            capsize=3,
            color=["#0072B2", "#D55E00", "#999999", "#009E73", "#CC79A7"],
        )
        for i, values in enumerate(points):
            ax.scatter(
                values,
                i + np.linspace(-0.09, 0.09, len(values)),
                s=23,
                color="#222222",
                edgecolor="white",
                linewidth=0.5,
                zorder=4,
            )
        ax.axvline(
            cpu["prospective_delta"]["no_change"][cpu_split]["mae"]["valence"],
            color="#333333",
            ls="--",
            lw=1.2,
        )
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.set_title(title, pad=12)
        ax.set_xlabel("MAE of predicted change in valence")
        ax.text(
            0,
            -0.22,
            f"Valid JSON: neural {coverage[0]:.1%}, history {coverage[1]:.1%}\n"
            f"constant {coverage[2]:.1%}. Dashed: predict no change.",
            transform=ax.transAxes,
            fontsize=8,
        )
        for i, value in enumerate(means):
            if not np.isfinite(value):
                ax.text(
                    0.02, i, "No valid outputs", transform=ax.get_yaxis_transform(), va="center"
                )
    fig.suptitle("Forecasting a response that was not supplied to the model", fontsize=14)
    fig.supxlabel(
        "Dots: individual initializations; error bars: their SD. LLM is frozen Qwen3-0.6B.",
        fontsize=8,
    )
    fig.savefig(a.out / "language_forecast.png", dpi=180)
    fig.savefig(a.out / "language_forecast.pdf")
    print(json.dumps({"status": "complete", "out": str(a.out)}))


if __name__ == "__main__":
    main()
