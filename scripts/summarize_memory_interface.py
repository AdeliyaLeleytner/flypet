#!/usr/bin/env python3
"""Summarize verified memory-interface runs without selecting on test outcomes."""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.memory_reader_data import parse_output


def average(values):
    a = np.asarray(values, dtype=float)
    return {
        "mean": float(a.mean()),
        "std_over_initializations": float(a.std(ddof=1)) if len(a) > 1 else None,
        "initializations": len(a),
    }


def contrasts(predictions, examples):
    grouped = defaultdict(dict)
    for p in predictions:
        if p["mode"] == "neural_state_swap":
            continue
        r = examples[p["example_id"]]
        key = (
            p["mode"],
            p["split"],
            r["split_group_id"],
            r["history_seed"],
            r["pairing_trials"],
            r["probe_compound"],
        )
        grouped[key][r["assignment"]] = (r, parse_output(p["generated"]))
    by_mode = defaultdict(list)
    counts = defaultdict(lambda: [0, 0])
    for key, assignments in grouped.items():
        if set(assignments) != {0, 1}:
            continue
        mode_split = key[:2]
        counts[mode_split][0] += 1
        (a, pa), (b, pb) = assignments[0], assignments[1]
        if pa is None or pb is None:
            continue
        counts[mode_split][1] += 1
        truth = b["target"]["valence"] - a["target"]["valence"]
        pred = pb["valence"] - pa["valence"]
        by_mode[mode_split].append(
            {"true_change": truth, "predicted_change": pred, "absolute_error": abs(truth - pred)}
        )
    result = {}
    for (mode, split), (total, valid) in counts.items():
        rows = by_mode[(mode, split)]
        result.setdefault(mode, {})[split] = {
            "paired_histories": total,
            "valid_pairs": valid,
            "coverage": valid / total,
            "contrast_mae_valid": float(np.mean([r["absolute_error"] for r in rows]))
            if rows
            else None,
            "mean_absolute_truth_contrast": float(np.mean([abs(r["true_change"]) for r in rows]))
            if rows
            else None,
            "mean_absolute_predicted_contrast": float(
                np.mean([abs(r["predicted_change"]) for r in rows])
            )
            if rows
            else None,
        }
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--analysis", type=Path, required=True)
    ap.add_argument("--reader", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    examples = {
        r["example_id"]: r
        for line in (a.data / "examples.jsonl").read_text().splitlines()
        if (r := json.loads(line))
    }
    cpu = json.loads((a.analysis / "report.json").read_text())
    runs = []
    for p in sorted(a.reader.glob("seed_*/results/report.json")):
        report = json.loads(p.read_text())
        if report.get("complete") is not True:
            raise ValueError(f"Incomplete reader run: {p}")
        predictions = json.loads(p.with_name("predictions.json").read_text())
        report["matched_assignment_contrasts"] = contrasts(predictions, examples)
        runs.append(report)
    if len(runs) != 3 or sorted(r["config"]["seed"] for r in runs) != [0, 1, 2]:
        raise ValueError("Expected the three prespecified completed seeds")
    grouped, flat = defaultdict(list), []
    for r in runs:
        for mode, splits in r["evaluation"].items():
            for split, result in splits.items():
                c = r["matched_assignment_contrasts"].get(mode, {}).get(split, {})
                row = {
                    "seed": r["config"]["seed"],
                    "mode": mode,
                    "split": split,
                    "n": result["n"],
                    "parse_rate": result["parse_rate"],
                    "valence_mae_valid": result["mae_valid"]["valence"],
                    "approach_mae_hz_valid": result["mae_valid"]["approach_hz"],
                    "avoid_mae_hz_valid": result["mae_valid"]["avoid_hz"],
                    "contrast_mae_valid": c.get("contrast_mae_valid"),
                    "contrast_coverage": c.get("coverage"),
                    "nll": result["nll"],
                }
                flat.append(row)
                grouped[(mode, split)].append(row)
    summary = {
        "complete": True,
        "seeds": [0, 1, 2],
        "reader": {},
        "scope": "Two held-out familiar-chemical conditioning families; one excluded chemical. Seed error bars describe training initializations, not independent animals or chemical samples.",
        "same_response_note": "Targets are known MBON readouts of the supplied response; learned decoding is not prospective prediction.",
        "prospective_note": "Two fixed diagnostic responses predict a different probe. Target response is excluded; delta is relative to matched naive state.",
        "reader_runs": [
            {
                "seed": r["config"]["seed"],
                "elapsed_seconds": r["elapsed_seconds"],
                "training": r["training"],
                "state_swap": r.get("state_swap"),
                "matched_assignment_contrasts": r["matched_assignment_contrasts"],
            }
            for r in runs
        ],
    }
    for (mode, split), rows in grouped.items():
        summary["reader"].setdefault(mode, {})[split] = {
            k: average([r[k] for r in rows if r[k] is not None])
            if any(r[k] is not None for r in rows)
            else None
            for k in (
                "parse_rate",
                "valence_mae_valid",
                "contrast_mae_valid",
                "contrast_coverage",
                "nll",
            )
        }
        summary["reader"][mode][split]["rows_per_initialization"] = rows[0]["n"]
    with (a.out / "reader_metrics.csv").open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(flat[0]))
        w.writeheader()
        w.writerows(flat)
    (a.out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

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
    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), constrained_layout=True)
    modes = [
        ("neural", "Neural prefix"),
        ("history", "Full-history prefix"),
        ("null", "Constant prefix"),
    ]
    for j, (split, title) in enumerate(
        [
            ("test", "New conditioning pairs"),
            ("query_chemical_test_from_test", "Excluded chemical: acetic acid"),
        ]
    ):
        labels, means, errors = [], [], []
        for mode, label in modes:
            value = summary["reader"][mode][split]
            mae = value["valence_mae_valid"]
            labels.append(label)
            means.append(mae["mean"] if mae is not None else np.nan)
            errors.append((mae["std_over_initializations"] or 0) if mae is not None else 0)
        cpu_split = "test" if split == "test" else "query_chemical_test__test"
        simple = [
            cpu["post_response"][f"mbon_rates__mlp_seed{s}"]["metrics"][cpu_split]["mae"]["valence"]
            for s in (0, 1, 2)
        ]
        labels += ["MBON-only MLP", "Train-mean baseline"]
        means += [
            float(np.mean(simple)),
            cpu["post_response"]["train_mean"][cpu_split]["mae"]["valence"],
        ]
        errors += [float(np.std(simple, ddof=1)), 0]
        axes[j].barh(
            np.arange(len(labels)),
            means,
            xerr=errors,
            color=["#0072B2", "#D55E00", "#999999", "#009E73", "#BBBBBB"],
            capsize=3,
        )
        axes[j].set_yticks(np.arange(len(labels)), labels)
        axes[j].invert_yaxis()
        axes[j].set_title(title, pad=12)
        axes[j].set_xlabel("Current-response valence MAE on valid JSON")
        for i, value in enumerate(means):
            if not np.isfinite(value):
                axes[j].text(
                    0.02,
                    i,
                    "No valid outputs",
                    va="center",
                    transform=axes[j].get_yaxis_transform(),
                    fontsize=9,
                )
        coverage = [summary["reader"][mode][split]["parse_rate"]["mean"] for mode, _ in modes]
        axes[j].text(
            0,
            -0.23,
            f"Valid JSON: neural {coverage[0]:.1%}, history {coverage[1]:.1%}\n"
            f"constant {coverage[2]:.1%}. Error bars: SD of 3 initializations.",
            transform=axes[j].transAxes,
            fontsize=8,
        )
    fig.suptitle("Language readout of experience-conditioned neural responses", fontsize=14)
    fig.savefig(a.out / "reader_comparison.png", dpi=180)
    fig.savefig(a.out / "reader_comparison.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.3, 4.8), constrained_layout=True)
    for j, (split, title) in enumerate(
        [
            ("test", "New conditioning pairs"),
            ("query_chemical_test__test", "Excluded chemical: acetic acid"),
        ]
    ):
        keys = [
            ("diagnostic_mbon_and_query", "Two diagnostic probes"),
            ("full_history", "Full input history"),
            ("stimulus_profile", "Query profile only"),
        ]
        means, errors, labels = [], [], []
        for prefix, label in keys:
            vals = [
                cpu["prospective_delta"][f"{prefix}__mlp_seed{s}"]["metrics"][split]["mae"][
                    "valence"
                ]
                for s in (0, 1, 2)
            ]
            means.append(np.mean(vals))
            errors.append(np.std(vals, ddof=1))
            labels.append(label)
        axes[j].barh(
            range(3), means, xerr=errors, color=["#009E73", "#D55E00", "#999999"], capsize=3
        )
        baseline = cpu["prospective_delta"]["no_change"][split]["mae"]["valence"]
        axes[j].axvline(baseline, color="#333333", ls="--", lw=1.2, label="Predict no change")
        axes[j].set_yticks(range(3), labels)
        axes[j].invert_yaxis()
        axes[j].set_title(title, pad=12)
        axes[j].set_xlabel("Predicted change in valence: MAE")
    fig.suptitle("MLP forecast of another response without observing it", fontsize=14)
    fig.supxlabel(
        "Dashed line: no-change prediction. Error bars: SD across 3 initializations.", fontsize=8
    )
    fig.savefig(a.out / "prospective_comparison.png", dpi=180)
    fig.savefig(a.out / "prospective_comparison.pdf")
    plt.close(fig)
    print(json.dumps({"status": "complete", "out": str(a.out), "reader_seeds": len(runs)}))


if __name__ == "__main__":
    main()
