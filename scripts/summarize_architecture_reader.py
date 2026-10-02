#!/usr/bin/env python3
"""Summarize all three prespecified architectures/seeds, including parse failures."""

from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import re
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.memory_reader_data import parse_output, target_values

MODES = ("mlp", "attention", "graph")


def finite_numeric_output(text):
    """Diagnostic only: retain finite out-of-range numbers; never clip them.

    The prespecified strict parser/coverage is retained separately. This extra
    readout prevents a model emitting valence=3 from disappearing from MAE.
    """
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return None
    try:
        value = json.loads(match.group())
        if set(value) != {"approach_hz", "avoid_hz", "valence"}:
            return None
        if any(type(v) not in (int, float) or not np.isfinite(v) for v in value.values()):
            return None
        return value
    except (ValueError, TypeError):
        return None


def contrasts(predictions, examples):
    grouped = defaultdict(dict)
    for p in predictions:
        if p["mode"] not in MODES:
            continue
        row = examples[p["example_id"]]
        key = (
            p["mode"],
            p["split"],
            row["split_group_id"],
            row["history_seed"],
            row["pairing_trials"],
            row["probe_compound"],
        )
        grouped[key][row["assignment"]] = (row, parse_output(p["generated"]))
    result = defaultdict(list)
    counts = defaultdict(int)
    for key, pair in grouped.items():
        if set(pair) != {0, 1}:
            continue
        counts[key[:2]] += 1
        (a, pa), (b, pb) = pair[0], pair[1]
        if pa is None or pb is None:
            continue
        truth = b["target"]["valence"] - a["target"]["valence"]
        predicted = pb["valence"] - pa["valence"]
        result[key[:2]].append(
            {
                "group": key[2],
                "chemical": key[-1],
                "truth": truth,
                "predicted": predicted,
                "absolute_error": abs(truth - predicted),
            }
        )
    return {
        f"{m}/{s}": {
            "total_pairs": n,
            "valid_pairs": len(result[(m, s)]),
            "coverage": len(result[(m, s)]) / n,
            "mae_valid": float(np.mean([r["absolute_error"] for r in result[(m, s)]]))
            if result[(m, s)]
            else None,
            "zero_contrast_mae_valid": float(np.mean([abs(r["truth"]) for r in result[(m, s)]]))
            if result[(m, s)]
            else None,
            "rows": result[(m, s)],
        }
        for (m, s), n in counts.items()
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    examples = {
        r["example_id"]: r
        for line in (a.data / "examples.jsonl").read_text().splitlines()
        if (r := json.loads(line))
    }
    median = {
        k: float(
            np.median([target_values(r)[k] for r in examples.values() if r["split"] == "train"])
        )
        for k in ("approach_hz", "avoid_hz", "valence")
    }
    runs, flat, paired, slices = [], [], [], []
    common_fingerprint = None
    for path in sorted(a.runs.glob("seed_*/results/report.json")):
        report = json.loads(path.read_text())
        if not report.get("complete") or set(report["training"]) != set(MODES):
            raise ValueError(f"Incomplete three-arm run: {path}")
        manifest = json.loads(path.with_name("manifest.json").read_text())
        identity = (
            manifest["split_fingerprint"],
            manifest["graph"]["graph_sha256"],
            manifest["sources"],
        )
        if common_fingerprint is not None and identity != common_fingerprint:
            raise ValueError("Seed inputs/splits differ")
        common_fingerprint = identity
        predictions = json.loads(path.with_name("predictions.json").read_text())
        seed = report["config"]["seed"]
        pairs = contrasts(predictions, examples)
        runs.append(
            {
                "seed": seed,
                "training": report["training"],
                "contrasts": pairs,
                "elapsed_seconds": report["elapsed_seconds"],
                "state_swap": report.get("state_swap"),
            }
        )
        by_split = defaultdict(lambda: defaultdict(dict))
        numeric_split = defaultdict(lambda: defaultdict(dict))
        for p in predictions:
            if p["mode"] in MODES:
                by_split[p["split"]][p["mode"]][p["example_id"]] = parse_output(p["generated"])
                numeric_split[p["split"]][p["mode"]][p["example_id"]] = finite_numeric_output(
                    p["generated"]
                )
        for split, mode_rows in by_split.items():
            ids = set(mode_rows["mlp"])
            if any(set(mode_rows[m]) != ids for m in MODES):
                raise ValueError("Architecture evaluation rows differ")
            common = sorted(i for i in ids if all(mode_rows[m][i] is not None for m in MODES))
            common_numeric = sorted(
                i for i in ids if all(numeric_split[split][m][i] is not None for m in MODES)
            )
            for mode in MODES:
                metric = report["evaluation"][mode][split]
                c = pairs.get(f"{mode}/{split}", {})
                numeric = numeric_split[split][mode]
                numeric_ids = [i for i in ids if numeric[i] is not None]
                row = {
                    "seed": seed,
                    "mode": mode,
                    "split": split,
                    "n": metric["n"],
                    "parse_rate": metric["parse_rate"],
                    "valence_mae_valid": metric["mae_valid"]["valence"],
                    "finite_numeric_coverage": len(numeric_ids) / len(ids),
                    "finite_but_out_of_range_count": sum(
                        mode_rows[mode][i] is None for i in numeric_ids
                    ),
                    "valence_mae_finite_numeric": float(
                        np.mean(
                            [
                                abs(numeric[i]["valence"] - examples[i]["target"]["valence"])
                                for i in numeric_ids
                            ]
                        )
                    )
                    if numeric_ids
                    else None,
                    "approach_mae_hz_valid": metric["mae_valid"]["approach_hz"],
                    "avoid_mae_hz_valid": metric["mae_valid"]["avoid_hz"],
                    "contrast_mae_valid": c.get("mae_valid"),
                    "contrast_coverage": c.get("coverage"),
                    "common_valid_n": len(common),
                    "valence_mae_common": float(
                        np.mean(
                            [
                                abs(
                                    mode_rows[mode][i]["valence"] - examples[i]["target"]["valence"]
                                )
                                for i in common
                            ]
                        )
                    )
                    if common
                    else None,
                    "train_median_valence_mae_all": float(
                        np.mean(
                            [abs(median["valence"] - examples[i]["target"]["valence"]) for i in ids]
                        )
                    ),
                    "nll": metric["nll"],
                    "parameters": report["training"][mode]["parameters"],
                    "generation_seconds_per_row": metric.get("generation_seconds_per_row"),
                }
                for field in ("approach_hz", "avoid_hz"):
                    row[field + "_mae_finite_numeric"] = (
                        float(
                            np.mean(
                                [
                                    abs(numeric[i][field] - examples[i]["target"][field])
                                    for i in numeric_ids
                                ]
                            )
                        )
                        if numeric_ids
                        else None
                    )
                flat.append(row)
                for chemical in sorted({examples[i]["probe_compound"] for i in ids}):
                    selected = [i for i in ids if examples[i]["probe_compound"] == chemical]
                    valid = [i for i in selected if mode_rows[mode][i] is not None]
                    slices.append(
                        {
                            "seed": seed,
                            "mode": mode,
                            "split": split,
                            "chemical": chemical,
                            "n": len(selected),
                            "valid": len(valid),
                            "valence_mae_valid": float(
                                np.mean(
                                    [
                                        abs(
                                            mode_rows[mode][i]["valence"]
                                            - examples[i]["target"]["valence"]
                                        )
                                        for i in valid
                                    ]
                                )
                            )
                            if valid
                            else None,
                        }
                    )
            for left, right in [("attention", "mlp"), ("graph", "attention"), ("graph", "mlp")]:
                errors = [
                    abs(mode_rows[left][i]["valence"] - examples[i]["target"]["valence"])
                    - abs(mode_rows[right][i]["valence"] - examples[i]["target"]["valence"])
                    for i in common
                ]
                numeric_errors = [
                    abs(numeric_split[split][left][i]["valence"] - examples[i]["target"]["valence"])
                    - abs(
                        numeric_split[split][right][i]["valence"] - examples[i]["target"]["valence"]
                    )
                    for i in common_numeric
                ]
                paired.append(
                    {
                        "seed": seed,
                        "split": split,
                        "left": left,
                        "right": right,
                        "n_common": len(common),
                        "mean_error_difference_left_minus_right": float(np.mean(errors))
                        if errors
                        else None,
                        "n_common_finite_numeric": len(common_numeric),
                        "mean_numeric_error_difference_left_minus_right": float(
                            np.mean(numeric_errors)
                        )
                        if numeric_errors
                        else None,
                        "scope": "paired rows conditional on all three parsers; negative favors left",
                    }
                )
    if sorted(r["seed"] for r in runs) != [0, 1, 2]:
        raise ValueError("Expected three completed prespecified initializations")
    for name, rows in [
        ("metrics.csv", flat),
        ("chemical_slices.csv", slices),
        ("paired_differences.csv", paired),
    ]:
        with (a.out / name).open("w") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {
        "complete": True,
        "seeds": [0, 1, 2],
        "runs": runs,
        "metrics": flat,
        "paired_differences": paired,
        "input_identity": {
            "split_fingerprint": common_fingerprint[0],
            "graph_sha256": common_fingerprint[1],
            "dataset_sources": common_fingerprint[2],
        },
        "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "One fixed graph/dataset split; three training initializations, not three independent animals. Current-response neural readout only.",
        "selection": "validation NLL; no test-driven architecture hyperparameter tuning",
        "graph_vs_attention_caveat": "same observations/metadata/pooling; graph adds topology and parameters",
        "posthoc_diagnostic": "After seeing an MLP emit finite valence=3 on the excluded chemical, added finite-numeric MAE without range rejection or clipping. Original strict valid-output metrics retained. No training or selection changed.",
    }
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
    panels = [
        ("test", "New conditioning pairs"),
        ("chemical_test", "New conditioning chemical"),
        ("query_chemical_test_from_test", "New query: acetic acid"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.1), constrained_layout=True)
    for ax, (split, title) in zip(axes, panels):
        vals = []
        coverage = []
        numeric_coverage = []
        for mode in MODES:
            selected = [r for r in flat if r["mode"] == mode and r["split"] == split]
            vals.append([r["valence_mae_finite_numeric"] for r in selected])
            coverage.append(np.mean([r["parse_rate"] for r in selected]) if selected else 0)
            numeric_coverage.append(
                np.mean([r["finite_numeric_coverage"] for r in selected]) if selected else 0
            )
        if any(not values for values in vals):
            ax.set_title(title)
            ax.text(0.1, 0.5, "Partition absent from this fixture")
            continue
        if any(any(v is None for v in x) for x in vals):
            ax.text(0.1, 0.5, "Unparsed outputs; see report")
            continue
        ax.barh(
            range(3),
            [np.mean(v) for v in vals],
            color=["#777777", "#0072B2", "#D55E00"],
            alpha=0.65,
        )
        for y, values in enumerate(vals):
            ax.scatter(
                values, y + np.linspace(-0.10, 0.10, len(values)), s=24, color="#222222", zorder=3
            )
        ax.set_yticks(range(3), ["MLP", "Attention", "Graph + attention"])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.set_xlabel("Valence MAE: all finite numeric outputs")
        ax.text(
            0,
            -0.24,
            "Numeric coverage: "
            + ", ".join(f"{v:.1%}" for v in numeric_coverage)
            + "\nValid-range coverage: "
            + ", ".join(f"{v:.1%}" for v in coverage),
            transform=ax.transAxes,
            fontsize=8,
        )
    fig.suptitle("Projector comparison: bars = means; dots = three initializations", fontsize=13)
    fig.savefig(a.out / "finite_numeric_error_diagnostic.png", dpi=180)
    fig.savefig(a.out / "finite_numeric_error_diagnostic.pdf")
    plt.close(fig)

    # Primary presentation keeps the prespecified strict scores and explicitly
    # shows missing/invalid outputs, rather than allowing huge invalid numbers
    # to dominate a single axis or silently averaging only successful seeds.
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 7.0), constrained_layout=True)
    colors = ["#777777", "#0072B2", "#D55E00"]
    for j, (split, title) in enumerate(panels):
        top, bottom = axes[0, j], axes[1, j]
        coverages = []
        for y, mode in enumerate(MODES):
            selected = [r for r in flat if r["mode"] == mode and r["split"] == split]
            values = [
                r["valence_mae_valid"] for r in selected if r["valence_mae_valid"] is not None
            ]
            coverage = [r["parse_rate"] * 100 for r in selected]
            if values:
                top.barh(y, np.mean(values), color=colors[y], alpha=0.65)
                top.scatter(
                    values,
                    y + np.linspace(-0.10, 0.10, len(values)),
                    s=24,
                    color="#222222",
                    zorder=3,
                )
                if len(values) < 3:
                    top.text(
                        np.mean(values) * 0.52,
                        y,
                        f"{len(values)}/3 seeds valid",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#9B2226",
                    )
            if coverage:
                bottom.barh(y, np.mean(coverage), color=colors[y], alpha=0.65)
                bottom.scatter(
                    coverage,
                    y + np.linspace(-0.10, 0.10, len(coverage)),
                    s=24,
                    color="#222222",
                    zorder=3,
                )
                bottom.text(
                    np.mean(coverage) + 1, y, f"{np.mean(coverage):.1f}%", va="center", fontsize=8
                )
        for ax in (top, bottom):
            ax.set_yticks(
                range(3), ["MLP", "Attention", "GNN + attention"] if j == 0 else ["", "", ""]
            )
            ax.set_ylim(2.6, -0.6)
        top.set_title(title)
        top.set_xlabel("Valence MAE, valid outputs only")
        bottom.set_xlim(0, 119)
        bottom.set_xlabel("Valid structured outputs (%)")
    fig.suptitle(
        "State-reading accuracy and reliability — three initializations per architecture",
        fontsize=13,
    )
    fig.savefig(a.out / "architecture_comparison.png", dpi=180)
    fig.savefig(a.out / "architecture_comparison.pdf")
    plt.close(fig)
    (a.out / "analysis_source.py").write_bytes(Path(__file__).read_bytes())
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()
    }
    (a.out / "SHA256.json").write_text(json.dumps(hashes, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"complete": True, "out": str(a.out), "seeds": len(runs)}))


if __name__ == "__main__":
    main()
