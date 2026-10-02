#!/usr/bin/env python3
"""Read-only rescoring of saved predictions; write clearly separate audit files."""

from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.memory_reader_data import digest, parse_output as parse_readout
from flypet.memory_forecast_data import parse_output as parse_forecast


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    data = ROOT / "data/memory_interface_20260924_v2/examples.jsonl"
    rows = {r["example_id"]: r for line in data.read_text().splitlines() if (r := json.loads(line))}
    train = [r["target"]["valence"] for r in rows.values() if r["split"] == "train"]
    mean, median = float(np.mean(train)), float(np.median(train))
    runs = []
    for seed in (0, 1, 2):
        p = ROOT / f"data/architecture_reader_20260924_gpu/seed_{seed}/results/predictions.json"
        valid = []
        for r in json.loads(p.read_text()):
            if r["mode"] != "graph" or r["split"] != "query_chemical_test_from_test":
                continue
            prediction = parse_readout(r["generated"])
            if prediction is not None:
                source = rows[r["example_id"]]
                truth = source["target"]["valence"]
                valid.append(
                    {
                        "example_id": r["example_id"],
                        "gnn": prediction["valence"],
                        "truth": truth,
                        "naive": source["naive_reference"]["valence"]["score"],
                    }
                )
        runs.append(
            {
                "seed": seed,
                "valid": len(valid),
                "total": 48,
                "source_sha256": digest(p),
                "gnn_mae": float(np.mean([abs(r["gnn"] - r["truth"]) for r in valid])),
                "train_mean_mae_same_rows": float(np.mean([abs(mean - r["truth"]) for r in valid])),
                "train_median_mae_same_rows": float(
                    np.mean([abs(median - r["truth"]) for r in valid])
                ),
                "matched_naive_mae_same_rows": float(
                    np.mean([abs(r["naive"] - r["truth"]) for r in valid])
                ),
                "valid_example_ids": [r["example_id"] for r in valid],
            }
        )
    out = ROOT / "data/architecture_reader_analysis_20260924/train_mean_comparison_audit.json"
    save(
        out,
        {
            "status": "complete",
            "source_examples_sha256": digest(data),
            "train_mean": mean,
            "train_median": median,
            "runs": runs,
            "scope": "Same GNN-valid rows. Matched naive is a privileged no-learning-change control. Original frozen analysis remains unchanged.",
            "audit_script_sha256": digest(Path(__file__)),
        },
    )

    forecasts = []
    for seed in (0, 1, 2):
        p = (
            ROOT
            / f"data/memory_forecast_reader_20260924_gpu_fresh/seed_{seed}/results/predictions.json"
        )
        grouped = defaultdict(list)
        for r in json.loads(p.read_text()):
            if r["mode"] not in ("neural", "history", "null") or r["split"] != "test":
                continue
            source = rows[r["example_id"]]
            prediction = parse_forecast(r["generated"])
            truth = source["target"]["delta_valence"]
            record = (truth, prediction["delta_valence"] if prediction else None)
            for family, chemical in [
                (source["split_group_id"], "all"),
                ("all", source["probe_compound"]),
                (source["split_group_id"], source["probe_compound"]),
            ]:
                grouped[(r["mode"], family, chemical)].append(record)
        for (mode, family, chemical), values in grouped.items():
            good = [(y, p) for y, p in values if p is not None]
            forecasts.append(
                {
                    "seed": seed,
                    "mode": mode,
                    "family": family,
                    "chemical": chemical,
                    "n": len(values),
                    "valid": len(good),
                    "mae_valid": float(np.mean([abs(y - p) for y, p in good])) if good else None,
                    "no_change_mae_all": float(np.mean([abs(y) for y, _ in values])),
                    "no_change_mae_same_valid": float(np.mean([abs(y) for y, _ in good]))
                    if good
                    else None,
                }
            )
    dy = [
        r["target"]["delta_valence"]
        for r in rows.values()
        if r["split"] == "train" and r["probe_compound"] not in ("methyl acetate", "1-hexanol")
    ]
    out = ROOT / "data/memory_forecast_reader_20260924_gpu_fresh/family_query_audit.json"
    save(
        out,
        {
            "status": "complete",
            "training_delta_mean": float(np.mean(dy)),
            "training_delta_median": float(np.median(dy)),
            "rows": forecasts,
            "audit_script_sha256": digest(Path(__file__)),
        },
    )
    for x in forecasts:
        if x["mode"] == "neural" and (x["chemical"] == "all" or x["family"] == "all"):
            print(json.dumps(x))


if __name__ == "__main__":
    main()
