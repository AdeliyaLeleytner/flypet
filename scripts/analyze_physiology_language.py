"""Summarize a completed two-arm pilot; paired bootstrap resamples whole history groups."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def interval(values, rng):
    v = np.asarray(values)
    draws = v[rng.integers(0, len(v), (10000, len(v)))].mean(1)
    return {
        "mean": float(v.mean()),
        "group_bootstrap_95_percent": np.quantile(draws, [0.025, 0.975]).tolist(),
        "groups": len(v),
    }


def main(a):
    root = Path(a.results)
    report = json.loads((root / "report.json").read_text())
    if not report.get("complete"):
        raise ValueError("Pilot is incomplete; do not manufacture a full comparison")
    out = {
        "scope": "One initialization, paired test groups, conditional on this synthetic task and selected checkpoints",
        "report_sha256": sha(root / "report.json"),
        "arms": {},
        "comparisons": {},
    }
    rng = np.random.default_rng(20260924)
    normal = {}
    group_values = {}
    for name in ["trainable_core", "frozen_core"]:
        rows = json.loads((root / name / "test.json").read_text())
        normal[name] = {r["id"]: r for r in rows}
        zero = {r["id"]: r for r in json.loads((root / name / "test_zero_state.json").read_text())}
        lookup = {(r["group"], r["assignment"], r["query_object"]): r for r in rows}
        groups = sorted({r["group"] for r in rows})
        group_values[name] = {}
        for g in groups:
            rr = [r for r in rows if r["group"] == g]
            group_values[name][g] = sum(r["correct"] for r in rr) / len(rr)
        earlier = [r for r in rows if not r["query_is_last_mentioned"]]
        donor = lambda r: lookup[(r["group"], 1 - r["assignment"], r["query_object"])]
        derived = []
        for row in rows:
            dd = donor(row)
            derived.append(
                {
                    **row,
                    "prediction": dd["prediction"],
                    "generated": dd["generated"],
                    "token_ids": dd["token_ids"],
                    "valid": dd["valid"],
                    "correct": dd["valid"] and dd["prediction"] == row["truth"],
                    "donor_id": dd["id"],
                    "first_answer_token_nll": None,
                    "intervention": "counterfactual_state_swap_derived",
                }
            )
        (root / name / "test_state_swap_derived.json").write_text(
            json.dumps(derived, indent=2) + "\n"
        )
        measured = sum(z["correct"] for z in derived) / len(derived)
        assert abs(measured - report["arms"][name]["test_state_swap"]["all"]["accuracy"]) < 1e-12
        state = torch.load(root / name / "best.pt", map_location="cpu", weights_only=True)
        gain = 0.25 + 1.75 * torch.sigmoid(state["core.raw_gain"])
        clipping = report["arms"][name]["gradient_clipping"]
        out["arms"][name] = {
            "test_accuracy": interval(list(group_values[name].values()), rng),
            "normal_minus_zero_state": interval(
                [
                    np.mean(
                        [
                            int(r["correct"]) - int(zero[r["id"]]["correct"])
                            for r in rows
                            if r["group"] == g
                        ]
                    )
                    for g in groups
                ],
                rng,
            ),
            "normal_minus_state_swap": interval(
                [
                    np.mean(
                        [
                            int(z["correct"]) - int(dd["correct"])
                            for z, dd in zip(rows, derived)
                            if z["group"] == g
                        ]
                    )
                    for g in groups
                ],
                rng,
            ),
            "earlier_object_pair_prediction_changes": float(
                np.mean([r["prediction"] != donor(r)["prediction"] for r in earlier])
            ),
            "earlier_object_both_counterfactual_answers_correct": float(
                np.mean([r["correct"] and donor(r)["correct"] for r in earlier])
            ),
            "gain_absolute_change_from_one_max": float((gain - 1).abs().max()),
            "gain_changed_above_1e_6": int(((gain - 1).abs() > 1e-6).sum()),
            "interface_clipped_fraction": float(
                np.mean([x["interface_norm"] > 1 for x in clipping])
            ),
            "core_clipped_fraction": float(np.mean([x["core_norm"] > 1 for x in clipping])),
        }
        if clipping and "encoder_norm" in clipping[0]:
            out["arms"][name]["encoder_clipped_fraction"] = float(
                np.mean([x["encoder_norm"] > 1 for x in clipping])
            )
            out["arms"][name]["decoder_clipped_fraction"] = float(
                np.mean([x["decoder_norm"] > 1 for x in clipping])
            )
            out["arms"][name]["median_encoder_to_decoder_raw_gradient_norm_ratio"] = float(
                np.median([x["encoder_norm"] / max(x["decoder_norm"], 1e-300) for x in clipping])
            )
            if report["config"].get("clip_groups") == "separate":
                del out["arms"][name]["interface_clipped_fraction"]
    groups = sorted(group_values["trainable_core"])
    out["comparisons"]["trainable_minus_frozen"] = interval(
        [group_values["trainable_core"][g] - group_values["frozen_core"][g] for g in groups], rng
    )
    (root / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True)
    main(p.parse_args())
