#!/usr/bin/env python3
"""Fit simple decoders on fixed family splits; keep prospective features separate."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
import platform
from pathlib import Path
import sys
import time
import warnings

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from flypet.memory_reader_data import FIELDS, digest, load_dataset, prepare_dataset, target_values

DIAGNOSTICS = ("methyl acetate", "1-hexanol")


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def profiles(directory):
    data = json.loads((Path(directory) / "input_profiles.json").read_text())
    groups = data["glomeruli"]
    result = {}
    for name, entry in data["compounds"].items():
        values = [entry["glomerular_profile"][g] for g in groups]
        result[name] = np.asarray(
            [0 if v is None else v for v in values] + [float(v is not None) for v in values],
            dtype=np.float32,
        )
    return result, groups


def yvalues(rows):
    return np.asarray([[target_values(r)[k] for k in FIELDS] for r in rows], dtype=np.float64)


def standardize(x, train):
    mean = x[train].mean(axis=0)
    std = x[train].std(axis=0)
    scale = np.where(std < 1e-8, 1.0, std)
    return ((x - mean) / scale).astype(np.float32), mean, scale


def bounded(pred, kind="readout"):
    pred = np.asarray(pred, dtype=np.float64).copy()
    if kind == "readout":
        pred[:, :2] = np.maximum(0, pred[:, :2])
        pred[:, 2] = np.clip(pred[:, 2], -1, 1)
    elif kind == "delta":
        pred[:, 2] = np.clip(pred[:, 2], -2, 2)
    else:
        raise ValueError("Unknown target kind")
    return pred


def metrics(pred, y, rows, kind="readout"):
    err = np.abs(pred - y)
    truth_class = np.where(y[:, 2] > 0.05, 1, np.where(y[:, 2] < -0.05, -1, 0))
    pred_class = np.where(pred[:, 2] > 0.05, 1, np.where(pred[:, 2] < -0.05, -1, 0))
    return {
        "n": len(rows),
        "history_families": len({r["split_group_id"] for r in rows}),
        "histories": len({r["history_id"] for r in rows}),
        "mae": {k: float(err[:, j].mean()) for j, k in enumerate(FIELDS)},
        "rmse": {
            k: float(np.sqrt(((pred[:, j] - y[:, j]) ** 2).mean())) for j, k in enumerate(FIELDS)
        },
        "valence_sign_accuracy": float(np.mean(truth_class == pred_class)),
        "sign_threshold": 0.05,
        "truth_sign_counts": {str(k): int(np.sum(truth_class == k)) for k in (-1, 0, 1)},
        "target_kind": kind,
        "predicted_readout_consistency_mae": float(
            np.mean(np.abs(pred[:, 2] - (pred[:, 0] - pred[:, 1]) / (pred[:, 0] + pred[:, 1] + 1)))
        )
        if kind == "readout"
        else None,
    }


def intervention_metrics(pred, y, rows):
    lookup = {
        (
            r["split_group_id"],
            r["history_seed"],
            r["pairing_trials"],
            r["probe_compound"],
            r["assignment"],
        ): i
        for i, r in enumerate(rows)
    }
    actual, predicted = [], []
    for key, i in lookup.items():
        if key[-1] != 0 or (*key[:-1], 1) not in lookup:
            continue
        j = lookup[(*key[:-1], 1)]
        actual.append(y[j, 2] - y[i, 2])
        predicted.append(pred[j, 2] - pred[i, 2])
    if not actual:
        return {"n_matched_interventions": 0}
    a, p = np.asarray(actual), np.asarray(predicted)
    return {
        "n_matched_interventions": len(a),
        "valence_contrast_mae": float(np.abs(a - p).mean()),
        "mean_absolute_actual_contrast": float(np.abs(a).mean()),
        "mean_absolute_predicted_contrast": float(np.abs(p).mean()),
    }


def eval_splits(pred, y, rows, kind="readout"):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["split"]].append(i)
        if row["split"] == "query_chemical_test":
            groups["query_chemical_test__" + row["history_split"]].append(i)
    report = {}
    for name, inds in groups.items():
        ids = np.asarray(inds)
        subset = [rows[i] for i in ids]
        report[name] = metrics(pred[ids], y[ids], subset, kind) | intervention_metrics(
            pred[ids], y[ids], subset
        )
        # Comparing identical exposure templates after averaging RNG repeats
        # distinguishes reproducible history effects from realized probe noise.
        templates = defaultdict(list)
        for i in ids:
            r = rows[i]
            templates[
                (r["split_group_id"], r["assignment"], r["pairing_trials"], r["probe_compound"])
            ].append(int(i))
        py = np.asarray([pred[ix].mean(axis=0) for ix in templates.values()])
        ty = np.asarray([y[ix].mean(axis=0) for ix in templates.values()])
        report[name]["template_mean_mae"] = {
            k: float(np.abs(py[:, j] - ty[:, j]).mean()) for j, k in enumerate(FIELDS)
        }
        report[name]["template_count"] = len(templates)
        report[name]["observed_within_template_target_std"] = {
            k: float(
                np.mean([y[ix, j].std(ddof=1) if len(ix) > 1 else 0.0 for ix in templates.values()])
            )
            for j, k in enumerate(FIELDS)
        }
        report[name]["by_compound"] = {}
        for c in sorted({r["probe_compound"] for r in subset}):
            ix = np.asarray([i for i in ids if rows[i]["probe_compound"] == c])
            report[name]["by_compound"][c] = metrics(pred[ix], y[ix], [rows[i] for i in ix], kind)
    return report


def fit_models(x, y, rows, train, validation, *, mlp=True, kind="readout"):
    """All model/epoch choices use designated validation families only."""
    x, xmu, xsd = standardize(x, train)
    ys, ymu, ysd = standardize(y, train)
    models = {}
    best = None
    for alpha in (0.1, 1.0, 10.0, 100.0):
        m = Ridge(alpha=alpha, solver="lsqr", tol=1e-5, max_iter=2000)
        m.fit(x[train], ys[train])
        error = np.abs(m.predict(x[validation]) - ys[validation])
        score = float(error[:, 2].mean() if kind == "delta" else error.mean())
        if best is None or score < best[0]:
            best = (score, alpha, m)
    score, alpha, m = best
    pred = bounded(m.predict(x) * ysd + ymu, kind)
    models["ridge"] = {
        "predictions": pred,
        "selection": {
            "alpha": alpha,
            "validation_standardized_mae": score,
            "grid": [0.1, 1, 10, 100],
        },
        "fitted": {
            "estimator": m,
            "input_mean": xmu,
            "input_scale": xsd,
            "target_mean": ymu,
            "target_scale": ysd,
            "target_kind": kind,
        },
    }
    if mlp:
        for seed in (0, 1, 2):
            m = MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=0.01,
                batch_size=min(64, len(train)),
                learning_rate_init=0.001,
                random_state=seed,
                max_iter=1,
                warm_start=True,
                shuffle=True,
            )
            best_score, best_weights, best_epoch, stagnant = float("inf"), None, None, 0
            for epoch in range(300):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    m.partial_fit(x[train], ys[train])
                error = np.abs(m.predict(x[validation]) - ys[validation])
                score = float(error[:, 2].mean() if kind == "delta" else error.mean())
                if score < best_score - 1e-6:
                    best_score, best_epoch, stagnant = score, epoch + 1, 0
                    best_weights = ([a.copy() for a in m.coefs_], [a.copy() for a in m.intercepts_])
                else:
                    stagnant += 1
                if stagnant >= 30:
                    break
            m.coefs_, m.intercepts_ = best_weights
            models[f"mlp_seed{seed}"] = {
                "predictions": bounded(m.predict(x) * ysd + ymu, kind),
                "selection": {
                    "epoch": best_epoch,
                    "epochs_run": epoch + 1,
                    "validation_standardized_mae": best_score,
                    "seed": seed,
                },
                "fitted": {
                    "estimator": m,
                    "input_mean": xmu,
                    "input_scale": xsd,
                    "target_mean": ymu,
                    "target_scale": ysd,
                    "target_kind": kind,
                },
            }
    return models


def prospective_view(rows, chemical_profiles):
    """Only two fixed OTHER probes from the same frozen memory may inform target."""
    by_history = defaultdict(dict)
    for row in rows:
        if row["probe_compound"] in by_history[row["history_id"]]:
            raise ValueError("Duplicate history/chemical probe")
        by_history[row["history_id"]][row["probe_compound"]] = row
    selected, features, provenance = [], [], []
    for row in rows:
        if row["probe_compound"] in DIAGNOSTICS:
            continue
        sources = [by_history[row["history_id"]][d] for d in DIAGNOSTICS]
        if any(s["checkpoint_hash"] != row["checkpoint_hash"] for s in sources):
            raise ValueError("Prospective context must use the same frozen checkpoint")
        if any(s["example_id"] == row["example_id"] for s in sources):
            raise ValueError("Target probe leaked into diagnostic features")
        selected.append(row)
        features.append(
            np.concatenate(
                [np.log1p(s["mbon_rates_hz"]) for s in sources]
                + [chemical_profiles[row["probe_compound"]]]
            )
        )
        provenance.append(
            {"target": row["example_id"], "sources": [s["example_id"] for s in sources]}
        )
    return selected, np.asarray(features, dtype=np.float32), provenance


def memory_effects(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["probe_compound"], r["probe_role"], r["pairing_trials"])].append(r)
    return [
        {
            "compound": c,
            "role": role,
            "pairing_trials": dose,
            "n": len(group),
            "mean_delta_valence": float(np.mean([r["target"]["delta_valence"] for r in group])),
            "mean_absolute_delta_valence": float(
                np.mean([abs(r["target"]["delta_valence"]) for r in group])
            ),
            "minimum_contributing_spikes": float(
                min(
                    (r["target"]["approach_hz"] + r["target"]["avoid_hz"]) * r["duration_ms"] / 1000
                    for r in group
                )
            ),
            "maximum_contributing_spikes": float(
                max(
                    (r["target"]["approach_hz"] + r["target"]["avoid_hz"]) * r["duration_ms"] / 1000
                    for r in group
                )
            ),
        }
        for (c, role, dose), group in sorted(groups.items())
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ridge-only", action="store_true")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    (a.out / "models").mkdir()
    start = time.monotonic()
    rows, rates, indices, hashes = load_dataset(a.dataset)
    chemical_profiles, glomeruli = profiles(a.dataset)
    prepared = prepare_dataset(rows, rates, indices, chemical_profiles=chemical_profiles)
    np.savez_compressed(
        a.out / "input_preprocessing.npz",
        **{
            k: prepared[k]
            for k in ("neuron_indices", "neural_mu", "neural_sd", "history_mu", "history_sd")
        },
    )
    train, val = prepared["splits"]["train"], prepared["splits"]["validation"]
    y = yvalues(rows)
    features = {
        "neural_downstream": prepared["neural"],
        "mbon_rates": np.asarray([r["mbon_rates_hz"] for r in rows], dtype=np.float32),
        "stimulus_profile": np.asarray([chemical_profiles[r["probe_compound"]] for r in rows]),
        "full_history": prepared["history"],
    }
    del rates
    report = {
        "dataset_files": hashes,
        "split": prepared["manifest"],
        "model_selection": "validation family standardized readout MAE only; never test selection",
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": __import__("sklearn").__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
        },
        "readout_semantics": "computed MBON sums and index, not physical behavior",
        "memory_effects": memory_effects(rows),
        "post_response": {},
        "prospective": {},
        "prospective_delta": {},
        "prospective_delta_selection": "validation delta-valence MAE; rate-delta errors secondary",
    }
    predictions = {}
    mean_pred = np.repeat(y[train].mean(axis=0)[None, :], len(rows), axis=0)
    naive_pred = np.asarray(
        [
            [
                r["naive_reference"]["valence"]["approach_hz"],
                r["naive_reference"]["valence"]["avoid_hz"],
                r["naive_reference"]["valence"]["score"],
            ]
            for r in rows
        ]
    )
    schema = json.loads((a.dataset / "feature_schema.json").read_text())
    signs = np.asarray(schema["mbon_signs"])
    mb = features["mbon_rates"]
    ap_rates, av_rates = mb[:, signs > 0].sum(axis=1), mb[:, signs < 0].sum(axis=1)
    analytic = np.c_[ap_rates, av_rates, (ap_rates - av_rates) / (ap_rates + av_rates + 1)]
    if not np.allclose(analytic, y, atol=1e-5, rtol=1e-6):
        raise ValueError("Raw MBON readout does not reproduce dataset targets")
    report["post_response"]["analytic_readout_identity_check"] = eval_splits(analytic, y, rows)
    report["analytic_readout_note"] = (
        "Known formula applied to the same response; no learned or prospective claim."
    )
    report["post_response"]["train_mean"] = eval_splits(mean_pred, y, rows)
    report["post_response"]["matched_naive_reference"] = eval_splits(naive_pred, y, rows)
    predictions["train_mean"] = mean_pred
    predictions["matched_naive_reference"] = naive_pred
    predictions["analytic_readout_identity_check"] = analytic
    with threadpool_limits(limits=1):
        import joblib

        for name, x in features.items():
            for model, result in fit_models(x, y, rows, train, val, mlp=not a.ridge_only).items():
                key = f"{name}__{model}"
                predictions[key] = result["predictions"]
                joblib.dump(result["fitted"], a.out / "models" / f"post__{key}.joblib", compress=3)
                report["post_response"][key] = {
                    "selection": result["selection"],
                    "metrics": eval_splits(result["predictions"], y, rows),
                }
                print(
                    json.dumps({"finished": key, "elapsed_s": round(time.monotonic() - start, 1)}),
                    flush=True,
                )
                save(a.out / "partial_report.json", report)
        prows, px, pp = prospective_view(rows, chemical_profiles)
        py = yvalues(prows)
        pt = np.flatnonzero(np.asarray([r["split"] for r in prows]) == "train")
        pv = np.flatnonzero(np.asarray([r["split"] for r in prows]) == "validation")
        source_positions = {r["example_id"]: i for i, r in enumerate(rows)}
        positions = [source_positions[r["example_id"]] for r in prows]
        report["prospective"]["matched_naive_reference"] = eval_splits(
            naive_pred[positions], py, prows
        )
        pfeatures = {
            "diagnostic_mbon_and_query": px,
            "stimulus_profile": features["stimulus_profile"][positions],
            "full_history": features["full_history"][positions],
            "global_memory_strength_and_query": np.c_[
                np.asarray([r["checkpoint"]["strength"] for r in prows]),
                features["stimulus_profile"][positions],
            ],
        }
        report["privileged_memory_summary_control"] = (
            "Mean fraction of KC->MBON weight removed plus query profile. It tests whether "
            "a global depression summary explains forecast performance; it is not a language-only baseline."
        )
        ppredictions = {}
        for name, x in pfeatures.items():
            for model, result in fit_models(x, py, prows, pt, pv, mlp=not a.ridge_only).items():
                key = f"{name}__{model}"
                ppredictions[key] = result["predictions"]
                joblib.dump(
                    result["fitted"], a.out / "models" / f"prospective__{key}.joblib", compress=3
                )
                report["prospective"][key] = {
                    "selection": result["selection"],
                    "metrics": eval_splits(result["predictions"], py, prows),
                }
                print(
                    json.dumps(
                        {
                            "finished": "prospective/" + key,
                            "elapsed_s": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
        dy = py - naive_pred[positions]
        dpredictions = {"no_change": np.zeros_like(dy)}
        report["prospective_delta"]["no_change"] = eval_splits(
            dpredictions["no_change"], dy, prows, "delta"
        )
        for name, x in pfeatures.items():
            for model, result in fit_models(
                x, dy, prows, pt, pv, mlp=not a.ridge_only, kind="delta"
            ).items():
                key = f"{name}__{model}"
                dpredictions[key] = result["predictions"]
                joblib.dump(
                    result["fitted"],
                    a.out / "models" / f"prospective_delta__{key}.joblib",
                    compress=3,
                )
                report["prospective_delta"][key] = {
                    "selection": result["selection"],
                    "metrics": eval_splits(result["predictions"], dy, prows, "delta"),
                }
                print(
                    json.dumps(
                        {
                            "finished": "prospective_delta/" + key,
                            "elapsed_s": round(time.monotonic() - start, 1),
                        }
                    ),
                    flush=True,
                )
    np.savez_compressed(
        a.out / "post_response_predictions.npz",
        example_ids=np.asarray([r["example_id"] for r in rows]),
        targets=y,
        **predictions,
    )
    np.savez_compressed(
        a.out / "prospective_predictions.npz",
        example_ids=np.asarray([r["example_id"] for r in prows]),
        targets=py,
        **ppredictions,
    )
    np.savez_compressed(
        a.out / "prospective_features.npz",
        example_ids=np.asarray([r["example_id"] for r in prows]),
        features=px,
    )
    np.savez_compressed(
        a.out / "prospective_delta_predictions.npz",
        example_ids=np.asarray([r["example_id"] for r in prows]),
        targets=dy,
        **dpredictions,
    )
    save(
        a.out / "prospective_context.json",
        {
            "fixed_diagnostic_chemicals": DIAGNOSTICS,
            "feature_order": "log1p MBON rates by diagnostic, followed by query receptor values and missingness masks",
            "target_response_excluded": True,
            "rows": pp,
        },
    )
    report["elapsed_s"] = time.monotonic() - start
    report["source_sha256"] = digest(Path(__file__))
    report["artifact_sha256"] = {
        str(p.relative_to(a.out)): digest(p)
        for p in sorted(a.out.rglob("*"))
        if p.is_file() and p.name not in ("report.json", "partial_report.json")
    }
    report["status"] = "complete"
    save(a.out / "report.json", report)
    print(
        json.dumps({"status": "complete", "out": str(a.out), "elapsed_s": report["elapsed_s"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
