#!/usr/bin/env python3
"""Offline spike-count resolution and saved KC->MBON weight-change audit; no Brain runs."""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet import connectome as C
from flypet.experiments import array_sha256, memory_schedule, sha256_file, write_json


def write_csv(path, rows):
    with Path(path).open("w") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row))
        )
        writer.writeheader()
        writer.writerows(rows)


def change_stats(before, after, mask, contacts):
    delta = np.abs(after.astype(np.float64) - before.astype(np.float64))
    values, weights = delta[mask], contacts[mask]
    changed = values > 0
    mass = float(weights.sum())
    changed_mass = float(weights[changed].sum())
    return {
        "n_edges": int(mask.sum()),
        "n_anatomical_contacts": mass,
        "n_edges_changed": int(changed.sum()),
        "changed_contact_mass": changed_mass,
        "sum_abs_delta_mem": float(values.sum()),
        "mean_abs_delta_mem_per_edge": float(values.mean()) if len(values) else None,
        "mean_abs_delta_mem_per_changed_edge": float(values[changed].mean())
        if changed.any()
        else None,
        "contact_weighted_sum_abs_delta_mem": float(np.dot(weights, values)),
        "contact_weighted_mean_abs_delta_mem": float(np.dot(weights, values) / mass)
        if mass
        else None,
        "contact_weighted_mean_abs_delta_mem_changed_edges": float(
            np.dot(weights, values) / changed_mass
        )
        if changed_mass
        else None,
        "max_abs_delta_mem": float(values.max()) if len(values) else None,
        "n_edges_at_implemented_floor_after": int(np.count_nonzero(after[mask] <= 0.05 + 1e-7)),
    }


def analyze(run, output):
    run, output = Path(run).resolve(), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    all_records = [
        json.loads(line) for line in (run / "records.jsonl").read_text().splitlines() if line
    ]
    records = [r for r in all_records if r["case"] == "C"]
    protocol = json.loads((run / "protocol.json").read_text())
    manifest = json.loads((run / "manifest.json").read_text())
    source_paths = [C.PATH_CON_NPZ, C.PATH_ANN, ROOT / "flypet/mb.py"]
    for path in source_paths:
        expected = manifest["files"][str(path.relative_to(ROOT))]["sha256"]
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen source/data changed: {path}")
    root_ids = np.load(run / "brain_root_ids.npy", allow_pickle=False)
    ann = C.annotations().reindex(root_ids)
    cell_class = ann.cell_class.astype(str).to_numpy()
    cell_type = ann.cell_type.astype(str).to_numpy()
    nt = ann.top_nt.astype(str).to_numpy()
    mbon = np.flatnonzero(cell_class == "MBON")
    kc = np.flatnonzero(cell_class == "Kenyon_Cell")
    # Exact priority/order in MushroomBody.sign, including exclusion of other GABA MBONs.
    sign = np.where(
        nt[mbon] == "glutamate",
        -1,
        np.where((nt[mbon] == "acetylcholine") | (cell_type[mbon] == "MBON11"), 1, 0),
    )
    mbon_groups = {"approach": mbon[sign > 0], "avoid": mbon[sign < 0], "excluded": mbon[sign == 0]}

    @lru_cache(maxsize=None)
    def memory(relative):
        with np.load(run / relative, allow_pickle=False) as z:
            return z["syn_idx"].copy(), z["mem"].copy()

    def checkpoint(snapshot):
        indices, mem = memory(snapshot["path"])
        if array_sha256(indices, mem) != snapshot["array_sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {snapshot['path']}")
        return indices, mem

    syn_idx, _ = checkpoint(records[0]["memory_before"])
    with np.load(C.PATH_CON_NPZ, allow_pickle=False) as z:
        if not np.array_equal(z["order"], root_ids):
            raise ValueError("Connectivity order differs from recorded model order")
        syn_pre, syn_post = z["i_pre"][syn_idx], z["i_post"][syn_idx]
        contacts = np.abs(z["w"][syn_idx].astype(np.float64))
    if not (np.isin(syn_pre, kc).all() and np.isin(syn_post, mbon).all()):
        raise ValueError("Snapshot syn_idx contains a non-KC->MBON edge")
    masks = {name: np.isin(syn_post, indices) for name, indices in mbon_groups.items()}

    @lru_cache(maxsize=None)
    def raw_counts(relative):
        with np.load(run / relative, allow_pickle=False) as z:
            dense = np.zeros(len(root_ids), dtype=np.int64)
            dense[z["idx"]] = z["counts"]
            return dense

    for r in records:
        if sha256_file(run / r["raw"]["path"]) != r["raw"]["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {r['record_id']}")
    naive = {
        (r["seed"], r["compound"]): r
        for r in records
        if r["history"] == "naive" and r["phase"] == "probe"
    }

    def weight_rows(before, after, seed, metadata):
        scopes = {"all_KC": np.ones(len(syn_idx), dtype=bool)}
        for compound in protocol["memory_compounds"]:
            probe = naive[(seed, compound)]
            scopes[f"KC_active_in_naive_probe:{compound}"] = (
                raw_counts(probe["raw"]["path"])[syn_pre] > 0
            )
        return [
            {
                **metadata,
                "scope": scope,
                "postsynaptic_valence_group": group,
                **change_stats(before, after, subset & mask, contacts),
            }
            for scope, subset in scopes.items()
            for group, mask in masks.items()
        ]

    probe_rows, per_cell = [], []
    for r in records:
        if r["phase"] not in ("probe", "post", "reload"):
            continue
        counts = raw_counts(r["raw"]["path"])
        duration_s = r["duration_ms"] / 1000
        sums = {name: int(counts[indices].sum()) for name, indices in mbon_groups.items()}
        approach_hz, avoid_hz = sums["approach"] / duration_s, sums["avoid"] / duration_s
        if not (
            np.isclose(approach_hz, r["valence"]["approach_hz"])
            and np.isclose(avoid_hz, r["valence"]["avoid_hz"])
        ):
            raise ValueError(f"Reconstructed valence sums disagree: {r['record_id']}")
        actual_population_rates = counts[r["populations"]["MBON"]["model_indices"]] / duration_s
        if not np.allclose(actual_population_rates, r["populations"]["MBON"]["rates_hz"]):
            raise ValueError(f"Saved MBON population differs from raw counts: {r['record_id']}")
        row = {
            k: r[k] for k in ("record_id", "history", "phase", "compound", "seed", "duration_ms")
        }
        row.update(
            approach_hz=approach_hz,
            avoid_hz=avoid_hz,
            approach_spikes=sums["approach"],
            avoid_spikes=sums["avoid"],
            contributing_spikes=sums["approach"] + sums["avoid"],
            excluded_mbon_spikes=sums["excluded"],
            all_mbon_spikes=int(counts[mbon].sum()),
            n_mbons=len(mbon),
            n_active_mbons=int(np.count_nonzero(counts[mbon])),
            n_active_contributing_mbons=int(np.count_nonzero(counts[mbon[sign != 0]])),
            valence_score=r["valence"]["score"],
            score_reconstructed_from_counts=(sums["approach"] - sums["avoid"])
            / (sums["approach"] + sums["avoid"] + duration_s),
        )
        if not np.isclose(row["score_reconstructed_from_counts"], row["valence_score"]):
            raise ValueError("Count-level score reconstruction failed")
        probe_rows.append(row)
        for i, group in zip(mbon, sign):
            per_cell.append(
                {
                    "record_id": r["record_id"],
                    "history": r["history"],
                    "compound": r["compound"],
                    "seed": r["seed"],
                    "model_index": int(i),
                    "root_id": int(root_ids[i]),
                    "cell_type": cell_type[i],
                    "neurotransmitter": nt[i],
                    "valence_group": {1: "approach", -1: "avoid", 0: "excluded"}[int(group)],
                    "spikes": int(counts[i]),
                    "rate_hz": float(counts[i] / duration_s),
                }
            )

    training, history_rows, pulse_counts = [], [], []
    train_records = [r for r in records if r["phase"] == "train"]
    base_seed_by_training_seed = {}
    for base_seed in manifest["seeds"]:
        for step in memory_schedule(protocol, base_seed):
            if step["action"] == "train":
                existing = base_seed_by_training_seed.setdefault(step["seed"], base_seed)
                if existing != base_seed:
                    raise ValueError(
                        "Training seed does not uniquely identify its base-seed repeat"
                    )
    for r in train_records:
        pre_idx, before = checkpoint(r["memory_before"])
        post_idx, after = checkpoint(r["memory_after"])
        if not (np.array_equal(pre_idx, syn_idx) and np.array_equal(post_idx, syn_idx)):
            raise ValueError("Synapse index mismatch across snapshots")
        actual_changed = int(np.count_nonzero(before != after))
        metadata = {
            k: r.get(k) for k in ("record_id", "history", "compound", "seed", "reinforcement")
        }
        base_seed = base_seed_by_training_seed[r["seed"]]
        metadata.update(
            seed=base_seed,
            training_seed=r["seed"],
            trial=r["metadata"]["trial"],
            reported_learning_touch_count=r["learning"]["n_synapses_changed"],
            actual_changed_edge_count=actual_changed,
        )
        pulse_counts.append(metadata)
        training.extend(weight_rows(before, after, base_seed, metadata))
    for seed in manifest["seeds"]:
        for history in ("A", "B", "unpaired"):
            pulses = [
                r
                for r in train_records
                if r["history"] == history and base_seed_by_training_seed[r["seed"]] == seed
            ]
            _, before = checkpoint(pulses[0]["memory_before"])
            _, after = checkpoint(pulses[-1]["memory_after"])
            history_rows.extend(
                weight_rows(
                    before,
                    after,
                    seed,
                    {
                        "seed": seed,
                        "history": history,
                        "first_training_record": pulses[0]["record_id"],
                        "last_training_record": pulses[-1]["record_id"],
                    },
                )
            )
    payload = {
        "analysis": "offline_descriptive_memory_resolution",
        "source_run": str(run),
        "source_records_sha256": sha256_file(run / "records.jsonl"),
        "source_data": {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths},
        "mbon_group_sizes": {name: len(indices) for name, indices in mbon_groups.items()},
        "n_mbons_total": len(mbon),
        "n_kc_mbon_edges": len(syn_idx),
        "definitions": {
            "spikes": "Exact raw counts summed across the specified neuron group; rates * 0.25 s for these probes.",
            "contributing": "Approach plus avoidance subset only; other MBONs are excluded from the valence score, not from all-MBON totals.",
            "sign": "glutamate=-1; otherwise acetylcholine or cell_type MBON11=+1; otherwise 0, exactly as mb.sign.",
            "weighted_delta": "abs(after_mem-before_mem) weighted by abs(connectivity w), the anatomical contact count per KC->MBON edge; not physical synaptic current or corrected effective weight.",
            "unweighted_delta": "Each stored KC->MBON edge receives equal weight; mem is a dimensionless learned multiplicative factor.",
            "active_scope": "Subset whose presynaptic KC spiked in the same-seed naive probe of the named chemical; subsets may overlap, and this restriction does not establish causal specificity.",
            "score_resolution": "The score equals (approach_spikes-avoid_spikes)/(contributing_spikes+duration_s), because its original denominator has a +1 Hz stabilizer.",
            "scope": "Descriptive finite-window counts and saved weight changes only. Low contributing counts neither erase the punishment shift nor demonstrate a biological floor or explain a null reward response causally.",
        },
        "probe_rows": probe_rows,
        "training_pulse_count_checks": pulse_counts,
        "training_weight_changes": training,
        "history_weight_changes": history_rows,
    }
    write_json(output / "memory_resolution.json", payload)
    write_csv(output / "memory_resolution.csv", probe_rows)
    write_csv(output / "memory_resolution_mbon_counts.csv", per_cell)
    write_csv(output / "memory_resolution_training_weights.csv", training)
    write_csv(output / "memory_resolution_history_weights.csv", history_rows)
    return payload


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, default=ROOT / "paper/results/cases-v1")
    ap.add_argument("--output", type=Path, default=ROOT / "paper/analysis")
    args = ap.parse_args()
    result = analyze(args.run, args.output)
    print(
        json.dumps(
            {
                "n_probe_rows": len(result["probe_rows"]),
                "mbon_group_sizes": result["mbon_group_sizes"],
                "n_training_pulses": len(result["training_pulse_count_checks"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
