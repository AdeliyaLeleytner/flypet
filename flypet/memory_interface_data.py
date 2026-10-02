"""Independent learning histories for the neural-to-language memory interface.

The plan is fixed before simulation.  Counterbalanced assignments and stochastic
repeats of the same chemical pair always share a split group.  This module does
not import Brian2 until execution and never loads or saves the personal pet.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import shutil
import time
from pathlib import Path

import numpy as np

from .experiments import (
    ROOT,
    CaseRunner,
    array_sha256,
    environment_manifest,
    jsonable,
    load_protocol,
    protocol_sha256,
    sha256_file,
    validate_protocol,
    verify_artifacts,
    write_json,
)

SCHEMA_VERSION = 1
PROBE_SPLIT_NAMESPACES = {"train": 0, "validation": 1, "test": 2, "chemical_test": 3}


def pair_id(compounds):
    """Identity is independent of sign, repeat, and input ordering."""
    pair = sorted(compounds)
    if len(pair) != 2 or len(set(pair)) != 2:
        raise ValueError("Each history needs exactly two distinct training compounds")
    digest = hashlib.sha256(json.dumps(pair, separators=(",", ":")).encode()).hexdigest()
    return f"pair-{digest[:12]}"


def make_plan(
    *,
    seeds=(101, 211, 307, 401, 503, 601),
    pairing_trials=(1, 3),
    split_seed=20260924,
    max_pairs=None,
    base_protocol=None,
    heldout_compound="acetic acid",
):
    p = dict(base_protocol or load_protocol())
    seeds = [int(s) for s in seeds]
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0 or max(seeds) >= 100000:
        raise ValueError("Unique seeds in [0, 100000) required for disjoint probe namespaces")
    pairing_trials = [pairing_trials] if isinstance(pairing_trials, int) else list(pairing_trials)
    if (
        not pairing_trials
        or min(pairing_trials) < 1
        or len(set(pairing_trials)) != len(pairing_trials)
    ):
        raise ValueError("pairing_trials must be unique positive integers")
    if heldout_compound not in p["odors"]:
        raise ValueError("Held-out compound must be part of the fixed panel")
    pairs = list(itertools.combinations(p["odors"], 2))
    if max_pairs is not None:
        if max_pairs < 1:
            raise ValueError("max_pairs must be positive")
        pairs = pairs[:max_pairs]
    # Fixed random assignment before neural outputs.  All repeats and opposite
    # signs of a pair go to the same split, not merely individual checkpoints.
    group_order = sorted(pair_id(pair) for pair in pairs if heldout_compound not in pair)
    np.random.default_rng(split_seed).shuffle(group_order)
    n = len(group_order)
    n_val = n_test = max(1, round(n * 0.2)) if n >= 5 else 0
    split_by_group = {
        g: ("train" if j < n - n_val - n_test else "validation" if j < n - n_test else "test")
        for j, g in enumerate(group_order)
    }
    split_by_group.update(
        {pair_id(pair): "chemical_test" for pair in pairs if heldout_compound in pair}
    )
    histories = []
    for pair_index, pair in enumerate(pairs):
        group = pair_id(pair)
        for repeat_index, seed in enumerate(seeds):
            # Run seeds are identical across sign counterfactuals of a pair.
            # Seed namespaces separate training from probe randomness.
            for trials in pairing_trials:
                for assignment in (0, 1):
                    hid = f"{group}-r{repeat_index:02d}-t{trials}-a{assignment}"
                    steps = []
                    for trial in range(trials):
                        for slot, compound in enumerate(pair):
                            mode = "reward" if slot == assignment else "punish"
                            steps.append(
                                {
                                    "trial": trial + 1,
                                    "chemical": compound,
                                    "compound": compound,
                                    "dan_mode": mode,
                                    "reinforcement": mode,
                                    "seed": int(
                                        seed + 100000 + 1000 * pair_index + 2 * trial + slot
                                    ),
                                    "rate_hz": p["odor_rate_max_hz"],
                                    "odor_scale": p["odor_scale"],
                                    "reinforcement_rate_hz": p["reinforcement_rate_hz"],
                                    "duration_ms": p["duration_ms"],
                                }
                            )
                    histories.append(
                        {
                            "history_id": hid,
                            "split_group_id": group,
                            "split": split_by_group[group],
                            "repeat_index": repeat_index,
                            "history_seed": seed,
                            "training_compounds": list(pair),
                            "pairing_trials": trials,
                            "assignment": assignment,
                            "steps": steps,
                        }
                    )
    p.update(
        {
            "protocol_id": "flypet-memory-interface-v2",
            "seeds": seeds,
            "pairing_trials": max(pairing_trials),
            "pairing_trial_counts": pairing_trials,
            "memory_protocol": "Independent reset histories: each unordered chemical pair, both reward/punish assignments, identical exposure order and seeds across assignments; plasticity on every paired training pulse and off on all probes; six odor-only probes of each frozen checkpoint.",
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": p,
        "protocol_sha256": protocol_sha256(p),
        "split_seed": split_seed,
        "heldout_compound": heldout_compound,
        "probe_seed_rule": {
            "base": 2000000,
            "split_stride": 10000000,
            "compound_stride": 100000,
            "split_namespaces": PROBE_SPLIT_NAMESPACES,
            "matched_within": "history split + history seed + compound, across signs and trial counts",
            "disjoint_across": "history splits and query compounds",
        },
        "revision": "v2 changes probe RNG namespaces after a design audit: prevent shared Poisson patterns across splits and diagnostic/query compounds; chemical pairs, outcomes, trial counts and split assignments unchanged. v1 partial artifacts retained.",
        "split_unit": "unordered_training_chemical_pair_across_all_repeats_and_assignments",
        "split_by_group": split_by_group,
        "histories": histories,
        "expected_histories": len(histories),
        "expected_examples": len(histories) * len(p["odors"]),
        "expected_records": len(PROBE_SPLIT_NAMESPACES) * len(seeds) * len(p["odors"])
        + sum(len(h["steps"]) + len(p["odors"]) for h in histories),
        "target_note": "Targets are model-derived MBON readouts, not biological valence measurements. Absolute valence decoding is a readout task, not future-response prediction.",
        "chemical_holdout_rule": "Exclude from fit and validation every row whose probe OR any training-history compound belongs to the held-out set. A chemical fold is separate from the prespecified history split.",
    }


def check_example_splits(examples):
    groups = {}
    for e in examples:
        history_split = e.get("history_split", e["split"])
        old = groups.setdefault(e["split_group_id"], history_split)
        if old != history_split:
            raise ValueError(f"Split leakage for {e['split_group_id']}")
    return {
        s: sum(e["split"] == s for e in examples)
        for s in ("train", "validation", "test", "chemical_test", "query_chemical_test")
    }


def pet_fingerprints():
    names = ("memory.npz", "memory.npz.log.json", "odor_resolved.json", "pet_state.json")
    return {
        str(Path("data") / name): sha256_file(ROOT / "data" / name)
        for name in names
        if (ROOT / "data" / name).is_file()
    }


def _probe_seed(seed, compound_index, history_split):
    # Common random numbers only within the declared counterfactual unit.
    # Split and compound strides exceed their lower-level seed namespaces.
    if not 0 <= seed < 100000 or not 0 <= compound_index < 100:
        raise ValueError("Probe seed namespace would overlap")
    return int(
        2000000 + PROBE_SPLIT_NAMESPACES[history_split] * 10000000 + compound_index * 100000 + seed
    )


def run_plan(output, plan, *, runner=None):
    """Run one brain serially; preserve full raw evidence and an aligned matrix."""
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be new or empty")
    output.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output).free < 2 * 1024**3:
        raise RuntimeError("At least 2 GiB free disk required")
    p = plan["protocol"]
    validation = validate_protocol(p, require_vnc=False)
    write_json(output / "plan.json", plan)
    write_json(output / "protocol.json", p)
    write_json(output / "validation.json", validation)
    before_pet = pet_fingerprints()
    runner = runner or CaseRunner(output, p)
    manifest = environment_manifest(p, [], p["seeds"])
    simulation_sources = {
        f"flypet/{name}.py"
        for name in (
            "__init__",
            "experiments",
            "engine",
            "connectome",
            "mb",
            "odor",
            "corrections",
            "analysis",
            "catalog",
            "memory_interface_data",
        )
    }
    # Reader/evaluator files may be developed concurrently.  Their hashes are
    # irrelevant to reconstructing these simulations and must not block replay.
    manifest["files"] = {
        name: info
        for name, info in manifest["files"].items()
        if not name.endswith(".py") or name in simulation_sources
    }
    manifest.update(
        {
            "status": "running",
            "cases": ["MI"],
            "expected_records": {"MI": plan["expected_records"]},
            "expected_histories": plan["expected_histories"],
            "expected_examples": plan["expected_examples"],
            "plan_sha256": sha256_file(output / "plan.json"),
            "pet_before": before_pet,
            "gpu_cost_usd": 0,
            "llm_calls": 0,
            "claim_scope": plan["target_note"],
        }
    )
    for path in (
        ROOT / "flypet/memory_interface_data.py",
        ROOT / "scripts/gen_memory_interface_data.py",
    ):
        manifest["files"][str(path.relative_to(ROOT))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    write_json(output / "manifest.json", manifest)
    started = time.monotonic()
    examples, rates = [], []
    naive = {}
    try:
        runner.initialize(need_brain=True, need_vnc=False)
        # Fixed anatomical schema, without selecting active neurons from test.
        feature_indices = np.arange(runner.brain.n_stim, runner.brain.n, dtype=np.int32)
        write_json(
            output / "feature_schema.json",
            {
                "features": "all downstream neurons, excluding all directly stimulable tiers",
                "n_neurons": runner.brain.n,
                "n_stim": runner.brain.n_stim,
                "n_features": len(feature_indices),
                "units": "spikes / probe duration, Hz",
                "transform": "none; normalization and active-neuron selection must use training rows only",
                "mbon_indices": runner.mb.MBON.tolist(),
                "mbon_signs": [runner.mb.sign[int(i)] for i in runner.mb.MBON],
                "mbon_types": [runner.mb.mbon_type[int(i)] for i in runner.mb.MBON],
                "mbon_nt": [runner.mb.mbon_nt[int(i)] for i in runner.mb.MBON],
            },
        )
        write_json(
            output / "input_profiles.json",
            {
                "source": "DoOR measured aggregate; missing receptor responses remain null, not zero",
                "glomeruli": runner.door.glomeruli,
                "compounds": {
                    c: {
                        "glomerular_profile": {
                            g: runner.door.glomerular(c).get(g) for g in runner.door.glomeruli
                        },
                        "serialized_inputs": jsonable(
                            [
                                {"ids": s.ids, "rate_hz": s.rate_hz, "key": s.key}
                                for s in runner.odor_inputs(c)
                            ]
                        ),
                    }
                    for c in p["odors"]
                },
            },
        )
        for history_split in PROBE_SPLIT_NAMESPACES:
            for seed in p["seeds"]:
                runner.mb.reset()
                for j, compound in enumerate(p["odors"]):
                    rec = runner.brain_record(
                        case="MI",
                        condition=f"naive-{history_split}-o{j}",
                        compound=compound,
                        inputs=runner.odor_inputs(compound),
                        seed=_probe_seed(seed, j, history_split),
                        history=f"naive-{history_split}-{seed}",
                        phase="naive",
                        metadata={
                            "used_as_training_example": False,
                            "history_split": history_split,
                        },
                    )
                    naive[(history_split, seed, compound)] = rec
        for hi, history in enumerate(plan["histories"]):
            runner.mb.reset()
            for k, step in enumerate(history["steps"]):
                inputs = runner.odor_inputs(step["compound"])
                fn = runner.mb.reward if step["reinforcement"] == "reward" else runner.mb.punishment
                inputs.append(fn(step["reinforcement_rate_hz"]))
                runner.brain_record(
                    case="MI",
                    condition=f"h{hi}-train{k}",
                    compound=step["compound"],
                    inputs=inputs,
                    seed=step["seed"],
                    history=history["history_id"],
                    phase="train",
                    learn=True,
                    reinforcement=step["reinforcement"],
                    metadata={"history": history, "step_index": k},
                )
            checkpoint = runner._memory_snapshot()
            for j, compound in enumerate(p["odors"]):
                probe_seed = _probe_seed(history["history_seed"], j, history["split"])
                rec = runner.brain_record(
                    case="MI",
                    condition=f"h{hi}-probe{j}",
                    compound=compound,
                    inputs=runner.odor_inputs(compound),
                    seed=probe_seed,
                    history=history["history_id"],
                    phase="post",
                    learn=False,
                    metadata={"history": history, "probe_index": j},
                )
                if (
                    rec["memory_before"]["array_sha256"] != checkpoint["array_sha256"]
                    or rec["memory_after"]["array_sha256"] != checkpoint["array_sha256"]
                ):
                    raise AssertionError("Probe changed the learned memory")
                ref = naive[(history["split"], history["history_seed"], compound)]
                example = {
                    "schema_version": SCHEMA_VERSION,
                    "example_id": f"{history['history_id']}-probe{j}",
                    "record_id": rec["record_id"],
                    "history_id": history["history_id"],
                    "split_group_id": history["split_group_id"],
                    "history_split": history["split"],
                    "split": (
                        "query_chemical_test"
                        if compound == plan["heldout_compound"]
                        and history["split"] != "chemical_test"
                        else history["split"]
                    ),
                    "probe_compound": compound,
                    "probe_role": "trained"
                    if compound in history["training_compounds"]
                    else "untrained",
                    "probe_seed": probe_seed,
                    "duration_ms": p["duration_ms"],
                    "training_compounds": history["training_compounds"],
                    "history": history["steps"],
                    "history_seed": history["history_seed"],
                    "assignment": history["assignment"],
                    "pairing_trials": history["pairing_trials"],
                    "checkpoint_hash": checkpoint["array_sha256"],
                    "checkpoint": checkpoint,
                    "state_npz": rec["raw"]["path"],
                    "raw": rec["raw"],
                    "naive_reference": {
                        "record_id": ref["record_id"],
                        "raw": ref["raw"],
                        "valence": ref["valence"],
                    },
                    "target": {
                        "approach_hz": rec["valence"]["approach_hz"],
                        "avoid_hz": rec["valence"]["avoid_hz"],
                        "valence": rec["valence"]["score"],
                        "delta_valence": rec["valence"]["score"] - ref["valence"]["score"],
                    },
                    "mbon_rates_hz": rec["populations"]["MBON"]["rates_hz"],
                }
                examples.append(example)
                with (output / "examples.jsonl").open("a") as f:
                    f.write(
                        json.dumps(jsonable(example), ensure_ascii=False, allow_nan=False) + "\n"
                    )
                with np.load(output / rec["raw"]["path"], allow_pickle=False) as z:
                    vector = np.zeros(runner.brain.n, dtype=np.float32)
                    vector[z["idx"]] = z["rates_hz"]
                rates.append(vector[feature_indices])
            print(
                json.dumps(
                    {
                        "history_complete": history["history_id"],
                        "histories_done": hi + 1,
                        "histories_total": len(plan["histories"]),
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        if (
            len(runner.records) != plan["expected_records"]
            or len(examples) != plan["expected_examples"]
        ):
            raise AssertionError("Unexpected record/example counts")
        splits = check_example_splits(examples)
        np.savez_compressed(
            output / "features.npz",
            rates=np.stack(rates),
            neuron_indices=feature_indices,
            example_ids=np.asarray([e["example_id"] for e in examples]),
            mbon_rates=np.asarray([e["mbon_rates_hz"] for e in examples], dtype=np.float32),
            mbon_indices=np.asarray(runner.mb.MBON, dtype=np.int32),
        )
        # Per-example raw artifacts include all stimulated and downstream neurons;
        # the learner matrix contains only the fixed downstream feature schema.
        verification = verify_artifacts(output)
        if not verification["ok"]:
            raise AssertionError(verification["errors"])
        manifest.update(
            {
                "status": "complete",
                "actual_records": len(runner.records),
                "actual_histories": len(plan["histories"]),
                "actual_examples": len(examples),
                "split_example_counts": splits,
                "artifacts": {
                    name: {
                        "sha256": sha256_file(output / name),
                        "bytes": (output / name).stat().st_size,
                    }
                    for name in (
                        "examples.jsonl",
                        "features.npz",
                        "feature_schema.json",
                        "input_profiles.json",
                        "records.jsonl",
                    )
                },
            }
        )
    except BaseException as exc:
        manifest.update(
            {"status": "failed_or_interrupted", "error": f"{type(exc).__name__}: {exc}"}
        )
        raise
    finally:
        manifest["pet_after"] = pet_fingerprints()
        manifest["pet_unchanged"] = manifest["pet_after"] == before_pet
        manifest["elapsed_s"] = time.monotonic() - started
        write_json(output / "manifest.json", manifest)
    if not manifest["pet_unchanged"]:
        raise AssertionError("Personal pet files changed during isolated execution")
    return manifest
