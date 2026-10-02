"""File-backed, LLM-free paper cases and exact-input replay.

Imports of the simulator are deliberately lazy: protocol validation and artifact
inspection must not build a 15-million-synapse network or touch pet memory.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "paper" / "cases" / "protocol.json"
SCHEMA_VERSION = 1


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_sha256(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str(a.dtype).encode())
        h.update(json.dumps(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def serialize_inputs(inputs):
    return [
        {"ids": [int(i) for i in x.ids], "rate_hz": float(x.rate_hz), "key": x.key} for x in inputs
    ]


def deserialize_inputs(rows):
    from .engine import StimInput

    return [StimInput([int(i) for i in r["ids"]], float(r["rate_hz"]), r["key"]) for r in rows]


def effective_inputs(brain, inputs):
    """Record final neuron rates, including the engine's ordered overwrite rule."""
    rates = {}
    for item in inputs:
        if not np.isfinite(item.rate_hz) or item.rate_hz < 0:
            raise ValueError(f"Invalid input rate: {item.key}")
        if item.rate_hz == 0:
            continue
        for root_id in item.ids:
            index = brain.flyid2i.get(int(root_id))
            if index is not None and index < brain.n_stim:
                rates[index] = float(item.rate_hz)
    return [
        {"model_index": i, "root_id": int(brain.i2flyid[i]), "rate_hz": rates[i]}
        for i in sorted(rates)
    ]


def load_protocol(path=DEFAULT_PROTOCOL):
    p = json.loads(Path(path).read_text())
    if p["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported protocol schema")
    if len(p["seeds"]) != len(set(p["seeds"])) or not p["seeds"]:
        raise ValueError("Protocol seeds must be unique and nonempty")
    if p["duration_ms"] <= 0 or p["pairing_trials"] <= 0:
        raise ValueError("Positive duration and trial count required")
    if len(p["memory_compounds"]) != 2 or len(set(p["memory_compounds"])) != 2:
        raise ValueError("Exactly two distinct memory compounds required")
    return p


def protocol_sha256(protocol):
    return hashlib.sha256(
        json.dumps(
            protocol, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def validate_protocol(protocol, require_vnc=False):
    """Read annotations and input data only; never construct Brain/Vnc."""
    from .odor import DoorOdor
    from .catalog import STIMULI
    from . import connectome as C

    door = DoorOdor(protocol["odor_rate_max_hz"], protocol["odor_min_response"])
    odors = {}
    for compound in dict.fromkeys(protocol["odors"] + protocol["memory_compounds"]):
        stimuli = door.stim(compound, scale=protocol["odor_scale"])
        if not stimuli:
            raise ValueError(f"Empty DoOR input: {compound}")
        measured, total = door.coverage(compound)
        odors[compound] = {
            "measured_glomeruli": measured,
            "total_model_glomeruli": total,
            "driven_glomeruli": len(stimuli),
            "glomerular_profile": door.glomerular(compound),
        }
    similarities = []
    for a in protocol["odors"]:
        similarities.append(
            [
                None if not np.isfinite(v := door.similarity(a, b)) else float(v)
                for b in protocol["odors"]
            ]
        )
    for condition in protocol["sensory"]:
        for stimulus in condition["stimuli"]:
            if stimulus["key"] not in STIMULI or not STIMULI[stimulus["key"]].resolve():
                raise ValueError(f"Unresolved sensory input: {stimulus['key']}")
    paths = [C.PATH_COMP, C.PATH_ANN, C.PATH_CON_NPZ]
    if require_vnc:
        from .vnc import PATH_NPZ, PATH_NODES

        paths += [PATH_NPZ, PATH_NODES]
    missing = [str(p.relative_to(ROOT)) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Required data missing: {missing}")
    if require_vnc:
        import pandas as pd

        nodes = pd.read_parquet(PATH_NODES)
        g = protocol["gf"]
        if not ((nodes.type == g["cell_type"]) | (nodes.flywire_type == g["cell_type"])).any():
            raise ValueError("Required GF driver missing from VNC table")
        if not ((nodes.type == g["target_motor_type"]) & (nodes.superclass == "vnc_motor")).any():
            raise ValueError("Required target motor type missing from VNC table")
    return {
        "status": "validated_without_simulation",
        "odors": odors,
        "input_similarity_compounds": protocol["odors"],
        "input_cosine_similarity": similarities,
        "limitations": "Pairwise cosine uses only jointly measured glomeruli; missing responses are not zeros.",
    }


def memory_schedule(protocol, seed):
    """Explicit reset/probe/train/checkpoint/reload schedule; no hidden extra runs."""
    compounds = protocol["memory_compounds"]

    def probes(history, phase):
        return [
            {
                "action": "probe",
                "history": history,
                "phase": phase,
                "compound": c,
                "seed": int(seed),
                "reinforcement": "none",
            }
            for c in compounds
        ]

    out = [{"action": "reset", "history": "naive"}] + probes("naive", "probe")
    for h, reinforcements in [
        ("A", ["reward", "punish"]),
        ("B", ["punish", "reward"]),
        ("unpaired", ["reward", "punish"]),
    ]:
        out.append({"action": "reset", "history": h})
        for trial in range(protocol["pairing_trials"]):
            for j, (compound, reinforcement) in enumerate(zip(compounds, reinforcements)):
                # A, B and unpaired odor exposures share seeds within each trial.
                train_seed = int(seed) + 10000 + 2 * trial + j
                common = {"history": h, "phase": "train", "trial": trial + 1}
                out.append(
                    common
                    | {
                        "action": "train",
                        "compound": compound,
                        "reinforcement": "none" if h == "unpaired" else reinforcement,
                        "seed": train_seed,
                    }
                )
                if h == "unpaired":
                    out.append(
                        common
                        | {
                            "action": "train",
                            "compound": None,
                            "reinforcement": reinforcement,
                            "seed": train_seed + 100000,
                        }
                    )
        out += probes(h, "post")
        if h == "A":
            out.append({"action": "checkpoint", "history": "A"})
    out += [
        {"action": "reset", "history": "A_reloaded"},
        {"action": "reload", "history": "A_reloaded"},
    ] + probes("A_reloaded", "reload")
    return out


def expected_counts(protocol, cases, seeds):
    n = len(seeds)
    return {
        c: n
        * {
            "A": len(protocol["odors"]),
            "B": len(protocol["sensory"]),
            "C": sum(
                x["action"] in ("probe", "train") for x in memory_schedule(protocol, seeds[0])
            ),
            "GF": 1 + protocol["gf"]["random_panels"],
            "smoke": 5,
        }[c]
        for c in cases
    }


def environment_manifest(protocol, cases, seeds):
    from . import connectome as C
    from .odor import DOOR
    from . import corrections
    from brian2 import defaultclock, ms, prefs

    tracked = list((ROOT / "flypet").glob("*.py")) + [
        ROOT / "scripts/run_paper_cases.py",
        DEFAULT_PROTOCOL,
    ]
    tracked += [
        C.PATH_COMP,
        C.PATH_ANN,
        C.PATH_CON_NPZ,
        DOOR / "door_response_matrix.csv",
        DOOR / "odor.csv",
        DOOR / "door_mappings.csv",
    ]
    if set(cases) & {"B", "GF", "smoke"}:
        from .vnc import PATH_NPZ, PATH_NODES

        tracked += [PATH_NPZ, PATH_NODES]
    artifacts = {
        str(p.relative_to(ROOT)): {"sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in sorted(set(tracked))
        if p.is_file()
    }
    versions = {}
    for package in ["numpy", "pandas", "brian2", "pyarrow", "scipy", "cython"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None

    def git(args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha256(protocol),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cases": cases,
        "seeds": seeds,
        "expected_records": expected_counts(protocol, cases, seeds),
        "protocol_deviations": []
        if seeds == protocol["seeds"]
        else ["Explicit seed subset/override"],
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "git_head": git(["rev-parse", "HEAD"]),
        "git_status": git(["status", "--porcelain"]),
        "files": artifacts,
        "modes": protocol["modes"] | {"execution": "scripted_physics"},
        "corrections": list(corrections.DEFAULT),
        "brian_target": str(prefs.codegen.target),
        "dt_ms": float(defaultclock.dt / ms),
        "claim_scope": "Simulation seeds are repeats of one connectome, not biological replicates.",
    }


class CaseRunner:
    def __init__(self, output, protocol, *, brain=None, mb=None, door=None, vnc=None):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if (self.output / "records.jsonl").exists():
            raise FileExistsError(
                "Refusing to overwrite an existing run; choose a new output directory"
            )
        self.protocol = protocol
        self.brain, self.mb, self.door, self.vnc = brain, mb, door, vnc
        self.records = []
        self.counter = 0

    def initialize(self, need_brain=True, need_vnc=False):
        if need_brain and self.brain is None:
            from .engine import Brain
            from .mb import MushroomBody
            from .odor import DoorOdor
            from . import corrections

            self.brain = corrections.apply(Brain())
            self.mb = MushroomBody(self.brain)
            self.door = DoorOdor(
                self.protocol["odor_rate_max_hz"], self.protocol["odor_min_response"]
            )
            self.mb.reset()
            mapping = np.asarray(
                [self.brain.i2flyid[i] for i in range(self.brain.n)], dtype=np.int64
            )
            np.save(self.output / "brain_root_ids.npy", mapping, allow_pickle=False)
        if need_vnc and self.vnc is None:
            from .vnc import Vnc

            self.vnc = Vnc()
            self.vnc.nodes.to_csv(self.output / "vnc_nodes.csv", index=False)
            self.vnc.drivers.to_csv(self.output / "vnc_drivers.csv", index=False)

    def _memory_snapshot(self):
        digest = array_sha256(self.mb.syn_idx, self.mb.mem)
        relative = Path("memory") / f"{digest}.npz"
        path = self.output / relative
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            np.savez_compressed(path, syn_idx=self.mb.syn_idx, mem=self.mb.mem)
        return {
            "path": str(relative),
            "array_sha256": digest,
            "strength": self.mb.memory_strength(),
            "log_length": len(self.mb.log),
        }

    def _append(self, record):
        record = jsonable(record)
        with (self.output / "records.jsonl").open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.records.append(record)
        print(
            json.dumps(
                {
                    "record": record["record_id"],
                    "case": record["case"],
                    "condition": record["condition"],
                    "seed": record["seed"],
                    "wall_s": record["wall_s"],
                }
            ),
            flush=True,
        )
        return record

    def _vnc_artifact(self, vr, prefix):
        relative = Path("raw") / f"{prefix}.vnc.npz"
        np.savez_compressed(
            self.output / relative, rates_hz=vr.rates, driver_rates_hz=vr.driver_rates
        )
        mn = vr.nodes.superclass.to_numpy() == "vnc_motor"
        indices = np.flatnonzero(mn)
        rows = [
            {
                "node_index": int(i),
                "type": str(vr.nodes.iloc[i].type),
                "side": str(vr.nodes.iloc[i].side),
                "muscle": str(vr.nodes.iloc[i].muscle),
                "rate_hz": float(vr.rates[i]),
            }
            for i in indices
        ]
        return {
            "path": str(relative),
            "sha256": sha256_file(self.output / relative),
            "duration_ms": vr.duration_s * 1000,
            "driver_nonzero": int(np.count_nonzero(vr.driver_rates)),
            "brief": vr.brief(),
            "all_motor_rates": rows,
            "driven": vr.driven,
        }

    def brain_record(
        self,
        *,
        case,
        condition,
        inputs,
        seed,
        history="naive",
        phase="probe",
        compound=None,
        learn=False,
        reinforcement="none",
        metadata=None,
        with_vnc=False,
    ):
        from . import analysis as A

        self.counter += 1
        rid = f"{self.counter:04d}-{case}-{condition}-{seed}"
        raw_dir = self.output / "raw"
        raw_dir.mkdir(exist_ok=True)
        before = self._memory_snapshot()
        actual_inputs = serialize_inputs(inputs)
        effective = effective_inputs(self.brain, inputs)
        result = self.brain.run(inputs, duration_ms=self.protocol["duration_ms"], seed=seed)
        sparse_idx = np.asarray(sorted(result.counts), dtype=np.int32)
        counts = np.asarray([result.counts[int(i)] for i in sparse_idx], dtype=np.int32)
        rates = counts.astype(np.float64) / result.duration_s
        relative = Path("raw") / f"{rid}.brain.npz"
        # Full spike trains preserve both rate and timing evidence; no float16 quantization.
        times = (
            np.concatenate([result.trains[int(i)] for i in sparse_idx])
            if len(sparse_idx)
            else np.zeros(0)
        )
        offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)])
        np.savez_compressed(
            self.output / relative,
            idx=sparse_idx,
            counts=counts,
            rates_hz=rates,
            spike_times_s=times,
            offsets=offsets,
        )
        valence = asdict(self.mb.valence(result))
        summary = A.summarize(result)
        populations = {}
        from . import connectome as C

        for label, cell_class in [
            ("ORN", "olfactory"),
            ("PN", "ALPN"),
            ("KC", "Kenyon_Cell"),
            ("MBON", "MBON"),
        ]:
            ids = C.select(cell_class=cell_class)
            indices = np.asarray([self.brain.flyid2i[i] for i in ids], dtype=np.int32)
            v = result.rate_vector(self.brain.n)[indices]
            populations[label] = {
                "n": len(ids),
                "n_active": int(np.count_nonzero(v)),
                "fraction_active": float(np.count_nonzero(v) / len(ids)) if ids else 0.0,
                "mean_rate_hz": float(v.mean()) if len(v) else 0.0,
                "model_indices": indices.tolist(),
                "rates_hz": v.tolist(),
            }
        vnc_result = (
            self.vnc.run_from_brain(result, duration_ms=self.protocol["vnc_duration_ms"], seed=seed)
            if with_vnc
            else None
        )
        learning = self.mb.learn(result, note=f"{case}/{history}/{condition}") if learn else None
        after = self._memory_snapshot()
        record = {
            "schema_version": SCHEMA_VERSION,
            "record_id": rid,
            "case": case,
            "kind": "brain",
            "condition": condition,
            "history": history,
            "phase": phase,
            "compound": compound,
            "seed": seed,
            "duration_ms": self.protocol["duration_ms"],
            "wall_s": result.wall_s,
            "inputs": actual_inputs,
            "effective_neuron_inputs": effective,
            "stimulated": result.stimulated,
            "input_overlap_policy": "last positive group overwrites neuron rate",
            "memory_before": before,
            "memory_after": after,
            "learn": learn,
            "reinforcement": reinforcement,
            "learning": learning,
            "summary": summary,
            "valence": valence,
            "populations": populations,
            "raw": {
                "path": str(relative),
                "sha256": sha256_file(self.output / relative),
                "count_sha256": array_sha256(sparse_idx, counts),
            },
            "vnc": self._vnc_artifact(vnc_result, rid) if vnc_result is not None else None,
            "language": {
                "planner": "off",
                "narrator": "off",
                "reader": "off",
                "user_text": None,
                "accepted_plan": None,
                "narrator_output": None,
                "reader_output": None,
            },
            "metadata": metadata or {},
        }
        return self._append(record)

    def odor_inputs(self, compound):
        return self.door.stim(compound, scale=self.protocol["odor_scale"]) if compound else []

    def sensory_inputs(self, condition):
        from .engine import StimInput
        from .catalog import STIMULI

        out = []
        for row in condition["stimuli"]:
            side = None if row["side"] == "both" else row["side"]
            ids = STIMULI[row["key"]].resolve(side=side)
            if not ids:
                raise ValueError(f"Missing stimulus neurons: {row['key']}/{side}")
            out.append(StimInput(ids, row["rate_hz"], f"{row['key']}/{row['side']}"))
        return out

    def run_case(self, case, seeds):
        if case == "GF":
            return self.run_gf(seeds)
        if case == "A":
            for seed in seeds:
                self.mb.reset()
                for j, compound in enumerate(self.protocol["odors"]):
                    self.brain_record(
                        case="A",
                        condition=f"odor{j + 1}",
                        compound=compound,
                        inputs=self.odor_inputs(compound),
                        seed=seed,
                        metadata={"glomerular_profile": self.door.glomerular(compound)},
                    )
        elif case in ("B", "smoke"):
            panel = (
                self.protocol["sensory"]
                if case == "B"
                else self.protocol["sensory"][:4] + self.protocol["sensory"][-1:]
            )
            for seed in seeds:
                self.mb.reset()
                for condition in panel:
                    self.brain_record(
                        case=case,
                        condition=condition["name"],
                        inputs=self.sensory_inputs(condition),
                        seed=seed,
                        with_vnc=True,
                        metadata={"condition_spec": condition},
                    )
        elif case == "C":
            for seed in seeds:
                checkpoint = self.output / "checkpoints" / f"A-{seed}.npz"
                for step in memory_schedule(self.protocol, seed):
                    action = step["action"]
                    if action == "reset":
                        self.mb.reset()
                    elif action == "checkpoint":
                        self.mb.save(checkpoint)
                    elif action == "reload":
                        self.mb.load(checkpoint)
                    else:
                        compound = step["compound"]
                        inputs = self.odor_inputs(compound)
                        reinf = step["reinforcement"]
                        if reinf != "none":
                            fn = self.mb.reward if reinf == "reward" else self.mb.punishment
                            inputs.append(fn(self.protocol["reinforcement_rate_hz"]))
                        label = (
                            f"odor{self.protocol['memory_compounds'].index(compound) + 1}"
                            if compound
                            else "dopamine"
                        )
                        label += f"-{step['history']}-{step['phase']}"
                        self.brain_record(
                            case="C",
                            condition=label,
                            compound=compound,
                            inputs=inputs,
                            seed=step["seed"],
                            history=step["history"],
                            phase=step["phase"],
                            learn=action == "train",
                            reinforcement=reinf,
                            metadata=step,
                        )
        else:
            raise ValueError(f"Unknown case: {case}")

    def run_gf(self, seeds):
        g = self.protocol["gf"]
        vnc = self.vnc
        selected = sorted(
            {int(i) for side in ("L", "R") for i in vnc.by_type.get((g["cell_type"], side), [])}
        )
        if not selected:
            raise ValueError("Required GF driver type unavailable")
        rng = np.random.default_rng(g["panel_seed"])
        pool = np.setdiff1d(np.arange(vnc.n_drv), selected)
        panels = [selected] + [
            rng.choice(pool, len(selected), replace=False).tolist()
            for _ in range(g["random_panels"])
        ]
        for seed in seeds:
            for j, panel in enumerate(panels):
                self.counter += 1
                condition = "GF" if j == 0 else f"random{j}"
                rid = f"{self.counter:04d}-GF-{condition}-{seed}"
                rates = np.zeros(vnc.n_drv, dtype=np.float32)
                rates[panel] = g["rate_hz"]
                vr = vnc.run(rates, duration_ms=g["duration_ms"], seed=seed)
                (self.output / "raw").mkdir(exist_ok=True)
                artifact = self._vnc_artifact(vr, rid)
                motors = artifact["all_motor_rates"]
                targets = [r for r in motors if r["type"] == g["target_motor_type"]]
                max_target = max((r["rate_hz"] for r in targets), default=None)
                artifact["target"] = {
                    "type": g["target_motor_type"],
                    "n": len(targets),
                    "max_rate_hz": max_target,
                    "best_rate_rank": 1 + sum(r["rate_hz"] > max_target for r in motors)
                    if max_target is not None
                    else None,
                    "ranking_note": "Competition rank by firing rate; not spike timing.",
                }
                self._append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_id": rid,
                        "case": "GF",
                        "kind": "vnc_direct",
                        "condition": condition,
                        "seed": seed,
                        "duration_ms": g["duration_ms"],
                        "wall_s": vr.wall_s,
                        "driver_indices": panel,
                        "driver_rate_hz": g["rate_hz"],
                        "vnc": artifact,
                        "language": {"planner": "off", "narrator": "off", "reader": "off"},
                    }
                )


def summarize_run(output):
    output = Path(output)
    records = [
        json.loads(line) for line in (output / "records.jsonl").read_text().splitlines() if line
    ]
    rows = []
    for record in records:
        row = {
            k: record.get(k)
            for k in (
                "record_id",
                "case",
                "condition",
                "history",
                "phase",
                "compound",
                "seed",
                "duration_ms",
                "wall_s",
            )
        }
        row["valence_score"] = record.get("valence", {}).get("score")
        row["memory_strength"] = record.get("memory_after", {}).get("strength")
        row["synapses_changed"] = (record.get("learning") or {}).get("n_synapses_changed")
        for name, population in record.get("populations", {}).items():
            row[f"{name}_fraction_active"] = population["fraction_active"]
            row[f"{name}_mean_rate_hz"] = population["mean_rate_hz"]
        for name, readout in record.get("summary", {}).get("behaviours", {}).items():
            row[f"readout_{name}_max_hz"] = readout["max_rate_hz"]
        if record.get("case") == "GF":
            row["TTMn_max_rate_hz"] = record["vnc"]["target"]["max_rate_hz"]
            row["TTMn_best_rate_rank"] = record["vnc"]["target"]["best_rate_rank"]
        rows.append(row)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    memory_comparisons = []
    for r in records:
        if r.get("case") != "C" or r.get("phase") not in ("post", "reload"):
            continue
        naive = next(
            (
                n
                for n in records
                if n.get("case") == "C"
                and n.get("history") == "naive"
                and n.get("compound") == r["compound"]
                and n["seed"] == r["seed"]
            ),
            None,
        )
        original = next(
            (
                n
                for n in records
                if n.get("case") == "C"
                and n.get("history") == "A"
                and n.get("phase") == "post"
                and n.get("compound") == r["compound"]
                and n["seed"] == r["seed"]
            ),
            None,
        )
        memory_comparisons.append(
            {
                "record_id": r["record_id"],
                "history": r["history"],
                "compound": r["compound"],
                "seed": r["seed"],
                "valence_score": r["valence"]["score"],
                "delta_from_naive": r["valence"]["score"] - naive["valence"]["score"]
                if naive
                else None,
                "reload_equal_counts": r["raw"]["count_sha256"] == original["raw"]["count_sha256"]
                if r["phase"] == "reload" and original
                else None,
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_records": len(records),
        "case_counts": {
            c: sum(r["case"] == c for r in records) for c in sorted({r["case"] for r in records})
        },
        "memory_comparisons": memory_comparisons,
        "rows": rows,
        "scope": "Descriptive simulation outputs. No significance or biological replication claim.",
    }
    write_json(output / "summary.json", summary)
    return summary


def replay_record(output, record_id, *, write=True):
    """Repeat stored exact inputs and memory; requires no planner, resolver or narrator."""
    output = Path(output)
    records = [
        json.loads(line) for line in (output / "records.jsonl").read_text().splitlines() if line
    ]
    record = next((r for r in records if r["record_id"] == record_id), None)
    if record is None:
        raise KeyError(record_id)
    if record["kind"] != "brain":
        raise ValueError("Replay currently supports brain records only")
    from .engine import Brain
    from .mb import MushroomBody
    from . import corrections

    manifest = json.loads((output / "manifest.json").read_text())
    changed = [
        name
        for name, value in manifest["files"].items()
        if not (ROOT / name).exists() or sha256_file(ROOT / name) != value["sha256"]
    ]
    if changed:
        raise ValueError(f"Recorded source/data hashes differ; replay refused: {changed}")
    if sha256_file(output / record["raw"]["path"]) != record["raw"]["sha256"]:
        raise ValueError("Original raw artifact checksum mismatch")
    brain = corrections.apply(Brain())
    mb = MushroomBody(brain)
    checkpoint = output / record["memory_before"]["path"]
    z = np.load(checkpoint, allow_pickle=False)
    if array_sha256(z["syn_idx"], z["mem"]) != record["memory_before"]["array_sha256"]:
        raise ValueError("Memory snapshot checksum mismatch")
    mb.load(checkpoint)
    result = brain.run(
        deserialize_inputs(record["inputs"]), duration_ms=record["duration_ms"], seed=record["seed"]
    )
    idx = np.asarray(sorted(result.counts), dtype=np.int32)
    counts = np.asarray([result.counts[int(i)] for i in idx], dtype=np.int32)
    digest = array_sha256(idx, counts)
    original = np.load(output / record["raw"]["path"], allow_pickle=False)
    spike_times = np.concatenate([result.trains[int(i)] for i in idx]) if len(idx) else np.zeros(0)
    equal_times = spike_times.shape == original["spike_times_s"].shape and np.allclose(
        spike_times, original["spike_times_s"], atol=1e-9, rtol=0
    )
    report = {
        "record_id": record_id,
        "seed": record["seed"],
        "llm_calls": 0,
        "equal_sparse_counts": digest == record["raw"]["count_sha256"],
        "original_count_sha256": record["raw"]["count_sha256"],
        "replay_count_sha256": digest,
        "equal_spike_times_atol_1e9_s": bool(equal_times),
        "wall_s": result.wall_s,
    }
    if record.get("vnc"):
        from .vnc import Vnc

        vr = Vnc().run_from_brain(
            result, duration_ms=record["vnc"]["duration_ms"], seed=record["seed"]
        )
        previous = np.load(output / record["vnc"]["path"], allow_pickle=False)
        report["equal_vnc_rates"] = bool(np.array_equal(vr.rates, previous["rates_hz"]))
        report["equal_vnc_driver_rates"] = bool(
            np.array_equal(vr.driver_rates, previous["driver_rates_hz"])
        )
    if write:
        write_json(output / f"replay-{record_id}.json", report)
    return report


def verify_artifacts(output):
    """Cheap integrity check for a completed or interrupted case run."""
    output = Path(output)
    records = [
        json.loads(line) for line in (output / "records.jsonl").read_text().splitlines() if line
    ]
    seen, checked, errors = set(), set(), []
    for record in records:
        rid = record["record_id"]
        if rid in seen:
            errors.append(f"Duplicate record id: {rid}")
        seen.add(rid)
        for key in ("raw", "vnc"):
            artifact = record.get(key)
            if artifact:
                path = output / artifact["path"]
                if not path.exists() or sha256_file(path) != artifact["sha256"]:
                    errors.append(f"Corrupt/missing artifact: {artifact['path']}")
        for key in ("memory_before", "memory_after"):
            snapshot = record.get(key)
            if snapshot and snapshot["path"] not in checked:
                checked.add(snapshot["path"])
                path = output / snapshot["path"]
                if not path.exists():
                    errors.append(f"Missing memory: {snapshot['path']}")
                else:
                    z = np.load(path, allow_pickle=False)
                    if array_sha256(z["syn_idx"], z["mem"]) != snapshot["array_sha256"]:
                        errors.append(f"Corrupt memory: {snapshot['path']}")
    result = {
        "n_records": len(records),
        "n_distinct_memories": len(checked),
        "ok": not errors,
        "errors": errors,
    }
    write_json(output / "verification.json", result)
    return result
