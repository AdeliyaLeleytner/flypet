#!/usr/bin/env python3
"""Fixed language panel. Default: dry-run. Use --run only for live model calls."""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.experiments import (
    array_sha256,
    deserialize_inputs,
    effective_inputs,
    jsonable,
    protocol_sha256,
    serialize_inputs,
    sha256_file,
    write_json,
)

DEFAULT_PANEL = ROOT / "paper/cases/language_panel.json"


def load_panel(path=DEFAULT_PANEL):
    panel = json.loads(Path(path).read_text())
    cases = panel["cases"]
    if panel["schema_version"] != 1 or not 0 < len(cases) <= 20:
        raise ValueError("Unsupported/oversized language panel")
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Duplicate case IDs")
    if panel["seed"] != 11 or panel["duration_ms"] != 250:
        raise ValueError("This fixed protocol requires seed 11 and 250 ms")
    if panel["maximum_llm_calls"] > 45:
        raise ValueError("Language call ceiling exceeds the agreed protocol")
    for case in cases:
        if Path(case["id"]).name != case["id"] or case["kind"] not in (
            "natural_language",
            "controlled_memory_query",
        ):
            raise ValueError(f"Invalid case: {case['id']}")
    return panel


def source_checkpoints(physics_run, panel):
    """Select content-verified pre-probe memory; labels are artifact metadata only."""
    root = Path(physics_run).resolve()
    records = [
        json.loads(line) for line in (root / "records.jsonl").read_text().splitlines() if line
    ]
    out = {}
    for case in panel["cases"]:
        if case["kind"] != "controlled_memory_query":
            continue
        history = case["history"]
        matches = [
            r
            for r in records
            if r.get("case") == "C"
            and r.get("history") == history
            and r.get("phase") == ("probe" if history == "naive" else "post")
            and r.get("compound") == case["compound"]
            and r["seed"] == panel["seed"]
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected one source probe for {case['id']}; found {len(matches)}")
        record = matches[0]
        if record["duration_ms"] != panel["duration_ms"]:
            raise ValueError("Memory source duration differs from the language protocol")
        snapshot = record["memory_before"]
        path = (root / snapshot["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Memory source is outside the physics run")
        with np.load(path, allow_pickle=False) as z:
            actual = array_sha256(z["syn_idx"], z["mem"])
        if actual != snapshot["array_sha256"]:
            raise ValueError(f"Source memory checksum mismatch: {case['id']}")
        out[case["id"]] = {
            "physics_run": str(root),
            "record_id": record["record_id"],
            "history": history,
            "compound": case["compound"],
            "seed": record["seed"],
            "path": str(path),
            "array_sha256": actual,
            "expected_count_sha256": record["raw"]["count_sha256"],
            "inputs": record["inputs"],
        }
    for history in ("naive", "A", "B"):
        hashes = {s["array_sha256"] for s in out.values() if s["history"] == history}
        if len(hashes) != 1:
            raise ValueError(f"Memory source probes do not share a checkpoint: {history}")
    return out


def accept_plan(normalized, case, panel):
    """Explicit protocol interventions, separately recorded from planner accuracy."""
    accepted = deepcopy(normalized)
    interventions = [
        {
            "field": "duration_ms",
            "from": accepted.get("duration_ms"),
            "to": panel["duration_ms"],
            "reason": "fixed protocol duration",
        }
    ]
    accepted["duration_ms"] = panel["duration_ms"]
    if case["kind"] == "controlled_memory_query":
        replacement = {
            "is_stimulus": True,
            "stimuli": [],
            "channels": [],
            "odor_text": case["compound"],
            "odor_compound": case["compound"],
            "odor_scale": 1.0,
            "reinforcement": "none",
        }
        for field, value in replacement.items():
            interventions.append(
                {
                    "field": field,
                    "from": accepted.get(field),
                    "to": value,
                    "reason": "controlled exact-chemical memory query; not planner accuracy",
                }
            )
        accepted.update(replacement)
    return accepted, interventions


class TraceLLM:
    """Persist each call before execution, including failures and actual model IDs."""

    def __init__(self, backend, output, max_calls=45):
        self.backend, self.output, self.max_calls = backend, Path(output), max_calls
        self.name, self.model = backend.name, backend.model
        self.count = 0
        self.case_id = None
        self.stage = None
        self.calls = []

    def complete(self, system, prompt, schema=None, max_tokens=2000):
        if self.count >= self.max_calls:
            raise RuntimeError(f"Protocol LLM-call ceiling reached ({self.max_calls})")
        self.count += 1
        relative = Path("llm") / f"{self.count:03d}-{self.case_id}-{self.stage}.json"
        trace = {
            "call_id": self.count,
            "case_id": self.case_id,
            "stage": self.stage,
            "backend": self.name,
            "requested_model": self.model,
            "system": system,
            "prompt": prompt,
            "schema": deepcopy(schema),
            "max_tokens_argument": max_tokens,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "status": "pending",
            "path": str(relative),
        }
        write_json(self.output / relative, trace)
        self.calls.append(trace)
        self.backend.last_model_ids = []
        self.backend.last_usage = {}
        started = time.monotonic()
        try:
            response = self.backend.complete(system, prompt, schema=schema, max_tokens=max_tokens)
            trace.update(status="success", response=deepcopy(response))
            return response
        except Exception as exc:
            trace.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            trace.update(
                wall_s=time.monotonic() - started,
                actual_model_ids=list(getattr(self.backend, "last_model_ids", []) or []),
                usage=deepcopy(getattr(self.backend, "last_usage", {}) or {}),
            )
            write_json(self.output / relative, trace)


def memory_snapshot(mb, output):
    digest = array_sha256(mb.syn_idx, mb.mem)
    relative = Path("memory") / f"{digest}.npz"
    path = Path(output) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        np.savez_compressed(path, syn_idx=mb.syn_idx, mem=mb.mem)
    return {
        "path": str(relative),
        "array_sha256": digest,
        "strength": mb.memory_strength(),
        "log_length": len(mb.log),
    }


def save_brain_result(result, output, relative):
    path = Path(output) / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = np.asarray(sorted(result.counts), dtype=np.int32)
    counts = np.asarray([result.counts[int(i)] for i in idx], dtype=np.int32)
    times = np.concatenate([result.trains[int(i)] for i in idx]) if len(idx) else np.zeros(0)
    np.savez_compressed(
        path,
        idx=idx,
        counts=counts,
        rates_hz=counts.astype(np.float64) / result.duration_s,
        spike_times_s=times,
        offsets=np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]),
    )
    return {
        "path": str(relative),
        "sha256": sha256_file(path),
        "count_sha256": array_sha256(idx, counts),
    }


class TrialRecorder:
    """Observe actual calls, including a primary result preceding a later failure."""

    def __init__(self, fly, output):
        self.fly, self.output = fly, Path(output)
        self.original_brain_run = fly.brain.run
        self.original_vnc_run = fly.vnc.run_from_brain
        self.original_reader_describe = fly.reader.describe
        fly.brain.run = self.brain_run
        fly.vnc.run_from_brain = self.vnc_run
        fly.reader.describe = self.reader_describe
        self.case_id = None
        self.brain_runs, self.vnc_runs, self.reader_runs = [], [], []

    def start(self, case_id):
        self.case_id = case_id
        self.brain_runs, self.vnc_runs, self.reader_runs = [], [], []

    def brain_run(self, inputs, duration_ms=300, seed=None, **kwargs):
        number = len(self.brain_runs)
        prefix = Path("trials") / self.case_id / f"brain-{number}"
        entry = {
            "role": "primary" if number == 0 else "naive_counterfactual",
            "seed": seed,
            "duration_ms": duration_ms,
            "inputs": serialize_inputs(inputs),
            "effective_neuron_inputs": effective_inputs(self.fly.brain, inputs),
            "memory_before": memory_snapshot(self.fly.mb, self.output),
            "status": "pending",
        }
        self.brain_runs.append(entry)
        write_json(self.output / prefix.with_suffix(".json"), entry)
        try:
            result = self.original_brain_run(inputs, duration_ms=duration_ms, seed=seed, **kwargs)
            entry.update(
                status="success",
                raw=save_brain_result(result, self.output, prefix.with_suffix(".npz")),
                stimulated=jsonable(result.stimulated),
                wall_s=result.wall_s,
            )
            return result
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            write_json(self.output / prefix.with_suffix(".json"), entry)

    def vnc_run(self, result, duration_ms=300, seed=None):
        entry = {"seed": seed, "duration_ms": duration_ms, "status": "pending"}
        self.vnc_runs.append(entry)
        try:
            vr = self.original_vnc_run(result, duration_ms=duration_ms, seed=seed)
            relative = Path("trials") / self.case_id / "vnc.npz"
            np.savez_compressed(
                self.output / relative, rates_hz=vr.rates, driver_rates_hz=vr.driver_rates
            )
            entry.update(
                status="success",
                raw={"path": str(relative), "sha256": sha256_file(self.output / relative)},
                driven=vr.driven,
                brief=vr.brief(),
                wall_s=vr.wall_s,
            )
            return vr
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise

    def reader_describe(self, result, max_new_tokens=256):
        entry = {"status": "pending", "do_sample": False, "max_new_tokens": max_new_tokens}
        self.reader_runs.append(entry)
        try:
            relative = Path("trials") / self.case_id / "reader_features.npz"
            np.savez_compressed(self.output / relative, features=self.fly.reader.features(result))
            entry["features"] = {
                "path": str(relative),
                "sha256": sha256_file(self.output / relative),
            }
            text = self.original_reader_describe(result, max_new_tokens=max_new_tokens)
            entry.update(status="success", output=text)
            return text
        except Exception as exc:
            entry.update(status="error", error=f"{type(exc).__name__}: {exc}")
            raise


def validate(panel, physics_run=None):
    """Read files and metadata only; no Brain, VNC, reader weights, or LLM calls."""
    from flypet.projector import resolve_checkpoint
    from flypet.odor import DOOR
    from flypet import connectome as C
    from flypet.vnc import PATH_NPZ, PATH_NODES

    required = [
        C.PATH_COMP,
        C.PATH_ANN,
        C.PATH_CON_NPZ,
        PATH_NPZ,
        PATH_NODES,
        DOOR / "door_response_matrix.csv",
        DOOR / "odor.csv",
        DOOR / "door_mappings.csv",
        resolve_checkpoint(),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required runtime files missing: {missing}")
    checkpoints = source_checkpoints(physics_run, panel) if physics_run else {}
    return {
        "status": "paths_and_checkpoints_validated"
        if checkpoints
        else "paths_validated_physics_run_required",
        "required_files": [str(path) for path in required],
        "reader_checkpoint": str(resolve_checkpoint()),
        "claude_cli": shutil.which("claude"),
        "n_cases": len(panel["cases"]),
        "planned_llm_calls": panel["planned_llm_calls"],
        "maximum_llm_calls": panel["maximum_llm_calls"],
        "source_checkpoints": checkpoints,
        "model_loaded": False,
        "llm_calls": 0,
    }


def write_trial(output, record):
    write_json(Path(output) / "trials" / record["case_id"] / "result.json", record)
    with (Path(output) / "records.jsonl").open("a") as stream:
        stream.write(json.dumps(jsonable(record), ensure_ascii=False, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "case_id": record["case_id"],
                "status": record["status"],
                "llm_calls": len(record.get("llm_call_ids", [])),
            }
        ),
        flush=True,
    )


def run_panel(output, panel, validation, physics_run, panel_path=DEFAULT_PANEL):
    from flypet.chat import FlyChat
    from flypet.llm import ClaudeCLI

    output = Path(output)
    physics_manifest_path = Path(physics_run) / "manifest.json"
    physics_manifest = json.loads(physics_manifest_path.read_text())
    tracked = list((ROOT / "flypet").glob("*.py")) + [Path(__file__)]
    if Path(panel_path).resolve().is_relative_to(ROOT):
        tracked.append(Path(panel_path).resolve())
    files = dict(physics_manifest["files"])
    # Preserve a verifiable receipt of the exact physics sources used by checkpoints.
    changed = [
        name
        for name, meta in files.items()
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != meta["sha256"]
    ]
    if changed:
        raise ValueError(f"Physics source/data drift before language run: {changed}")
    files.update({str(p.relative_to(ROOT)): {"sha256": sha256_file(p)} for p in tracked})
    manifest = {
        "schema_version": 1,
        "protocol_id": panel["protocol_id"],
        "status": "initializing",
        "seed": panel["seed"],
        "duration_ms": panel["duration_ms"],
        "files": files,
        "panel_sha256": protocol_sha256(panel),
        "panel_source": str(Path(panel_path).resolve()),
        "physics_run": str(Path(physics_run).resolve()),
        "physics_manifest_sha256": sha256_file(physics_manifest_path),
        "source_checkpoints": validation["source_checkpoints"],
        "n_cases": len(panel["cases"]),
        "maximum_llm_calls": panel["maximum_llm_calls"],
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "brian2", "torch", "transformers")
        },
        "annotation_status": "not_annotated; raw material for manual fidelity audit",
    }
    write_json(output / "manifest.json", manifest)
    write_json(output / "panel.json", panel)
    try:
        fly = FlyChat(
            backend="claude-cli",
            model="sonnet",
            state_dir=output / "state",
            log=False,
            seed=panel["seed"],
            enable_door=True,
            enable_vnc=True,
            enable_reader=True,
            require_components=("door", "vnc", "reader"),
            strict_door=True,
            load_memory=False,
            persist_memory=False,
        )
        trace = TraceLLM(ClaudeCLI(model="sonnet"), output, panel["maximum_llm_calls"])
        fly._llm = trace
        recorder = TrialRecorder(fly, output)
        np.save(
            output / "brain_root_ids.npy",
            np.asarray([fly.brain.i2flyid[i] for i in range(fly.brain.n)], dtype=np.int64),
        )
        fly.vnc.nodes.to_csv(output / "vnc_nodes.csv", index=False)
        fly.vnc.drivers.to_csv(output / "vnc_drivers.csv", index=False)
        from brian2 import defaultclock, ms, prefs

        manifest.update(
            status="running",
            reader=fly.reader.checkpoint_metadata,
            reader_device=str(fly.reader.dev),
            loaded_reader_model_revision=getattr(fly.reader.llm.config, "_commit_hash", None),
            brian_target=str(prefs.codegen.target),
            dt_ms=float(defaultclock.dt / ms),
            claude_flags=trace.backend.flags,
            components=fly.component_status,
        )
        write_json(output / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(status="initialization_failed", error=f"{type(exc).__name__}: {exc}")
        for case in panel["cases"]:
            write_trial(
                output,
                {
                    "case_id": case["id"],
                    "kind": case["kind"],
                    "user_text": case["text"],
                    "status": "initialization_failed",
                    "error": manifest["error"],
                    "llm_call_ids": [],
                },
            )
        write_json(output / "manifest.json", manifest)
        return 1
    outcomes = []
    for case in panel["cases"]:
        trace.case_id = case["id"]
        start_call = trace.count
        recorder.start(case["id"])
        record = {
            "schema_version": 1,
            "case_id": case["id"],
            "kind": case["kind"],
            "user_text": case["text"],
            "seed": panel["seed"],
            "duration_ms": panel["duration_ms"],
            "status": "pending",
            "history_metadata_only": case.get("history", "naive"),
            "manual_fidelity_annotation": None,
        }
        write_json(output / "trials" / case["id"] / "result.json", record)
        stage = "reset"
        started = time.monotonic()
        try:
            fly.reset(memory=True, history=True, persist_memory=False)
            if case["kind"] == "controlled_memory_query":
                source = validation["source_checkpoints"][case["id"]]
                # Copy arrays alone: source log labels never enter narrator associations.
                copied = output / "source_memory" / f"{source['array_sha256']}.npz"
                copied.parent.mkdir(exist_ok=True)
                with np.load(source["path"], allow_pickle=False) as z:
                    np.savez_compressed(copied, syn_idx=z["syn_idx"], mem=z["mem"])
                fly.mb.load(copied)
                record["source_checkpoint"] = source
            record["memory_before"] = memory_snapshot(fly.mb, output)
            stage = trace.stage = "planner"
            normalized = fly.plan(case["text"])
            record["planner_output"] = deepcopy(trace.calls[-1].get("response"))
            record["normalized_planner_plan"] = deepcopy(normalized)
            accepted, interventions = accept_plan(normalized, case, panel)
            record.update(accepted_plan=accepted, protocol_interventions=interventions)
            if not accepted.get("is_stimulus"):
                record.update(
                    status="planner_no_stimulus", reply=accepted.get("reply_if_no_stimulus")
                )
            else:
                stage = trace.stage = "simulation_and_resolver"
                result, summary = fly.simulate(accepted, seed=panel["seed"])
                record["summary"] = summary
                if case["kind"] == "controlled_memory_query":
                    record["physical_comparison"] = {
                        "equal_source_inputs": recorder.brain_runs[0]["inputs"] == source["inputs"],
                        "equal_source_sparse_counts": recorder.brain_runs[0]["raw"]["count_sha256"]
                        == source["expected_count_sha256"],
                    }
                stage = trace.stage = "narrator"
                record["reply"] = fly.narrate(case["text"], accepted, summary)
                record["status"] = "complete_unannotated"
        except Exception as exc:
            record.update(status="error", error_stage=stage, error=f"{type(exc).__name__}: {exc}")
        finally:
            record.update(
                wall_s=time.monotonic() - started,
                brain_runs=recorder.brain_runs,
                vnc_runs=recorder.vnc_runs,
                reader_runs=recorder.reader_runs,
                llm_call_ids=list(range(start_call + 1, trace.count + 1)),
                memory_after=memory_snapshot(fly.mb, output),
            )
            write_trial(output, record)
            outcomes.append(record["status"])
    manifest.update(
        status="complete"
        if all(s == "complete_unannotated" for s in outcomes)
        else "complete_with_recorded_failures",
        outcomes={s: outcomes.count(s) for s in sorted(set(outcomes))},
        actual_llm_calls=trace.count,
        actual_model_ids=sorted(
            {m for call in trace.calls for m in call.get("actual_model_ids", [])}
        ),
        reported_cost_usd=sum(call.get("usage", {}).get("cost_usd") or 0 for call in trace.calls),
    )
    write_json(output / "manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else 1


def replay(output, case_id):
    """Replay saved physical inputs only, without constructing any language model."""
    output = Path(output)
    record = json.loads((output / "trials" / case_id / "result.json").read_text())
    primary = record["brain_runs"][0]
    manifest = json.loads((output / "manifest.json").read_text())
    if protocol_sha256(json.loads((output / "panel.json").read_text())) != manifest["panel_sha256"]:
        raise ValueError("Saved panel fingerprint mismatch")
    changed = [
        name
        for name, meta in manifest["files"].items()
        if not (ROOT / name).is_file() or sha256_file(ROOT / name) != meta["sha256"]
    ]
    if changed:
        raise ValueError(f"Source/data drift; replay refused: {changed}")
    from flypet.engine import Brain
    from flypet.mb import MushroomBody
    from flypet import corrections

    snapshot = primary["memory_before"]
    checkpoint = output / snapshot["path"]
    with np.load(checkpoint, allow_pickle=False) as z:
        if array_sha256(z["syn_idx"], z["mem"]) != snapshot["array_sha256"]:
            raise ValueError("Replay memory checksum mismatch")
    if sha256_file(output / primary["raw"]["path"]) != primary["raw"]["sha256"]:
        raise ValueError("Replay source result checksum mismatch")
    brain = corrections.apply(Brain())
    mb = MushroomBody(brain)
    mb.load(checkpoint)
    result = brain.run(
        deserialize_inputs(primary["inputs"]),
        duration_ms=primary["duration_ms"],
        seed=primary["seed"],
    )
    idx = np.asarray(sorted(result.counts), dtype=np.int32)
    counts = np.asarray([result.counts[int(i)] for i in idx], dtype=np.int32)
    times = np.concatenate([result.trains[int(i)] for i in idx]) if len(idx) else np.zeros(0)
    with np.load(output / primary["raw"]["path"], allow_pickle=False) as old:
        report = {
            "case_id": case_id,
            "llm_calls": 0,
            "equal_sparse_counts": array_sha256(idx, counts) == primary["raw"]["count_sha256"],
            "equal_spike_times_atol_1e9_s": bool(
                times.shape == old["spike_times_s"].shape
                and np.allclose(times, old["spike_times_s"], atol=1e-9, rtol=0)
            ),
        }
    if record.get("vnc_runs") and record["vnc_runs"][0]["status"] == "success":
        from flypet.vnc import Vnc

        prior = record["vnc_runs"][0]
        if sha256_file(output / prior["raw"]["path"]) != prior["raw"]["sha256"]:
            raise ValueError("VNC artifact checksum mismatch")
        vr = Vnc().run_from_brain(result, duration_ms=prior["duration_ms"], seed=prior["seed"])
        with np.load(output / prior["raw"]["path"], allow_pickle=False) as old:
            report["equal_vnc_rates"] = bool(np.array_equal(vr.rates, old["rates_hz"]))
            report["equal_vnc_driver_rates"] = bool(
                np.array_equal(vr.driver_rates, old["driver_rates_hz"])
            )
    write_json(output / f"replay-{case_id}.json", report)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--physics-run", type=Path)
    action = ap.add_mutually_exclusive_group()
    action.add_argument(
        "--run", action="store_true", help="Enable live planner, simulation, reader and narrator"
    )
    action.add_argument(
        "--dry-run", action="store_true", help="Default: validate files without model construction"
    )
    action.add_argument(
        "--replay", metavar="CASE_ID", help="Replay exact saved physical inputs; no language calls"
    )
    args = ap.parse_args(argv)
    if args.replay:
        report = replay(args.output, args.replay)
        print(json.dumps(report, indent=2))
        return 0 if all(v for k, v in report.items() if k.startswith("equal_")) else 1
    panel = load_panel(args.panel)
    if args.output.exists() and any(args.output.iterdir()):
        ap.error("Choose a new empty output directory; no run files are overwritten")
    if args.run and not args.physics_run:
        ap.error("--run requires --physics-run for predeclared naive/A/B memory checkpoints")
    validation = validate(panel, args.physics_run)
    write_json(args.output / "validation.json", validation)
    write_json(args.output / "panel.json", panel)
    if not args.run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "status": validation["status"],
                    "n_cases": validation["n_cases"],
                    "planned_llm_calls": validation["planned_llm_calls"],
                    "actual_llm_calls": 0,
                    "output": str(args.output),
                },
                indent=2,
            )
        )
        return 0
    return run_panel(args.output, panel, validation, args.physics_run, args.panel)


if __name__ == "__main__":
    raise SystemExit(main())
