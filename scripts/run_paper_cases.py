#!/usr/bin/env python3
"""Reproducible paper cases: --dry-run first; explicit run directory required."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypet.experiments import (
    CaseRunner,
    DEFAULT_PROTOCOL,
    environment_manifest,
    expected_counts,
    load_protocol,
    replay_record,
    protocol_sha256,
    summarize_run,
    validate_protocol,
    verify_artifacts,
    write_json,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    ap.add_argument("--output", type=Path, required=True, help="New isolated output directory")
    ap.add_argument(
        "--cases", nargs="+", choices=["A", "B", "C", "GF", "smoke"], default=["A", "B", "C", "GF"]
    )
    ap.add_argument("--seeds", nargs="+", type=int, help="Explicit override; recorded as deviation")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and export fixed plan without building networks",
    )
    ap.add_argument(
        "--replay", help="Replay this record from --output, with source/data hash verification"
    )
    ap.add_argument(
        "--summarize", action="store_true", help="Regenerate summary from existing records only"
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="Check stored raw/checkpoint checksums without simulation",
    )
    args = ap.parse_args(argv)
    if args.replay:
        report = replay_record(args.output, args.replay)
        print(json.dumps(report, indent=2))
        equal = all(value for key, value in report.items() if key.startswith("equal_"))
        return 0 if equal else 1
    if args.verify:
        report = verify_artifacts(args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    if args.summarize:
        report = summarize_run(args.output)
        print(
            json.dumps(
                {"n_records": report["n_records"], "case_counts": report["case_counts"]}, indent=2
            )
        )
        return 0
    protocol = load_protocol(args.protocol)
    seeds = args.seeds or protocol["seeds"]
    if len(set(seeds)) != len(seeds):
        ap.error("--seeds must be unique")
    if len(set(args.cases)) != len(args.cases):
        ap.error("--cases must be unique")
    need_vnc = bool(set(args.cases) & {"B", "GF", "smoke"})
    validation = validate_protocol(protocol, require_vnc=need_vnc)
    plan = {
        "protocol": protocol,
        "cases": args.cases,
        "seeds": seeds,
        "protocol_sha256": protocol_sha256(protocol),
        "expected_records": expected_counts(protocol, args.cases, seeds),
        "validation": validation,
    }
    if args.dry_run:
        if args.output.exists() and any(args.output.iterdir()):
            ap.error("Dry-run output must be empty to avoid overwriting a plan")
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "output": str(args.output),
                    "expected_records": plan["expected_records"],
                },
                indent=2,
            )
        )
        return 0
    runner = CaseRunner(args.output, protocol)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "validation.json", validation)
    manifest = environment_manifest(protocol, args.cases, seeds)
    manifest["status"] = "running"
    write_json(args.output / "manifest.json", manifest)
    try:
        runner.initialize(need_brain=any(c != "GF" for c in args.cases), need_vnc=need_vnc)
        for case in args.cases:
            runner.run_case(case, seeds)
        summary = summarize_run(args.output)
        expected = sum(plan["expected_records"].values())
        if summary["n_records"] != expected:
            raise AssertionError(f"Expected {expected} records, got {summary['n_records']}")
        manifest["status"] = "complete"
        manifest["actual_records"] = summary["n_records"]
    except BaseException as exc:
        manifest["status"] = "failed_or_interrupted"
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        if (args.output / "records.jsonl").exists():
            summarize_run(args.output)
        raise
    finally:
        write_json(args.output / "manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
