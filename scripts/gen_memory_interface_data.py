#!/usr/bin/env python3
"""Generate a bounded, isolated DoOR learning-history dataset (one brain process)."""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flypet.experiments import write_json
from flypet.memory_interface_data import make_plan, run_plan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 211, 307, 401, 503, 601])
    parser.add_argument("--pairing-trials", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--split-seed", type=int, default=20260924)
    parser.add_argument("--max-pairs", type=int, help="Bounded smoke subset, recorded in the plan")
    parser.add_argument(
        "--plan",
        type=Path,
        help="Execute an already fixed plan instead of CLI generation parameters",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    plan = (
        json.loads(args.plan.read_text())
        if args.plan
        else make_plan(
            seeds=args.seeds,
            pairing_trials=args.pairing_trials,
            split_seed=args.split_seed,
            max_pairs=args.max_pairs,
        )
    )
    if args.dry_run:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Dry-run output must be new or empty")
        write_json(args.output / "plan.json", plan)
        print(
            json.dumps(
                {
                    "status": "dry_run_only",
                    "output": str(args.output),
                    **{
                        k: plan[k]
                        for k in ("expected_records", "expected_histories", "expected_examples")
                    },
                },
                indent=2,
            )
        )
    else:
        manifest = run_plan(args.output, plan)
        print(
            json.dumps(
                {
                    k: manifest[k]
                    for k in (
                        "status",
                        "actual_records",
                        "actual_histories",
                        "actual_examples",
                        "elapsed_s",
                        "pet_unchanged",
                    )
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
