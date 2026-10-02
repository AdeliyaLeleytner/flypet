#!/usr/bin/env python3
"""Pilot: does cutting KC->DAN synapses remove the unpaired-history shift in case C?

Runs the unchanged case-C memory schedule (paper protocol, seeds 11/23/47) twice on one Brain:
intact, and with every KC->DAN synapse scale set to 0. DAN outputs are already non-synaptic
(corrections.dan_modulatory), so the cut acts only through which DANs gate plasticity.
Output: <output>/{intact,kc_dan_cut}/records.jsonl in the CaseRunner format, then a valence table.
Not part of the audited paper run: no manifest, validation or source hashes are written.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet import corrections
from flypet.engine import Brain
from flypet.experiments import CaseRunner, DEFAULT_PROTOCOL, load_protocol
from flypet.mb import MushroomBody
from flypet.odor import DoorOdor

CONDITIONS = [("intact", 1.0), ("kc_dan_cut", 0.0)]


def table(records):
    valence, pulses = defaultdict(list), defaultdict(list)
    for r in records:
        if r.get("case") != "C":
            continue
        if r["phase"] in ("probe", "post", "reload"):
            valence[(r["history"], r["compound"])].append(r["valence"]["score"])
        elif r["phase"] == "train" and r["compound"] and r["history"] == "unpaired":
            pulses[r["compound"]].append(
                (r["learning"]["dan_active"], r["learning"]["n_synapses_changed"])
            )
    return valence, pulses


def summarize(output):
    runs = {
        "paper cases-v1": table(
            json.loads(l) for l in open(ROOT / "paper/results/cases-v1/records.jsonl")
        )
    }
    for cond, _ in CONDITIONS:
        path = output / cond / "records.jsonl"
        if path.exists():
            runs[cond] = table(json.loads(l) for l in open(path))
    for compound in ["ethyl acetate", "pyrrolidine"]:
        print(f"\n== {compound}: mean valence over seeds")
        print(f"{'history':12s}" + "".join(f"{k:>16s}" for k in runs))
        for history in ["naive", "A", "B", "unpaired", "A_reloaded"]:
            cells = [
                np.mean(v[(history, compound)]) if v.get((history, compound)) else np.nan
                for v, _ in runs.values()
            ]
            print(f"{history:12s}" + "".join(f"{x:>16.3f}" for x in cells))
        for name, (_, pulses) in runs.items():
            dan, syn = zip(*pulses[compound])
            print(
                f"  {name}: unpaired odor-only pulses: DAN active {min(dan)}-{max(dan)}, "
                f"synapses changed {min(syn)}-{max(syn)} (n={len(dan)})"
            )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--output", type=Path, required=True, help="New directory; one subdirectory per condition"
    )
    ap.add_argument(
        "--summarize", action="store_true", help="Only print the table from existing records"
    )
    args = ap.parse_args(argv)
    if not args.summarize:
        protocol = load_protocol(DEFAULT_PROTOCOL)
        t0 = time.time()
        brain = corrections.apply(Brain())
        mb = MushroomBody(brain)
        door = DoorOdor(protocol["odor_rate_max_hz"], protocol["odor_min_response"])
        kc_dan = np.flatnonzero(np.isin(brain.i_pre, mb.KC) & np.isin(brain._i_post, mb.DAN))
        assert np.all(brain.w_scale[kc_dan] == 1.0), (
            "a correction already rescales KC->DAN; the intact factor is not 1"
        )
        print(f"KC->DAN synapse rows: {len(kc_dan)}", flush=True)
        for cond, factor in CONDITIONS:
            brain.set_weights(kc_dan, factor)
            mb.reset()
            CaseRunner(args.output / cond, protocol, brain=brain, mb=mb, door=door).run_case(
                "C", protocol["seeds"]
            )
            print(f"done {cond} after {time.time() - t0:.0f} s", flush=True)
    summarize(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
