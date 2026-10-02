#!/usr/bin/env python3
"""Exploratory, matched-stimulus FlyWire-female versus MaleCNS-male response panel.

This compares neural rates, not behaviour or a calibrated sex effect. Both networks
receive the same per-neuron Poisson rate and trial duration. The ``matched`` condition
uses the same number of input neurons on each side; ``native`` uses all annotated
members. Male weights use the previously documented global 0.537 scale.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import connectome as F, malecns as M, corrections
from flypet.engine import Brain, StimInput

STIMULI = {
    "looming": (
        200.0,
        lambda: F.select(cell_type=["LC4", "LPLC2"]),
        lambda: M.select("LC4", exact=True) + M.select("LPLC2", exact=True),
    ),
    "cVA": (100.0, lambda: F.select(cell_type="ORN_DA1"), lambda: M.select("ORN_DA1", exact=True)),
    "auditory": (
        200.0,
        lambda: F.select(cell_class="mechanosensory", cell_sub_class="auditory"),
        lambda: M.select("JO-A") + M.select("JO-B"),
    ),
    "moving_object": (
        100.0,
        lambda: F.select(cell_type="LC10a"),
        lambda: M.select("LC10a", exact=True),
    ),
}
SHARED_READOUTS = ("DNp01", "DNa02", "MDN", "vpoEN", "DA1_lPN")
SEEDS = (11, 23, 47)


def ids_by_type(ann, column, name):
    return [int(x) for x in ann.index[ann[column].astype(str).str.startswith(name)]]


def rate(result, brain, ids):
    return float(
        sum(result.counts.get(brain.flyid2i[x], 0) for x in ids if x in brain.flyid2i)
        / max(1, len(ids))
        / result.duration_s
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "data/courtship_20260927/response_comparison.json"
    )
    args = ap.parse_args()
    female_ann, male_ann = F.annotations(), M.annotations()
    inputs = {
        name: {"female": sorted(set(f())), "male": sorted(set(m()))}
        for name, (_, f, m) in STIMULI.items()
    }
    for name, pair in inputs.items():
        if not pair["female"] or not pair["male"]:
            raise ValueError(f"{name}: missing input neurons in one connectome")
    readouts = {
        "female": {n: ids_by_type(female_ann, "cell_type", n) for n in SHARED_READOUTS},
        "male": {n: ids_by_type(male_ann, "type", n) for n in SHARED_READOUTS},
    }
    readouts["female"]["pC1"] = ids_by_type(female_ann, "cell_type", "pC1")
    readouts["male"]["pC1"] = ids_by_type(male_ann, "type", "pC1")
    readouts["male"]["pIP10"] = ids_by_type(male_ann, "type", "pIP10")
    readouts["male"]["TTMn"] = ids_by_type(male_ann, "type", "TTMn")
    if any(not readouts[sex][n] for sex in ("female", "male") for n in SHARED_READOUTS):
        raise ValueError("a shared readout is absent from one connectome")

    data = {
        "status": "exploratory, one reconstructed animal of each sex, three stochastic seeds",
        "duration_ms": 300,
        "seeds": SEEDS,
        "female": "FlyWire v783; Shiu LIF with four flypet corrections",
        "male": "MaleCNS v1.0; same LIF parameters, global synaptic scale 0.537, dopamine fast output 0",
        "input_selection": "native: all annotated neurons; matched: seeded random subset of equal size per network",
        "rows": [],
        "group_sizes": {
            sex: {n: len(ids) for n, ids in readouts[sex].items()} for sex in ("female", "male")
        },
    }
    for sex, conn in (("female", F), ("male", M)):
        brain = Brain(conn=conn)
        if sex == "female":
            corrections.apply(brain)
        else:
            brain.set_weights(np.arange(brain.n_syn), 0.537)
        print(f"{sex}: {brain.n} neurons, {brain.n_syn} connections", flush=True)
        for name, (hz, _, _) in STIMULI.items():
            native = inputs[name][sex]
            n_match = min(len(inputs[name]["female"]), len(inputs[name]["male"]))
            rng = np.random.default_rng(20260927 + list(STIMULI).index(name))
            matched = sorted(int(x) for x in rng.choice(native, n_match, replace=False))
            for condition, ids in (("native", native), ("matched", matched)):
                for seed in SEEDS:
                    result = brain.run([StimInput(ids, hz, key=name)], duration_ms=300, seed=seed)
                    row = {
                        "sex": sex,
                        "stimulus": name,
                        "condition": condition,
                        "seed": seed,
                        "input_neurons": len(ids),
                        "input_hz_per_neuron": hz,
                        "active_neurons": len(result.counts),
                        "spikes": int(sum(result.counts.values())),
                        "readout_hz_per_neuron": {
                            n: round(rate(result, brain, group), 4)
                            for n, group in readouts[sex].items()
                        },
                    }
                    data["rows"].append(row)
                summary = {
                    n: round(
                        float(
                            np.mean(
                                [
                                    r["readout_hz_per_neuron"][n]
                                    for r in data["rows"]
                                    if r["sex"] == sex
                                    and r["stimulus"] == name
                                    and r["condition"] == condition
                                ]
                            )
                        ),
                        2,
                    )
                    for n in readouts[sex]
                }
                print(f"  {name:13s} {condition:7s} n={len(ids):3d} {summary}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    print("saved", args.out)


if __name__ == "__main__":
    main()
