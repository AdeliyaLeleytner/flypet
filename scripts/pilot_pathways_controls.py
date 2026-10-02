#!/usr/bin/env python3
"""Pilot controls for two abstract claims, on the case-B stimuli (paper protocol rates, 250 ms, seeds 11/23/47):
1. the three olfactory corrections leave the taste, grooming and escape pathways unchanged (raw Shiu rule vs corrected);
2. degree-preserving rewiring (postsynaptic targets permuted; in/out degree and presynaptic sign kept) abolishes them.
Rewiring is applied to the raw graph, then corrections are recomputed on the rewired graph. Not an audited run.
"""

from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet import corrections, analysis as A
from flypet.catalog import STIMULI
from flypet.engine import Brain, StimInput
from flypet.experiments import DEFAULT_PROTOCOL, load_protocol

protocol = load_protocol(DEFAULT_PROTOCOL)
names = {c["name"]: c for c in protocol["sensory"]}
print("sensory conditions:", list(names), flush=True)
PANEL = [
    (n, read)
    for n, read in [
        ("sugar", "proboscis_extension"),
        ("antenna_touch", "antennal_grooming"),
        ("looming", "escape_takeoff"),
    ]
    if n in names
]
assert len(PANEL) == 3, f"condition names differ: {list(names)}"


def inputs(cond):
    out = []
    for row in cond["stimuli"]:
        side = None if row["side"] == "both" else row["side"]
        out.append(
            StimInput(
                STIMULI[row["key"]].resolve(side=side),
                row["rate_hz"],
                f"{row['key']}/{row['side']}",
            )
        )
    return out


def panel(brain, tag, rows):
    for name, read in PANEL:
        for seed in protocol["seeds"]:
            r = brain.run(inputs(names[name]), duration_ms=protocol["duration_ms"], seed=seed)
            s = A.summarize(r)
            b = s["behaviours"]
            rows.append(
                {
                    "graph": tag,
                    "stimulus": name,
                    "seed": seed,
                    "readout": read,
                    "max_rate_hz": b[read]["max_rate_hz"],
                    "n_active": b[read]["n_active"],
                    "all_groups_max": {
                        k: v["max_rate_hz"]
                        for k, v in b.items()
                        if isinstance(v, dict) and "max_rate_hz" in v
                    },
                    "n_active_downstream": s["n_active_downstream"],
                }
            )
    print("done", tag, flush=True)


rows = []
t0 = time.time()
brain = Brain()
panel(brain, "raw_shiu", rows)
corrections.apply(brain)
panel(brain, "corrected", rows)
orig_post = brain._i_post.copy()
for p in (1, 2, 3):
    brain._i_post = orig_post[np.random.default_rng(p).permutation(len(orig_post))]
    brain._build()
    corrections.apply(brain)
    panel(brain, f"rewired_{p}", rows)
out = Path(sys.argv[1])
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(rows, indent=1))
print(f"total {time.time() - t0:.0f} s", flush=True)
for name, read in PANEL:
    print(f"\n{name} -> {read}: max rate Hz per seed (n active)")
    for tag in ["raw_shiu", "corrected", "rewired_1", "rewired_2", "rewired_3"]:
        xs = [r for r in rows if r["graph"] == tag and r["stimulus"] == name]
        print(f"  {tag:10s} " + ", ".join(f"{r['max_rate_hz']:.0f} ({r['n_active']})" for r in xs))
