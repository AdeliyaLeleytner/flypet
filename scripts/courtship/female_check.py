#!/usr/bin/env python3
"""Does the female brain model (FlyWire v783 with the project's corrections) hear courtship song, care about its
rhythm, and reach the neurons of her decision? Exploratory female-connectome response checks.

  rhythm    2 s of pulse song into all 387 auditory Johnston's-organ neurons (8 ms pulses at 300 Hz), inter-pulse
            intervals 15-100 ms (D. melanogaster sings at about 35 ms), each against the same number of pulses at
            random times; spikes downstream of the ears, CB1484 (main input of vpoEN), WED, vpoEN, vpoDN, pC1.
  gain      synapses from the ears multiplied by 1, 3, 6, 10: does vpoEN ever respond?
  cva       male pheromone cVA via ORN_DA1 at 100 and 200 Hz.
  scan      every catalog stimulus for 500 ms: which reach pC1, vpoEN, vpoDN (DNp37), oviDN, DNp13?
  inputs    excitatory and inhibitory synapses onto vpoEN by presynaptic type.
"""

from __future__ import annotations
import argparse, collections, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import connectome as C, corrections
from flypet.catalog import STIMULI
from flypet.engine import Brain, StimInput

PULSE_MS = 8
READ = ("CB1484", "WED", "vpoEN", "DNp37", "pC1", "oviDN", "DNp13", "aSP-g", "DA1_lPN")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", type=Path, default=ROOT / "data/courtship_20260926/female_check.json")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ann = C.annotations()
    brain = corrections.apply(Brain())
    order = np.array(sorted(brain.flyid2i, key=brain.flyid2i.get))
    typ = np.array([str(ann.cell_type.get(int(b), "")) for b in order]).astype(str)
    sub = np.array([str(ann.cell_sub_class.get(int(b), "")) for b in order]).astype(str)
    ear_i = np.flatnonzero(np.char.startswith(typ, "JO-") & (sub == "auditory"))
    ears, earset = [int(order[i]) for i in ear_i], set(ear_i.tolist())
    groups = {p: np.flatnonzero(np.char.startswith(typ, p)) for p in READ}
    base = brain.w_scale.copy()
    out = {"ears": len(ears), "groups": {k: len(v) for k, v in groups.items()}}
    t0 = time.time()

    def readout(r):
        row = {
            p: round(sum(r.counts.get(int(i), 0) for i in v) / max(len(v), 1) / r.duration_s, 2)
            for p, v in groups.items()
        }
        row["downstream_spikes"] = int(sum(c for i, c in r.counts.items() if i not in earset))
        row["active"] = len(r.counts)
        return row

    def train(onsets, T, hz):
        x = np.zeros(T)
        for on in onsets:
            x[int(on) : int(on) + PULSE_MS] = hz
        return x

    # rhythm: regular vs random pulse timing
    T, rng, rows = 2000, np.random.default_rng(0), []
    for ipi in (15, 25, 35, 50, 75, 100):
        reg = np.arange(0, T - PULSE_MS, ipi)
        for seed in range(a.seeds):
            rnd = np.sort(
                rng.choice(np.arange(0, T - PULSE_MS, PULSE_MS + 2), size=len(reg), replace=False)
            )
            for timing, on in (("regular", reg), ("random", rnd)):
                r = brain.run_timed([(ears, train(on, T, 300.0))], T, seed=seed)
                rows.append(
                    {
                        "ipi_ms": ipi,
                        "pulses": len(reg),
                        "timing": timing,
                        "seed": seed,
                        **readout(r),
                    }
                )
        m = {
            k: np.mean(
                [x["downstream_spikes"] for x in rows if x["ipi_ms"] == ipi and x["timing"] == k]
            )
            for k in ("regular", "random")
        }
        print(
            f"rhythm IPI {ipi:3d} ms: downstream spikes regular {m['regular']:.0f}, random {m['random']:.0f}  ({time.time() - t0:.0f} s)",
            flush=True,
        )
    out["rhythm"] = rows
    # gain on the synapses from the ears
    out_syn = np.flatnonzero(np.isin(brain.i_pre, ear_i))
    rows = []
    for g in (1, 3, 6, 10):
        f = base.copy()
        f[out_syn] *= g
        brain.set_weights(np.arange(brain.n_syn), f)
        for name, x in (
            ("pulse IPI 35", train(np.arange(0, 1000 - PULSE_MS, 35), 1000, 200.0)),
            ("sine 150 Hz", np.full(1000, 150.0)),
        ):
            rows.append(
                {"gain": g, "stimulus": name, **readout(brain.run_timed([(ears, x)], 1000, seed=1))}
            )
        print(
            f"gain {g}: "
            + ", ".join(
                f"{x['stimulus']} CB1484 {x['CB1484']} vpoEN {x['vpoEN']}" for x in rows[-2:]
            ),
            flush=True,
        )
    brain.set_weights(np.arange(brain.n_syn), base)
    out["gain"] = rows
    # cVA
    cva = [int(order[i]) for i in np.flatnonzero(typ == "ORN_DA1")]
    out["cva"] = [
        {"rate_hz": hz, **readout(brain.run_timed([(cva, np.full(1000, float(hz)))], 1000, seed=1))}
        for hz in (100, 200)
    ]
    print(
        "cVA:", [(x["rate_hz"], x["DA1_lPN"], x["pC1"], x["vpoEN"]) for x in out["cva"]], flush=True
    )
    # catalog scan
    rows = []
    for k, st in STIMULI.items():
        r = brain.run([StimInput(st.resolve(), st.default_rate_hz, key=k)], duration_ms=500, seed=1)
        rows.append({"stimulus": k, **readout(r)})
    out["scan"] = rows
    print(
        "scan, stimuli reaching pC1/vpoEN/DNp37:",
        [x["stimulus"] for x in rows if x["pC1"] or x["vpoEN"] or x["DNp37"]],
        flush=True,
    )
    # inputs onto vpoEN
    v = groups["vpoEN"]
    m = np.isin(brain._i_post, v)
    exc, inh = collections.Counter(), collections.Counter()
    for pre, w in zip(brain.i_pre[m], brain.w0[m]):
        (exc if w > 0 else inh)[typ[pre] or "?"] += int(w)
    out["vpoEN_inputs"] = {
        "excitatory": sum(exc.values()),
        "inhibitory": sum(inh.values()),
        "top_excitatory": exc.most_common(8),
        "top_inhibitory": sorted(inh.items(), key=lambda x: x[1])[:8],
    }
    print(
        "vpoEN inputs:",
        out["vpoEN_inputs"]["excitatory"],
        out["vpoEN_inputs"]["inhibitory"],
        flush=True,
    )
    json.dump(out, open(a.out, "w"), indent=1)
    print("saved", a.out, f"({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
