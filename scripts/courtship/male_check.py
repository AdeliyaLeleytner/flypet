#!/usr/bin/env python3
"""Does the male whole-CNS model (MaleCNS v1.0 brain and nerve cord, flypet.malecns) behave like a male fly?

Five checks, each 300 ms, three seeds, at two synaptic scales: the female model's weight (1.0) and one scaled by the
ratio of synapses per neuron (FlyWire 393, MaleCNS 732: 0.537), because MaleCNS reports nearly twice the synapses
per neuron and the Shiu et al. weight was fitted on FlyWire.
  looming   LC4 + LPLC2 at 200 Hz: does vision reach the giant fibre DNp01 and the jump motor neuron TTMn, all inside
            one connectome? (the female model stops at the neck)
  DNp01     the giant fibre alone at 240 Hz: rank of TTMn among the 708 motor neurons; random descending pairs as control
  pIP10     the song command neuron at 240 Hz: ranks of the song motor neurons (ps1, hg1-4, i1, i2, b1-3, tp1-2)
  P1        the male-specific pC1 (P1) cluster at 100 Hz: does it recruit pIP10 and the song motor neurons?
  ppk       pheromone receptor neurons (ppk) at 100 Hz: does touching a female reach P1 and pIP10?
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import malecns as M
from flypet.engine import Brain, StimInput

SONG_MN = (
    "ps1 MN",
    "hg1 MN",
    "hg2 MN",
    "hg3 MN",
    "hg4 MN",
    "i1 MN",
    "i2 MN",
    "b1 MN",
    "b2 MN",
    "b3 MN",
    "tp1 MN",
    "tp2 MN",
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--controls", type=int, default=10)
    ap.add_argument("--scales", default="1.0,0.537")
    ap.add_argument("--out", type=Path, default=ROOT / "data/courtship_20260926/male_check.json")
    a = ap.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    ann = M.annotations()
    mns = ann.index[ann.superclass == "vnc_motor"].tolist()
    mn_type = ann.type.to_dict()
    p1 = [
        int(b)
        for b in ann.index[
            ann.type.str.startswith("pC1") & ann.dimorphism.str.contains("male-specific")
        ]
    ]
    dns = ann.index[ann.superclass == "descending_neuron"].tolist()
    sets = {
        "looming": M.select("LC4", exact=True) + M.select("LPLC2", exact=True),
        "DNp01": M.select("DNp01", exact=True),
        "pIP10": M.select("pIP10", exact=True),
        "P1": p1,
        "ppk": M.select(receptor="ppk"),
    }
    rates = {"looming": 200, "DNp01": 240, "pIP10": 240, "P1": 100, "ppk": 100}
    print(
        "sizes: "
        + ", ".join(f"{k} {len(v)}" for k, v in sets.items())
        + f"; motor neurons {len(mns)}",
        flush=True,
    )
    t0 = time.time()
    brain = Brain(conn=M)
    print(
        f"male brain: {brain.n} neurons, {brain.n_syn} connections, built in {time.time() - t0:.0f} s",
        flush=True,
    )
    base = brain.w_scale.copy()
    f2i = brain.flyid2i
    rng = np.random.default_rng(0)
    out = {"sizes": {k: len(v) for k, v in sets.items()}, "results": {}}

    def rate(r, ids):
        return float(np.mean([r.counts.get(f2i[i], 0) for i in ids if i in f2i])) / r.duration_s

    def mn_table(r):
        return {b: r.counts.get(f2i[b], 0) / r.duration_s for b in mns if b in f2i}

    for scale in [float(x) for x in a.scales.split(",")]:
        brain.set_weights(np.arange(brain.n_syn), base * scale)
        res = {}
        for name, ids in sets.items():
            rows = []
            for seed in range(a.seeds):
                r = brain.run(
                    [StimInput(ids, rates[name], key=name)], duration_ms=300, seed=200 + seed
                )
                mt = mn_table(r)
                ranked = sorted(mt, key=lambda b: -mt[b])
                rows.append(
                    {
                        "active": len(r.counts),
                        "spikes": int(sum(r.counts.values())),
                        "wall_s": round(r.wall_s, 1),
                        "DNp01": rate(r, sets["DNp01"]),
                        "TTMn": rate(r, M.select("TTMn", exact=True)),
                        "pIP10": rate(r, sets["pIP10"]),
                        "P1": rate(r, p1),
                        "song_MN": rate(r, [b for b in mns if mn_type[b] in SONG_MN]),
                        "TTMn_rank": min(
                            ranked.index(b) for b in M.select("TTMn", exact=True) if b in mt
                        )
                        + 1,
                        "best_song_MN_rank": min(
                            ranked.index(b) for b in mns if mn_type[b] in SONG_MN
                        )
                        + 1,
                        "top_MN": [(mn_type[b], round(mt[b], 1)) for b in ranked[:6]],
                    }
                )
            res[name] = rows
            m = {
                k: np.mean([x[k] for x in rows])
                for k in (
                    "active",
                    "DNp01",
                    "TTMn",
                    "pIP10",
                    "P1",
                    "song_MN",
                    "TTMn_rank",
                    "best_song_MN_rank",
                )
            }
            print(
                f"scale {scale} {name:8s} active {m['active']:7.0f}  DNp01 {m['DNp01']:6.1f} Hz  TTMn {m['TTMn']:6.1f} (rank {m['TTMn_rank']:.0f})  "
                f"pIP10 {m['pIP10']:6.1f}  P1 {m['P1']:5.1f}  song MN {m['song_MN']:5.1f} (best rank {m['best_song_MN_rank']:.0f})  "
                f"top {rows[0]['top_MN'][:3]}  ({time.time() - t0:.0f} s)",
                flush=True,
            )
        ctrl = []
        for k in range(a.controls):
            pair = [int(x) for x in rng.choice(dns, 2, replace=False)]
            r = brain.run([StimInput(pair, 240, key="control")], duration_ms=300, seed=300 + k)
            mt = mn_table(r)
            ranked = sorted(mt, key=lambda b: -mt[b])
            ctrl.append(
                {
                    "pair": [mn_type.get(b, "") for b in pair],
                    "TTMn": rate(r, M.select("TTMn", exact=True)),
                    "song_MN": rate(r, [b for b in mns if mn_type[b] in SONG_MN]),
                    "best_song_MN_rank": min(ranked.index(b) for b in mns if mn_type[b] in SONG_MN)
                    + 1,
                }
            )
        res["random DN pairs"] = ctrl
        print(
            f"scale {scale} random DN pairs: TTMn median {np.median([c['TTMn'] for c in ctrl]):.1f} Hz, song MN median "
            f"{np.median([c['song_MN'] for c in ctrl]):.1f} Hz, best song MN rank median {np.median([c['best_song_MN_rank'] for c in ctrl]):.0f}",
            flush=True,
        )
        if abs(scale - 0.537) < 1e-9:  # seeing and touching a female, 1 s with run_timed
            lc10a, lc10 = M.select("LC10a", exact=True), M.select("LC10")
            leg_ppk = [
                b
                for b in sets["ppk"]
                if ann.superclass[b] == "vnc_sensory" and ann.type[b].startswith("LgLG")
            ]
            song = [b for b in mns if mn_type[b] in SONG_MN]
            res["female cues, 1 s"] = []
            for name, inputs in (
                ("LC10a 100 Hz", [(lc10a, 100.0)]),
                ("all LC10 100 Hz", [(lc10, 100.0)]),
                ("leg ppk 100 Hz", [(leg_ppk, 100.0)]),
                ("LC10a + leg ppk 100 Hz", [(lc10a, 100.0), (leg_ppk, 100.0)]),
            ):
                r = brain.run_timed([(ids, np.full(1000, hz)) for ids, hz in inputs], 1000, seed=1)
                row = {
                    "cue": name,
                    "active": len(r.counts),
                    "P1": rate(r, p1),
                    "pIP10": rate(r, sets["pIP10"]),
                    "song_MN": rate(r, song),
                }
                res["female cues, 1 s"].append(row)
                print(
                    f"scale {scale} {name:24s} active {row['active']:6d}  P1 {row['P1']:5.1f} Hz  pIP10 {row['pIP10']:6.1f}  "
                    f"song MN {row['song_MN']:5.1f}",
                    flush=True,
                )
        out["results"][str(scale)] = res
        json.dump(out, open(a.out, "w"), indent=1, default=str)
    print("saved", a.out)


if __name__ == "__main__":
    main()
