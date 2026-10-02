#!/usr/bin/env python3
"""Choose a teaching panel, measure it, run random protocols on the whole-brain fly, fit and validate the surrogate.

Panel rule (fixed before looking at any teaching result): keep candidates whose naive MBON valence is stable
across the three evaluation seeds (SD < 0.05) with at least 40 spikes in valence-contributing MBONs; take the
highest-drive odour of each chemical class (acetate ester, alcohol, ketone, aldehyde, acid, terpene), and add
the candidate with the largest Kenyon-cell Jaccard overlap to the ester as its hard-discrimination partner.
The first screen chose by drive alone and gave five esters; this rule replaces it for chemical diversity.
"""

from __future__ import annotations
import argparse, json, pickle, sys, time
from itertools import combinations
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet.teach import REINFORCEMENTS, FlyPool
from flypet.teach_surrogate import Surrogate

OUT = ROOT / "data/teach_20260925"
CANDIDATES = [
    "3-octanol",
    "4-methylcyclohexanol",
    "ethyl acetate",
    "methyl acetate",
    "isoamyl acetate",
    "pentyl acetate",
    "ethyl butyrate",
    "1-hexanol",
    "1-octen-3-ol",
    "benzaldehyde",
    "2-heptanone",
    "hexanal",
    "linalool",
    "acetic acid",
    "geraniol",
    "butyl acetate",
]
CLASSES = {
    "ester": ["ethyl acetate", "pentyl acetate", "butyl acetate", "ethyl butyrate"],
    "alcohol": ["3-octanol", "1-hexanol", "1-octen-3-ol", "4-methylcyclohexanol"],
    "ketone": ["2-heptanone"],
    "aldehyde": ["hexanal", "benzaldehyde"],
    "acid": ["acetic acid"],
    "terpene": ["linalool", "geraniol"],
}
EVAL_SEEDS, TRAIN_SEED = (101, 102, 103), 5000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--n-fit", type=int, default=60)
    ap.add_argument("--n-val", type=int, default=30)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pool = FlyPool(a.workers)
    spec, kc_index = pool.spec()
    from flypet.odor import DoorOdor

    door = DoorOdor()
    cands = [
        c
        for c in CANDIDATES
        if c.lower() in door.name2key and door.name2key[c.lower()] in door.resp.index
    ]
    print(f"{len(cands)} candidates in DoOR: {cands}", flush=True)

    # 1. screen candidates with naive probes
    probe = pool.measure([((o, s), o, "none", s) for o in cands for s in EVAL_SEEDS])
    contrib = spec.sign != 0
    stats = {}
    for o in cands:
        vals = []
        for s in EVAL_SEEDS:
            r = probe[(o, s)]["mbon"]
            ap_, av_ = r[spec.sign > 0].sum(), r[spec.sign < 0].sum()
            vals.append((ap_ - av_) / (ap_ + av_ + 1))
        spikes = np.mean([probe[(o, s)]["mbon"][contrib].sum() * 0.25 for s in EVAL_SEEDS])
        kc = set(np.flatnonzero(np.mean([probe[(o, s)]["kc"] for s in EVAL_SEEDS], 0) > 0))
        stats[o] = {
            "valence_mean": float(np.mean(vals)),
            "valence_sd": float(np.std(vals)),
            "contributing_spikes": float(spikes),
            "kc": kc,
        }
    ok = [
        o for o in cands if stats[o]["valence_sd"] < 0.05 and stats[o]["contributing_spikes"] >= 40
    ]
    jac = {
        (x, y): len(stats[x]["kc"] & stats[y]["kc"]) / max(1, len(stats[x]["kc"] | stats[y]["kc"]))
        for x, y in combinations(ok, 2)
    }
    panel = []
    for cls, members in CLASSES.items():
        live = [o for o in members if o in ok]
        if live:
            panel.append(max(live, key=lambda o: stats[o]["contributing_spikes"]))
    ester = panel[0]
    partner = max(
        (o for o in ok if o not in panel), key=lambda o: jac.get((ester, o), jac.get((o, ester), 0))
    )
    panel.insert(1, partner)
    pair = (ester, partner)
    print(
        "screen:",
        {
            o: (
                round(stats[o]["valence_mean"], 3),
                round(stats[o]["valence_sd"], 3),
                round(stats[o]["contributing_spikes"]),
            )
            for o in cands
        },
        flush=True,
    )
    print(
        f"panel: {panel}; most similar pair {pair} Jaccard {jac.get(pair, float('nan')):.3f}",
        flush=True,
    )
    overlaps = {f"{x} | {y}": round(v, 3) for (x, y), v in jac.items() if x in panel and y in panel}

    # 2. naive-memory trials for every action and training step, and naive probes of the panel
    trials = pool.measure(
        [
            ((o, r, TRAIN_SEED + k), o, r, TRAIN_SEED + k)
            for o in panel
            for r in REINFORCEMENTS
            for k in range(a.budget)
        ]
    )
    sur = Surrogate(
        spec,
        kc_index,
        {k: v["kc"] for k, v in trials.items()},
        {k: v["dop"] for k, v in trials.items()},
        {(o, s): probe[(o, s)]["kc"] for o in panel for s in EVAL_SEEDS},
        {(o, s): probe[(o, s)]["mbon"] for o in panel for s in EVAL_SEEDS},
    )

    # 3. random protocols on the real fly: half uniform, half focused (one or two actions repeated)
    rng = np.random.default_rng(0)
    acts = [(o, r) for o in panel for r in REINFORCEMENTS]
    protos = {}
    for i in range(a.n_fit + a.n_val):
        if i % 2 == 0:
            p = [acts[j] for j in rng.integers(len(acts), size=a.budget)]
        else:
            pick = [acts[j] for j in rng.choice(len(acts), size=rng.integers(1, 3), replace=False)]
            p = [pick[j % len(pick)] for j in range(a.budget)]
        protos[f"ep{i:03d}"] = p
    print(f"running {len(protos)} random protocols on the whole-brain fly", flush=True)
    eps = pool.episodes_full(
        [(k, p, panel, TRAIN_SEED, EVAL_SEEDS) for k, p in protos.items()], log_every=10
    )
    pool.close()
    keys = sorted(eps)
    fit, val = keys[: a.n_fit], keys[a.n_fit :]
    sur.fit_slopes([(eps[k]["mem"], eps[k]["rates"]) for k in fit], panel, EVAL_SEEDS)

    # 4. validation: surrogate vs real valence change on held-out protocols, per odour
    naive = {o: stats[o]["valence_mean"] for o in panel}
    true_d, sur_d, zero_d = [], [], []
    for k in val:
        sv, _ = sur.episode(protos[k], panel, TRAIN_SEED, EVAL_SEEDS)
        for o in panel:
            true_d.append(eps[k]["valence"][o] - naive[o])
            sur_d.append(sv[o] - naive[o])
    true_d, sur_d = np.array(true_d), np.array(sur_d)
    r2 = 1 - np.sum((true_d - sur_d) ** 2) / np.sum((true_d - true_d.mean()) ** 2)
    report = {
        "panel": panel,
        "naive_valence": naive,
        "screen": {o: {k: v for k, v in s.items() if k != "kc"} for o, s in stats.items()},
        "kc_jaccard_panel": overlaps,
        "budget": a.budget,
        "n_fit": len(fit),
        "n_val": len(val),
        "validation": {
            "r": float(np.corrcoef(true_d, sur_d)[0, 1]),
            "r2": float(r2),
            "mae": float(np.mean(np.abs(true_d - sur_d))),
            "true_change_sd": float(true_d.std()),
            "sign_agreement_when_true_abs_gt_0.15": float(
                np.mean(
                    np.sign(true_d[np.abs(true_d) > 0.15]) == np.sign(sur_d[np.abs(true_d) > 0.15])
                )
            ),
        },
        "wall_s": round(time.time() - t0, 1),
    }
    json.dump(report, open(OUT / "surrogate_report.json", "w"), indent=1)
    with open(OUT / "surrogate.pkl", "wb") as f:
        pickle.dump(
            {
                "spec": spec,
                "kc_index": kc_index,
                "trial_kc": sur.trial_kc,
                "trial_dop": sur.trial_dop,
                "probe_kc": sur.probe_kc,
                "probe_mbon": sur.probe_mbon,
                "slopes": sur.slopes,
                "panel": panel,
                "naive": naive,
                "budget": a.budget,
                "eval_seeds": EVAL_SEEDS,
                "train_seed": TRAIN_SEED,
            },
            f,
        )
    with open(OUT / "true_episodes.pkl", "wb") as f:
        pickle.dump({k: {"actions": protos[k], "valence": eps[k]["valence"]} for k in keys}, f)
    print(json.dumps(report["validation"], indent=1), f"\nwall {report['wall_s']} s", flush=True)


if __name__ == "__main__":
    main()
