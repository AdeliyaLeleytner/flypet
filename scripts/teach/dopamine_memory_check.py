#!/usr/bin/env python3
"""Does training change the dopamine an odour-alone trial recruits? (The surrogate assumes it does not.)

For each panel odour: compartment dopamine of an odour-alone trial at naive memory, and after three trials of the
same odour paired with reward or with punishment (same test seed). Also the resulting learning step size.
"""

from __future__ import annotations
import json, pickle, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import teach as T

DATA = ROOT / "data/teach_20260925"


def job(args):
    odour, history = args
    mb = T._F["mb"]
    mb.reset()
    for k in range(3 if history != "naive" else 0):
        T._trial(odour, history, 7000 + k, learn=True)
    res, _ = T._trial(odour, "none", 5000, learn=False)
    dop = T._dopamine(res)
    v = res.rate_vector(T._F["brain"].n)
    return (odour, history), {
        "dop": dop.tolist(),
        "dan_active": int((v[mb.DAN] > 0).sum()),
        "compartments_on": int((dop > 0).sum()),
        "dop_sum": float(dop.sum()),
    }


def main():
    panel = pickle.load(open(DATA / "surrogate.pkl", "rb"))["panel"]
    pool = T.FlyPool(24)
    out = dict(
        pool.pool.imap_unordered(
            job, [(o, h) for o in panel for h in ("naive", "reward", "punish")]
        )
    )
    pool.close()
    print(
        f"{'odour':16s} {'history':8s} {'DAN active':>10s} {'compartments':>12s} {'dopamine sum':>12s}"
    )
    for o in panel:
        for h in ("naive", "reward", "punish"):
            r = out[(o, h)]
            print(
                f"{o:16s} {h:8s} {r['dan_active']:10d} {r['compartments_on']:12d} {r['dop_sum']:12.2f}"
            )
    json.dump(
        {f"{o}|{h}": v for (o, h), v in out.items()},
        open(DATA / "dopamine_memory_check.json", "w"),
        indent=1,
    )


if __name__ == "__main__":
    main()
