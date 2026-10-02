"""Does the bridge reproduce known descending-neuron -> muscle pathways, or does anything loud win?

Test 1  giant fibre. DNp01 is the giant fibre; it drives the tergotrochanteral (jump) motor neuron
        directly and the dorsal longitudinal wing muscles through the peripherally synapsing
        interneuron (Koto et al. 1981, Allen et al. 2006). Predicted top motor neurons: TTMn, DLMn.
Test 2  control. The same rate handed to a random pair of descending neurons, repeated, to see how
        often TTMn lands that high by chance.
Test 3  specificity. Drive DNp01 / MDN (backward walking) / DNp09 (forward walking) separately and
        compare the muscle profiles.
"""

from __future__ import annotations

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
import numpy as np
import pandas as pd
from flypet.vnc import Vnc

RATE = 240.0
DUR = 300


def drive(vnc, types, rate=RATE):
    r = np.zeros(vnc.n_drv, dtype=np.float32)
    hit = []
    for t in types:
        for side in ("L", "R"):
            idx = vnc.by_type.get((t, side))
            if idx is not None:
                r[idx] = rate
                hit += list(idx)
    return r, hit


def motor_table(vr):
    mn = vr.nodes.superclass.to_numpy() == "vnc_motor"
    return pd.DataFrame(
        {
            "тип": vr.nodes.type.to_numpy()[mn],
            "мышца": vr.nodes.muscle.to_numpy()[mn],
            "сторона": vr.nodes.side.to_numpy()[mn],
            "Гц": vr.rates[mn],
        }
    )


def rank_of(tbl, name):
    t = tbl.sort_values("Гц", ascending=False).reset_index(drop=True)
    pos = t.index[t["тип"] == name]
    return (int(pos[0]) + 1, float(t.loc[pos[0], "Гц"])) if len(pos) else (None, 0.0)


def main():
    vnc = Vnc()
    print(f"ВНЦ: {vnc.n} нейронов, мотонейронов {(vnc.nodes.superclass == 'vnc_motor').sum()}\n")

    # --- 1. гигантское волокно ---------------------------------------------------------------
    r, hit = drive(vnc, ["DNp01"])
    print(f"=== гигантское волокно DNp01: {len(hit)} нейрона при {RATE:.0f} Гц")
    vr = vnc.run(r, duration_ms=DUR, seed=11)
    tbl = motor_table(vr)
    print(tbl.sort_values("Гц", ascending=False).head(10).to_string(index=False))
    for name in ("TTMn", "DLMn a, b", "DLMn c-f"):
        pos, hz = rank_of(tbl, name)
        print(f"    {name:12s} место {pos} из {len(tbl)}, {hz:.0f} Гц")
    ttm_real = rank_of(tbl, "TTMn")[1]

    # --- 2. контроль: тот же вход случайной паре нисходящих -----------------------------------
    rng = np.random.default_rng(0)
    pairs = [p for p in vnc.drivers.index.to_numpy()]
    ttm_null, rank_null = [], []
    N = 25
    print(f"\n=== контроль: {RATE:.0f} Гц случайной паре нисходящих, {N} повторов")
    for k in range(N):
        rr = np.zeros(vnc.n_drv, dtype=np.float32)
        rr[rng.choice(pairs, size=len(hit), replace=False)] = RATE
        v2 = vnc.run(rr, duration_ms=DUR, seed=11)
        t2 = motor_table(v2)
        p, hz = rank_of(t2, "TTMn")
        ttm_null.append(hz)
        rank_null.append(p)
    ttm_null = np.array(ttm_null)
    print(
        f"    TTMn при случайном входе: медиана {np.median(ttm_null):.0f} Гц, "
        f"макс {ttm_null.max():.0f} Гц, при DNp01 {ttm_real:.0f} Гц"
    )
    print(
        f"    доля повторов, где случайный вход дал TTMn >= {ttm_real:.0f} Гц: "
        f"{(ttm_null >= ttm_real).mean():.0%}"
    )
    print(
        f"    место TTMn при случайном входе: медиана {np.median([x for x in rank_null if x]):.0f}"
    )

    # --- 3. специфичность ----------------------------------------------------------------------
    print("\n=== профили мышц для разных нисходящих")
    profiles = {}
    for label, types in [
        ("DNp01 побег", ["DNp01"]),
        ("MDN назад", ["MDN"]),
        ("DNp09 вперёд", ["DNp09"]),
        ("DNa02 поворот", ["DNa02"]),
    ]:
        rr, hh = drive(vnc, types)
        if not hh:
            print(f"    {label}: нет в MaleCNS")
            continue
        v = vnc.run(rr, duration_ms=DUR, seed=11)
        t = motor_table(v)
        prof = t.groupby("мышца")["Гц"].mean()
        profiles[label] = prof
        top = t.sort_values("Гц", ascending=False).head(3)
        print(
            f"    {label:15s} ({len(hh)} нейр.) топ: "
            + ", ".join(f"{a} {b:.0f}Гц" for a, b in zip(top["тип"], top["Гц"]))
        )
    P = pd.DataFrame(profiles).fillna(0)
    print("\n    средняя частота мотонейронов по группам мышц (Гц):")
    print(P.round(1).to_string())
    print("\n    корреляция профилей между командами:")
    print(P.corr().round(2).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
