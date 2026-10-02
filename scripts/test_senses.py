"""Прогнать все сенсорные каналы через мозг и посмотреть, что доходит до поведения."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys

sys.path.insert(0, _ROOT + "")
import numpy as np
from flypet.engine import Brain
from flypet import corrections, analysis as A
from flypet.senses import CHANNELS

b = corrections.apply(Brain())


def run(tag, stims):
    if not stims:
        print(f"{tag:30s} — стимул пустой")
        return
    r = b.run(stims, duration_ms=250, seed=4)
    s = A.summarize(r)
    beh = {
        k: v["max_rate_hz"]
        for k, v in s["behaviours"].items()
        if v["n_active"] and k not in ("other_motor", "descending_all")
    }
    n_in = sum(len(x.ids) for x in stims)
    print(
        f"{tag:30s} подано {n_in:5d} → активных {s['n_active_downstream']:5d} | {beh or 'тихо'}",
        flush=True,
    )


print("=== звук: частота ===")
for f in (30, 100, 400, 1000):
    run(f"звук {f} Гц", CHANNELS["sound"].stim(f, 1.0))
print("\n=== ветер: направление ===")
for d, n in ((0, "спереди"), (90, "справа"), (-90, "слева"), (180, "сзади")):
    run(f"ветер {n}", CHANNELS["wind"].stim(d, 1.0))
print("\n=== вкус ===")
for mod in ("sugar", "bitter", "salt"):
    for c in (0.3, 1.0):
        run(f"{mod} конц. {c}", CHANNELS["taste"].stim(mod, c))
run("сахар на лапках", CHANNELS["taste"].stim("sugar", 1.0, site="tarsi"))
run("горечь в глотке", CHANNELS["taste"].stim("bitter", 1.0, site="pharynx"))
print("\n=== температура и влажность ===")
for t in (15, 22, 30, 38):
    run(f"температура {t} °C", CHANNELS["temperature"].stim(t))
for h in (20, 60, 95):
    run(f"влажность {h} %", CHANNELS["humidity"].stim(h))
print("\n=== касание ===")
for site in ("head", "eye", "groom"):
    run(f"касание {site}", CHANNELS["touch"].stim(site, 1.0))
run("касание головы слева", CHANNELS["touch"].stim("head", 1.0, side="left"))
print("\n=== свет ===")
run("свет яркий", CHANNELS["light"].stim(1.0))
