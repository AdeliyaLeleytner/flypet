"""Пробы на измеренных запахах DoOR: 52 вещества девяти химических классов, разные концентрации."""

import sys, json, time, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from flypet.engine import Brain, StimInput
from flypet import corrections, analysis as A
from flypet.catalog import STIMULI
from flypet.odor import DoorOdor
from flypet.mb import MushroomBody

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data/odor_dataset")
OUT.mkdir(parents=True, exist_ok=True)
PER_ODOR = int(sys.argv[2]) if len(sys.argv) > 2 else 35
CLASSES = json.loads((ROOT / "paper/cases/legacy_odor_classes.json").read_text())
comp2class = {c: cl for cl, lst in CLASSES.items() for c in lst}
COMPOUNDS = sorted(comp2class)

b = corrections.apply(Brain())
mb = MushroomBody(b)
door = DoorOdor(rate_max=150.0)
other_keys = [k for k in STIMULI if not k.startswith("smell_") and k != "sugar_all_labellar"]
rng = random.Random(11)
plan = [(c, i) for c in COMPOUNDS for i in range(PER_ODOR)]
plan += [(None, i) for i in range(300)]
rng.shuffle(plan)
print(
    f"запланировано {len(plan)} проб: {len(COMPOUNDS)} веществ × {PER_ODOR} + 300 без запаха",
    flush=True,
)

buf, shard, t0, done = [], 0, time.time(), 0


def flush():
    global buf, shard
    if not buf:
        return
    np.savez_compressed(
        OUT / f"shard_{shard:04d}.npz",
        n=len(buf),
        idx=np.concatenate([s["idx"] for s in buf]),
        rate=np.concatenate([s["rate"] for s in buf]),
        offsets=np.cumsum([0] + [len(s["idx"]) for s in buf]).astype(np.int64),
        labels=np.array([json.dumps(s["label"], ensure_ascii=False) for s in buf]),
    )
    shard += 1
    buf = []


for comp, _ in plan:
    inputs, label = (
        [],
        {"compound": comp, "chem_class": comp2class.get(comp), "stimuli": [], "dopamine": None},
    )
    if comp:
        scale = rng.choice([0.5, 0.7, 1.0, 1.0, 1.3])
        st = door.stim(comp, scale=scale)
        if not st:
            continue
        inputs += st
        label["scale"] = scale
        label["glomeruli"] = {
            g: round(v, 3)
            for g, v in sorted(door.glomerular(comp).items(), key=lambda x: -x[1])[:8]
        }
    for k in rng.sample(other_keys, rng.choice([0, 0, 0, 1, 1, 2])):
        s = STIMULI[k]
        side = rng.choice([None, None, "left", "right"])
        rate = round(s.default_rate_hz * rng.uniform(0.4, 1.4))
        inputs.append(StimInput(s.resolve(side=side) or s.resolve(), rate, key=k))
        label["stimuli"].append({"key": k, "side": side or "both", "rate_hz": rate})
    if rng.random() < 0.12:
        d = rng.choice(["reward", "punish"])
        inputs.append(mb.reward() if d == "reward" else mb.punishment())
        label["dopamine"] = d
    if not inputs:
        continue
    r = b.run(inputs, duration_ms=250, seed=rng.randrange(10**6))
    v = r.rate_vector(b.n)
    nz = np.flatnonzero(v > 0)
    s = A.summarize(r)
    label["behaviour"] = A.behaviour_vector(s)
    label["valence"] = mb.valence(r).__dict__ | {"top": None}
    label["n_active"] = int(len(nz))
    label["duration_ms"] = 250
    buf.append({"idx": nz.astype(np.int32), "rate": v[nz].astype(np.float16), "label": label})
    done += 1
    if len(buf) >= 200:
        flush()
        print(f"{done}/{len(plan)}, {(time.time() - t0) / 60:.1f} мин", flush=True)
flush()
print(f"готово: {done} проб за {(time.time() - t0) / 60:.1f} мин")
