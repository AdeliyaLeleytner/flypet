"""Targeted samples for the interaction test: sugar/bitter/water alone and in pairs, plus a few other pairs.
Same format as gen_state_data.py, separate folder data/state_pairs."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, time, random, numpy as np
from pathlib import Path

sys.path.insert(0, _ROOT + "")
from flypet.engine import Brain, StimInput
from flypet import corrections, analysis as A
from flypet.catalog import STIMULI
from flypet.mb import MushroomBody

OUT = Path(_ROOT + "/data/state_pairs")
OUT.mkdir(parents=True, exist_ok=True)
b = corrections.apply(Brain())
mb = MushroomBody(b)
combos = [
    ("sugar",),
    ("bitter",),
    ("water",),
    ("sugar", "bitter"),
    ("sugar", "water"),
    ("bitter", "water"),
    ("antenna_touch",),
    ("looming",),
    ("antenna_touch", "looming"),
    ("sugar", "looming"),
    ("sugar", "antenna_touch"),
    ("smell_geosmin",),
    ("smell_geosmin", "sugar"),
]
rng = random.Random(7)
buf = []
t0 = time.time()
i = 0
for combo in combos:
    for scale in (0.6, 1.0, 1.4):
        for seed in range(6):
            inputs, label = [], {"stimuli": [], "odor": None, "dopamine": None}
            for k in combo:
                st = STIMULI[k]
                rate = round(st.default_rate_hz * scale)
                side = rng.choice([None, None, "left", "right"])
                ids = st.resolve(side=side) or st.resolve()
                inputs.append(StimInput(ids, rate, key=k))
                label["stimuli"].append({"key": k, "side": side or "both", "rate_hz": rate})
            r = b.run(inputs, duration_ms=250, seed=1000 + seed)
            v = r.rate_vector(b.n)
            nz = np.flatnonzero(v > 0)
            s = A.summarize(r)
            label["behaviour"] = A.behaviour_vector(s)
            label["valence"] = mb.valence(r).__dict__ | {"top": None}
            label["n_active"] = int(len(nz))
            label["duration_ms"] = 250
            label["combo"] = "+".join(combo)
            buf.append(
                {"idx": nz.astype(np.int32), "rate": v[nz].astype(np.float16), "label": label}
            )
            i += 1
    print(f"{'+'.join(combo):24s} done, {i} samples, {(time.time() - t0) / 60:.1f} min", flush=True)
np.savez_compressed(
    OUT / "shard_0000.npz",
    n=len(buf),
    idx=np.concatenate([s["idx"] for s in buf]),
    rate=np.concatenate([s["rate"] for s in buf]),
    offsets=np.cumsum([0] + [len(s["idx"]) for s in buf]).astype(np.int64),
    labels=np.array([json.dumps(s["label"], ensure_ascii=False) for s in buf]),
)
print("saved", len(buf))
