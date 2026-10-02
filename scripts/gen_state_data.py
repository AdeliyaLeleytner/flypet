"""Background data generation for the state->LLM projector.
Each sample: random stimulus mixture (catalog stimuli with random sides/rates and/or a word odour, optional
dopamine) -> 250 ms run -> sparse rate vector of all active neurons + labels. Shards of 200 samples."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, time, random, numpy as np
from pathlib import Path

sys.path.insert(0, _ROOT + "")
from flypet.engine import Brain, StimInput
from flypet import corrections, analysis as A
from flypet.catalog import STIMULI
from flypet.mb import MushroomBody, OdorEncoder

OUT = Path(sys.argv[2] if len(sys.argv) > 2 else _ROOT + "/data/state_dataset")
OUT.mkdir(parents=True, exist_ok=True)
N_TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
b = corrections.apply(Brain())
mb = MushroomBody(b)
enc = OdorEncoder(k_active=3, rate_hz=100.0, frac=1.0)
WORDS = [
    "яблоко",
    "груша",
    "банан",
    "вишня",
    "молоток",
    "гвоздь",
    "отвёртка",
    "пила",
    "книга",
    "дождь",
    "кофе",
    "сыр",
    "уксус",
    "дым",
    "мята",
    "рыба",
    "хлеб",
    "мыло",
    "трава",
    "бензин",
]
keys = [k for k in STIMULI if k not in ("sugar_all_labellar",)]
rng = random.Random(int(time.time()))
existing = sorted(OUT.glob("shard_*.npz"))
n_done = sum(int(np.load(p)["n"]) for p in existing) if existing else 0
shard_id = len(existing)
buf = []


def flush():
    global buf, shard_id
    if not buf:
        return
    np.savez_compressed(
        OUT / f"shard_{shard_id:04d}.npz",
        n=len(buf),
        idx=np.concatenate([s["idx"] for s in buf]),
        rate=np.concatenate([s["rate"] for s in buf]),
        offsets=np.cumsum([0] + [len(s["idx"]) for s in buf]).astype(np.int64),
        labels=np.array([json.dumps(s["label"], ensure_ascii=False) for s in buf]),
    )
    shard_id += 1
    buf = []


t0 = time.time()
i = n_done
print(f"resuming at {n_done} samples; corrections {b.corrections}", flush=True)
while i < N_TARGET:
    inputs, label = [], {"stimuli": [], "odor": None, "dopamine": None}
    n_stim = rng.choice([0, 1, 1, 1, 2, 2, 3])
    for k in rng.sample(keys, n_stim):
        st = STIMULI[k]
        side = rng.choice([None, None, "left", "right"])
        rate = round(st.default_rate_hz * rng.uniform(0.3, 1.5))
        ids = st.resolve(side=side) or st.resolve()
        inputs.append(StimInput(ids, rate, key=k))
        label["stimuli"].append({"key": k, "side": side or "both", "rate_hz": rate})
    if rng.random() < 0.5:
        w = rng.choice(WORDS)
        inputs.append(enc.stim(w, seed=rng.randrange(1000)))
        label["odor"] = w
    if rng.random() < 0.15:
        d = rng.choice(["reward", "punish"])
        inputs.append(mb.reward() if d == "reward" else mb.punishment())
        label["dopamine"] = d
    r = b.run(inputs, duration_ms=250, seed=rng.randrange(10**6))
    v = r.rate_vector(b.n)
    nz = np.flatnonzero(v > 0)
    s = A.summarize(r)
    label["behaviour"] = A.behaviour_vector(s)
    label["valence"] = mb.valence(r).__dict__ | {"top": None}
    label["n_active"] = int(len(nz))
    label["duration_ms"] = 250
    buf.append({"idx": nz.astype(np.int32), "rate": v[nz].astype(np.float16), "label": label})
    i += 1
    if len(buf) >= 200:
        flush()
        print(f"{i} samples, {(time.time() - t0) / 60:.1f} min", flush=True)
flush()
print("done", i)
