"""Разложить пространство состояний на моды без подсказок (NMF), затем описать каждую моду
через анатомию (какие типы клеток) и через то, что её включает."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, glob, argparse

sys.path.insert(0, _ROOT + "")
import numpy as np, pandas as pd
from sklearn.decomposition import NMF
from flypet import connectome as C

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=24)
ap.add_argument("--out", default="data/state_modes.json")
a = ap.parse_args()

order, tiers = C.neuron_order()
N_STIM = tiers[-1]
flyid2i, i2flyid = C.id_maps()
ann = C.annotations()

idxs, rates, labels = [], [], []
for f in sorted(glob.glob("data/state_dataset*/shard_*.npz")) + sorted(
    glob.glob("data/state_pairs/shard_*.npz")
):
    z = np.load(f)
    off = z["offsets"]
    for j in range(int(z["n"])):
        idxs.append(z["idx"][off[j] : off[j + 1]])
        rates.append(z["rate"][off[j] : off[j + 1]].astype(np.float32))
        labels.append(json.loads(z["labels"][j]))
n = len(labels)

cnt = {}
for ix in idxs:
    for i in ix[ix >= N_STIM]:
        cnt[int(i)] = cnt.get(int(i), 0) + 1
feats = np.array(sorted(i for i, c in cnt.items() if c >= 5))
pos = {int(i): j for j, i in enumerate(feats)}
X = np.zeros((n, len(feats)), dtype=np.float32)
for r, (ix, rt) in enumerate(zip(idxs, rates)):
    for i, v in zip(ix, rt):
        j = pos.get(int(i))
        if j is not None:
            X[r, j] = np.log1p(v)
print(f"{n} проб × {len(feats)} нейронов", flush=True)

model = NMF(n_components=a.k, init="nndsvda", max_iter=400, random_state=0, tol=1e-4)
W = model.fit_transform(X)  # проба × мода
H = model.components_  # мода × нейрон
rec = 1 - model.reconstruction_err_**2 / (X**2).sum()
print(f"NMF k={a.k}: объяснено {100 * rec:.1f} % энергии, итераций {model.n_iter_}", flush=True)

# описание мод
desc_cache = {int(f): C.describe(int(i2flyid[int(f)])) for f in feats}
stim_keys = sorted({s["key"] for l in labels for s in l["stimuli"]})
odors = sorted({l["odor"] for l in labels if l["odor"]})
Wn = W / (W.max(0) + 1e-9)

modes = []
for m in range(a.k):
    w = H[m]
    top = np.argsort(-w)[:400]
    cls, typ = {}, {}
    for t in top:
        d = desc_cache[int(feats[t])]
        key = d["cell_class"] or d["super_class"] or "?"
        cls[key] = cls.get(key, 0) + w[t]
        if d["cell_type"]:
            typ[d["cell_type"]] = typ.get(d["cell_type"], 0) + w[t]
    tot = sum(cls.values()) or 1
    cls = {k: round(v / tot, 3) for k, v in sorted(cls.items(), key=lambda x: -x[1])[:5]}
    typ = [k for k, _ in sorted(typ.items(), key=lambda x: -x[1])[:8]]
    act = Wn[:, m]
    drv = []
    for k in stim_keys:
        mask = np.array([any(s["key"] == k for s in l["stimuli"]) for l in labels])
        if mask.sum() >= 15:
            drv.append((k, float(act[mask].mean() - act[~mask].mean())))
    drv.sort(key=lambda x: -x[1])
    odr = []
    for o in odors:
        mask = np.array([l["odor"] == o for l in labels])
        if mask.sum() >= 15:
            odr.append((o, float(act[mask].mean() - act[~mask].mean())))
    odr.sort(key=lambda x: -x[1])
    dop = {}
    for tag in ("reward", "punish"):
        mask = np.array([l["dopamine"] == tag for l in labels])
        if mask.sum() >= 15:
            dop[tag] = round(float(act[mask].mean() - act[~mask].mean()), 3)
    modes.append(
        {
            "mode": m,
            "energy": float((W[:, m] ** 2).sum() * (H[m] ** 2).sum()),
            "n_neurons_top": int((w > 0.1 * w.max()).sum()),
            "cell_classes": cls,
            "cell_types": typ,
            "stimuli": [(k, round(v, 3)) for k, v in drv[:4]],
            "odors": [(k, round(v, 3)) for k, v in odr[:4]],
            "dopamine": dop,
        }
    )
tot_e = sum(x["energy"] for x in modes)
for x in modes:
    x["share"] = round(x["energy"] / tot_e, 3)
modes.sort(key=lambda x: -x["share"])
json.dump(
    {"k": a.k, "explained": float(rec), "modes": modes},
    open(a.out, "w"),
    ensure_ascii=False,
    indent=1,
    default=float,
)

pd.set_option("display.width", 250)
for x in modes:
    st = ", ".join(f"{k}+{v:.2f}" for k, v in x["stimuli"] if v > 0.02) or "—"
    od = ", ".join(f"{k}+{v:.2f}" for k, v in x["odors"] if v > 0.02) or "—"
    print(f"мода {x['mode']:2d} ({100 * x['share']:4.1f} %): классы {x['cell_classes']}")
    print(f"    типы: {', '.join(x['cell_types'][:6])}")
    print(f"    включают стимулы: {st}")
    print(f"    запахи: {od}   дофамин: {x['dopamine']}", flush=True)
