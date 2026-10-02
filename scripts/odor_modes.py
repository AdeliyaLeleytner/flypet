"""Обонятельные моды на измеренной химии: следуют ли они за рецепторной близостью веществ?"""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, glob, argparse

sys.path.insert(0, _ROOT + "")
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_mutual_info_score
from flypet import connectome as C
from flypet.odor import DoorOdor

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data/odor_dataset")
ap.add_argument("--k", type=int, default=20)
ap.add_argument("--out", default="data/odor_modes.json")
a = ap.parse_args()

order, tiers = C.neuron_order()
N_STIM = tiers[-1]
flyid2i, i2flyid = C.id_maps()
idxs, rates, labels = [], [], []
for f in sorted(glob.glob(f"{a.data}/shard_*.npz")):
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
W = model.fit_transform(X)
H = model.components_
rec = 1 - model.reconstruction_err_**2 / (X**2).sum()
print(f"NMF k={a.k}: объяснено {100 * rec:.1f} % энергии", flush=True)

desc = {int(f): C.describe(int(i2flyid[int(f)])) for f in feats}
comps = sorted({l["compound"] for l in labels if l.get("compound")})
cls_of = {l["compound"]: l.get("chem_class") for l in labels if l.get("compound")}
Wn = W / (W.max(0) + 1e-9)

# профиль каждого вещества в пространстве мод
prof = np.zeros((len(comps), a.k))
for i, c in enumerate(comps):
    m = np.array([l.get("compound") == c for l in labels])
    prof[i] = Wn[m].mean(0)

# какие моды обонятельные (доминируют клетки Кеньона / ALPN / ALLN)
olf_share = []
for m in range(a.k):
    w = H[m]
    top = np.argsort(-w)[:400]
    s = sum(
        w[t]
        for t in top
        if desc[int(feats[t])]["cell_class"] in ("Kenyon_Cell", "ALPN", "ALLN", "MBON")
    )
    olf_share.append(s / max(1e-9, sum(w[t] for t in top)))
olf = [m for m in range(a.k) if olf_share[m] > 0.5]
print(f"обонятельных мод (KC/ALPN/ALLN/MBON > 50 %): {len(olf)} из {a.k}\n", flush=True)

# описание мод
for m in sorted(range(a.k), key=lambda m: -(W[:, m] ** 2).sum() * (H[m] ** 2).sum()):
    w = H[m]
    top = np.argsort(-w)[:400]
    cls = {}
    for t in top:
        d = desc[int(feats[t])]
        k = d["cell_class"] or d["super_class"] or "?"
        cls[k] = cls.get(k, 0) + w[t]
    tot = sum(cls.values()) or 1
    cls = {k: round(v / tot, 2) for k, v in sorted(cls.items(), key=lambda x: -x[1])[:3]}
    order_c = np.argsort(-prof[:, m])[:5]
    tops = [(comps[i], round(float(prof[i, m]), 2)) for i in order_c if prof[i, m] > 0.15]
    classes = {}
    for i in order_c:
        if prof[i, m] > 0.15:
            classes[cls_of[comps[i]]] = classes.get(cls_of[comps[i]], 0) + 1
    tag = "обонятельная" if m in olf else "            "
    print(f"мода {m:2d} {tag} {cls}")
    print(f"   вещества: {tops}")
    print(f"   классы: {classes}", flush=True)

# --- главная проверка: следуют ли моды за химией
door = DoorOdor()
P = prof[:, olf] if olf else prof
mode_sim, chem_sim, same_class = [], [], []
for i in range(len(comps)):
    for j in range(i + 1, len(comps)):
        cs = door.similarity(comps[i], comps[j])
        if np.isnan(cs):
            continue
        a_, b_ = P[i], P[j]
        if np.linalg.norm(a_) < 1e-6 or np.linalg.norm(b_) < 1e-6:
            continue
        mode_sim.append(float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_))))
        chem_sim.append(cs)
        same_class.append(cls_of[comps[i]] == cls_of[comps[j]])
mode_sim, chem_sim, same_class = map(np.array, (mode_sim, chem_sim, same_class))
print(f"\n=== проверка: пар веществ {len(mode_sim)}")
print(
    f"корреляция «похожи по рецепторам» ↔ «похожи по модам»: r = {np.corrcoef(chem_sim, mode_sim)[0, 1]:+.2f}"
)
print(f"похожесть по модам внутри химического класса: {mode_sim[same_class].mean():.2f}")
print(f"между разными классами:                        {mode_sim[~same_class].mean():.2f}")
km = KMeans(9, n_init=10, random_state=0).fit(P)
print(
    f"AMI кластеров мод с химическим классом: {adjusted_mutual_info_score([cls_of[c] for c in comps], km.labels_):.2f}"
)
json.dump(
    {
        "k": a.k,
        "explained": float(rec),
        "olfactory_modes": olf,
        "compounds": comps,
        "profiles": prof.tolist(),
        "r_chem_vs_mode": float(np.corrcoef(chem_sim, mode_sim)[0, 1]),
    },
    open(a.out, "w"),
    ensure_ascii=False,
    indent=1,
    default=float,
)
print("сохранено в", a.out)
