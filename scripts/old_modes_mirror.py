"""Зеркальная проверка на старых данных: следовали ли прежние обонятельные моды за смыслом слов?"""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, glob

sys.path.insert(0, _ROOT + "")
import numpy as np
from sklearn.decomposition import NMF
from flypet import connectome as C
from flypet.mb import OdorEncoder, embed_texts

order, tiers = C.neuron_order()
N_STIM = tiers[-1]
flyid2i, i2flyid = C.id_maps()
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
print(f"старые данные: {n} проб × {len(feats)} нейронов", flush=True)

m = NMF(n_components=20, init="nndsvda", max_iter=400, random_state=0, tol=1e-4)
W = m.fit_transform(X)
H = m.components_
Wn = W / (W.max(0) + 1e-9)
words = sorted({l["odor"] for l in labels if l.get("odor")})
prof = np.zeros((len(words), 20))
for i, w in enumerate(words):
    mask = np.array([l.get("odor") == w for l in labels])
    prof[i] = Wn[mask].mean(0)

desc = {int(f): C.describe(int(i2flyid[int(f)])) for f in feats}
olf = []
for k in range(20):
    h = H[k]
    top = np.argsort(-h)[:400]
    s = sum(
        h[t]
        for t in top
        if desc[int(feats[t])]["cell_class"] in ("Kenyon_Cell", "ALPN", "ALLN", "MBON")
    )
    if s / max(1e-9, sum(h[t] for t in top)) > 0.5:
        olf.append(k)
P = prof[:, olf] if olf else prof
print(f"обонятельных мод: {len(olf)} из 20", flush=True)

E = embed_texts(words)  # те же эмбеддинги, что задавали старые гломерулы
ms, ws = [], []
for i in range(len(words)):
    for j in range(i + 1, len(words)):
        a_, b_ = P[i], P[j]
        if np.linalg.norm(a_) < 1e-6 or np.linalg.norm(b_) < 1e-6:
            continue
        ms.append(float(a_ @ b_ / (np.linalg.norm(a_) * np.linalg.norm(b_))))
        ws.append(float(E[i] @ E[j] / (np.linalg.norm(E[i]) * np.linalg.norm(E[j]))))
ms, ws = np.array(ms), np.array(ws)
print(f"\nпар слов: {len(ms)}")
print(
    f"корреляция «похожи по смыслу слова» ↔ «похожи по модам»: r = {np.corrcoef(ws, ms)[0, 1]:+.2f}"
)
print(f"средняя похожесть по модам: {ms.mean():.2f}")
