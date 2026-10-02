"""Linear probes on the brain state: how much of the input (stimuli, odour word, dopamine) and of the
behaviour is decodable from the downstream population (sensory/afferent block excluded), and how the
accuracy grows with the number of samples."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, glob, numpy as np

sys.path.insert(0, _ROOT + "")
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from flypet import connectome as C
from flypet.catalog import STIMULI

order, tiers = C.neuron_order()
N_STIM = tiers[-1]
files = sorted(glob.glob(_ROOT + "/data/state_dataset/shard_*.npz"))
idxs, rates, labels = [], [], []
for f in files:
    z = np.load(f)
    off = z["offsets"]
    for j in range(int(z["n"])):
        idxs.append(z["idx"][off[j] : off[j + 1]])
        rates.append(z["rate"][off[j] : off[j + 1]].astype(np.float32))
        labels.append(json.loads(z["labels"][j]))
n = len(labels)
print(f"{n} samples")
KEYS = [k for k in STIMULI if k != "sugar_all_labellar"]
WORDS = sorted({l["odor"] for l in labels if l["odor"]})
Yk = np.array([[any(s["key"] == k for s in l["stimuli"]) for k in KEYS] for l in labels], dtype=int)
yw = np.array([WORDS.index(l["odor"]) if l["odor"] else -1 for l in labels])
yd = np.array([{"reward": 1, "punish": 2}.get(l["dopamine"], 0) for l in labels])
B = np.array(
    [
        [
            l["behaviour"][k]
            for k in ("proboscis", "escape", "grooming", "walk_backward", "turn_left", "turn_right")
        ]
        for l in labels
    ],
    dtype=np.float32,
)


def build(min_idx, min_count=3):
    cnt = {}
    for ix in idxs:
        for i in ix[ix >= min_idx]:
            cnt[int(i)] = cnt.get(int(i), 0) + 1
    feats = sorted(i for i, c in cnt.items() if c >= min_count)
    pos = {i: j for j, i in enumerate(feats)}
    X = np.zeros((n, len(feats)), dtype=np.float32)
    for r, (ix, rt) in enumerate(zip(idxs, rates)):
        for i, v in zip(ix, rt):
            j = pos.get(int(i))
            if j is not None:
                X[r, j] = np.log1p(v)
    return X, feats


def probe(X, tag, n_train_list=(150, 300, None)):
    for nt in n_train_list:
        accs_k, f1_k, acc_w, acc_d, r2 = [], [], [], [], []
        for tr, te in KFold(5, shuffle=True, random_state=0).split(X):
            if nt:
                tr = tr[:nt]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
            # stimulus keys: one-vs-rest
            pred = np.zeros((len(te), len(KEYS)), dtype=int)
            for k in range(len(KEYS)):
                if Yk[tr, k].sum() < 2 or Yk[tr, k].sum() > len(tr) - 2:
                    continue
                clf.fit(X[tr], Yk[tr, k])
                pred[:, k] = clf.predict(X[te])
            accs_k.append((pred == Yk[te]).all(axis=1).mean())
            tp = (pred & Yk[te]).sum()
            f1_k.append(2 * tp / max(1, pred.sum() + Yk[te].sum()))
            trw = tr[yw[tr] >= 0]
            tew = te[yw[te] >= 0]
            if len(set(yw[trw])) > 1:
                clf.fit(X[trw], yw[trw])
                acc_w.append((clf.predict(X[tew]) == yw[tew]).mean())
            clf.fit(X[tr], yd[tr])
            acc_d.append((clf.predict(X[te]) == yd[te]).mean())
            rg = make_pipeline(StandardScaler(), Ridge(alpha=100.0)).fit(X[tr], B[tr])
            pb = rg.predict(X[te])
            r2.append(
                1
                - ((pb - B[te]) ** 2).sum(0)
                / np.maximum(1e-6, ((B[te] - B[te].mean(0)) ** 2).sum(0))
            )
        print(
            f"{tag:22s} n_train={nt or int(n * 0.8):4d} | stimuli: exact-set {np.mean(accs_k):.2f}, F1 {np.mean(f1_k):.2f} | odour word {np.mean(acc_w) if acc_w else float('nan'):.2f} (chance {1 / len(WORDS):.2f}) | dopamine {np.mean(acc_d):.2f} (majority {np.bincount(yd).max() / n:.2f}) | behaviour R² {np.round(np.mean(r2, axis=0), 2).tolist()}",
            flush=True,
        )


X_down, f_down = build(N_STIM)
print(f"downstream features: {len(f_down)} neurons active in >=3 samples")
probe(X_down, "downstream only")
X_all, f_all = build(0)
print(f"all-neuron features: {len(f_all)}")
probe(X_all, "all neurons", n_train_list=(None,))
