"""Predict held-out odorants' glomerular responses (fly model input) from name embeddings; 5-fold CV."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[2])  # repository root
import sys, json, numpy as np, warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, _ROOT + "")
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from flypet.odor import DoorOdor

sp = sys.argv[1]
door = DoorOdor()
E = np.load(f"{sp}/qwen4b_names.npz")
M = np.load(f"{sp}/morgan.npz")
key2row = {k: i for i, k in enumerate(E["keys"])}
key2fp = {k: i for i, k in enumerate(M["keys"])}
G = door.glomeruli
Y, keep = [], []
for k in E["keys"]:
    v = door.vector(door.key2name[k])
    if (
        np.sum(~np.isnan(v)) >= 10
        and k in key2fp
        and door.key2name[k].lower() not in ("water", "mineral oil", "paraffin oil", "solvent")
    ):
        Y.append(v)
        keep.append(k)
Y = np.array(Y)
keep = np.array(keep)
names = [door.key2name[k] for k in keep]
print(f"odorants {len(keep)}, glomeruli {len(G)}, measured cells {int(np.sum(~np.isnan(Y)))}")
feats = {f"qwen_L{l}": E[f"L{l}"][[key2row[k] for k in keep]] for l in (9, 18, 27, 36)}
feats["char_ngrams"] = (
    TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)).fit_transform(names).toarray()
)
feats["morgan"] = M["X"][[key2fp[k] for k in keep]]
rng = np.random.default_rng(0)
feats["qwen_L18_shuffled"] = feats["qwen_L18"][rng.permutation(len(keep))]
kf = KFold(5, shuffle=True, random_state=0)
results, preds = {}, {}
for name, X in feats.items():
    P = np.full_like(Y, np.nan)
    for tr, te in kf.split(X):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        for g in range(Y.shape[1]):
            m = ~np.isnan(Y[tr, g])
            if m.sum() < 20:
                continue
            P[te, g] = RidgeCV(alphas=np.logspace(0, 5, 11)).fit(Xtr[m], Y[tr, g][m]).predict(Xte)
    per = []
    for i in range(len(Y)):
        m = ~np.isnan(Y[i]) & ~np.isnan(P[i])
        if m.sum() >= 8 and Y[i, m].std() > 0:
            per.append(np.corrcoef(Y[i, m], P[i, m])[0, 1])
    m = ~np.isnan(Y) & ~np.isnan(P)
    results[name] = {
        "per_odorant_r_mean": float(np.mean(per)),
        "per_odorant_r_median": float(np.median(per)),
        "pooled_r": float(np.corrcoef(Y[m], P[m])[0, 1]),
        "n_odorants": len(per),
    }
    preds[name] = P
    print(
        f"{name:18s} per-odorant r mean {results[name]['per_odorant_r_mean']:.3f} median {results[name]['per_odorant_r_median']:.3f} | pooled r {results[name]['pooled_r']:.3f}"
    )
# baseline that ignores the odorant: predict each glomerulus's training mean
P0 = np.full_like(Y, np.nan)
for tr, te in kf.split(Y):
    P0[te] = np.nanmean(Y[tr], axis=0)
per0 = [
    np.corrcoef(Y[i, m], P0[i, m])[0, 1]
    for i in range(len(Y))
    for m in [~np.isnan(Y[i])]
    if m.sum() >= 8 and Y[i, m].std() > 0
]
print(
    f"{'glomerulus_mean':18s} per-odorant r mean {np.mean(per0):.3f} (same prediction for every odorant)"
)
np.savez(
    f"{sp}/cv_preds.npz",
    keys=keep,
    Y=Y,
    glomeruli=np.array(G),
    **{k: v for k, v in preds.items()},
    glom_mean=P0,
)
json.dump(results, open(f"{sp}/cv_results.json", "w"), indent=1)
