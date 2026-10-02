"""Odorant-specific metrics for cv_preds.npz: per-glomerulus r across held-out odorants and odorant retrieval."""

import sys, numpy as np

sp = sys.argv[1]
Z = np.load(f"{sp}/cv_preds.npz")
Y = Z["Y"]
P0 = Z["glom_mean"]
names = [
    "qwen_L9",
    "qwen_L18",
    "qwen_L27",
    "qwen_L36",
    "char_ngrams",
    "morgan",
    "qwen_L18_shuffled",
]


def per_glom(P):
    rs = []
    for g in range(Y.shape[1]):
        m = ~np.isnan(Y[:, g]) & ~np.isnan(P[:, g])
        if m.sum() >= 30 and Y[m, g].std() > 0:
            rs.append(np.corrcoef(Y[m, g], P[m, g])[0, 1])
    return np.array(rs)


def retrieval(P):
    n = len(Y)
    D = Y - P0
    Q = P - P0
    ranks = []
    for i in range(n):
        s = []
        for j in range(n):
            m = ~np.isnan(D[j]) & ~np.isnan(Q[i])
            a, b = Q[i, m], D[j, m]
            s.append(
                a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9) if m.sum() >= 8 else -np.inf
            )
        ranks.append(int(np.sum(np.array(s) > s[i])))
    ranks = np.array(ranks)
    return (ranks == 0).mean(), (ranks < 5).mean(), np.median(ranks) + 1


print(
    f"{'features':18s} {'per-glom r mean':>15s} {'>0.3':>5s} {'top-1':>6s} {'top-5':>6s} {'median rank':>11s}"
)
for nm in names:
    r = per_glom(Z[nm])
    t1, t5, mr = retrieval(Z[nm])
    print(
        f"{nm:18s} {r.mean():15.3f} {int((r > 0.3).sum()):>3d}/{len(r)} {t1:6.3f} {t5:6.3f} {mr:11.0f}"
    )
print(f"chance: top-1 {1 / len(Y):.3f}, top-5 {5 / len(Y):.3f}, median rank {len(Y) / 2:.0f}")
