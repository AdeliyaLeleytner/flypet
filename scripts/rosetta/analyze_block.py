#!/usr/bin/env python3
"""Words <-> fly analyses on the complete DoOR block, level by level along the olfactory hierarchy.

1. Forward: does the brain response to an input predicted from text land nearest the brain response to the
   real odorant? Retrieval among all block odorants; noise ceiling = one seed against the other two.
2. Inverse: ridge from the fly code (actual inputs) to a text/structure space, cross-validated by the same
   folds; retrieval of the odorant among block odorants.
3. Round trip: word -> predicted input -> brain -> inverse map -> word.
4. RSA of each level against each text/structure space, with odorant bootstrap and partial control for structure.
"""

from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import rosetta as R
from flypet.odor import DoorOdor

DATA = ROOT / "data/rosetta_20260925"
LEVELS = ["ORN", "PN", "KC", "LH", "MBON", "DN"]
SOURCES = ["qwen_name_L18", "qwen_smell_L18", "morgan", "ngrams", "shuffled"]


class _Order:
    def __init__(self):
        order = np.load(ROOT / "data/connectivity_783.npz")["order"]
        self.flyid2i = {int(r): i for i, r in enumerate(order)}


def corr_rows(A, B):
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    A /= np.linalg.norm(A, axis=1, keepdims=True) + 1e-12
    B /= np.linalg.norm(B, axis=1, keepdims=True) + 1e-12
    return A @ B.T


def ranks_of_truth(S):
    """S[i, j] = similarity of query i to candidate j; returns rank (0 = best) of candidate i for each query."""
    return np.array([(S[i] > S[i, i]).sum() for i in range(len(S))])


def summary(rk, n):
    return {
        "top1": float((rk == 0).mean()),
        "top5": float((rk < 5).mean()),
        "median_rank": float(np.median(rk) + 1),
        "chance_top1": 1 / n,
        "chance_top5": 5 / n,
        "chance_median_rank": (n + 1) / 2,
    }


def text_spaces(door, block):
    E = np.load(DATA / "qwen4b_names.npz")
    S = np.load(DATA / "qwen4b_smells_like.npz")
    M = np.load(DATA / "morgan.npz")
    keys = [door.name2key[b.lower()] for b in block]
    row = lambda Z: {k: i for i, k in enumerate(Z["keys"])}
    e, s, m = row(E), row(S), row(M)
    return {
        "qwen_name_L18": E["L18"][[e[k] for k in keys]],
        "qwen_smell_L18": S["L18"][[s[k] for k in keys]],
        "morgan": M["X"][[m[k] for k in keys]].astype(float),
        "ngrams": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5))
        .fit_transform(block)
        .toarray(),
    }


def rdm(X, metric="corr"):
    X = StandardScaler().fit_transform(X) if metric == "zcorr" else X
    return 1 - corr_rows(X, X)


def upper(D):
    return D[np.triu_indices(len(D), 1)]


def partial_spearman(a, b, c):
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def main():
    door = DoorOdor()
    P = np.load(DATA / "block_predictions.npz")
    block, gl, folds = [str(b) for b in P["block"]], [str(g) for g in P["glomeruli"]], P["folds"]
    sims = R.load_sims(DATA / "block_sims.npz")
    pops = R.populations(_Order())
    n = len(block)
    print(
        f"{n} odorants; population sizes: " + ", ".join(f"{k} {len(v)}" for k, v in pops.items()),
        flush=True,
    )

    def code(source, level, seeds=R.SEEDS):
        return np.log1p(
            np.stack(
                [
                    np.mean([R.rates(sims[f"{source}|{b}|{s}"], pops[level]) for s in seeds], 0)
                    for b in block
                ]
            )
        )

    actual = {L: code("actual", L) for L in LEVELS}
    out = {
        "n_odorants": n,
        "glomeruli": gl,
        "forward": {},
        "ceiling": {},
        "inverse": {},
        "round_trip": {},
        "rsa": {},
        "note": "retrieval compares odour-specific deviations from the mean actual response across block odorants",
    }

    def dev(X, ref):
        return X - ref.mean(0, keepdims=True)

    # 1. forward retrieval, input level and each brain level
    A0 = P["actual"]
    out["forward"]["input"] = {
        s: summary(ranks_of_truth(corr_rows(dev(P[f"pred_{s}"], A0), dev(A0, A0))), n)
        for s in SOURCES
    }
    for L in LEVELS:
        live = actual[L].std(0) > 0
        A = actual[L][:, live]
        a_one, a_rest = code("actual", L, (11,))[:, live], code("actual", L, (23, 47))[:, live]
        out["ceiling"][L] = summary(ranks_of_truth(corr_rows(dev(a_one, A), dev(a_rest, A))), n)
        out["forward"][L] = {
            s: summary(ranks_of_truth(corr_rows(dev(code(s, L)[:, live], A), dev(A, A))), n)
            for s in SOURCES
        }

    # 2. inverse: fly code -> text/structure space, same folds; 3. round trip through the predicted-input brain response
    spaces = text_spaces(door, block)
    for L in ["input"] + LEVELS:
        X = P["actual"] if L == "input" else actual[L][:, actual[L].std(0) > 0]
        Xrt = None if L == "input" else code("qwen_name_L18", L)[:, actual[L].std(0) > 0]
        out["inverse"][L] = {}
        for sname, Y in spaces.items():
            Yz = StandardScaler().fit_transform(Y)
            pred, pred_rt = np.zeros_like(Yz), np.zeros_like(Yz)
            for f in np.unique(folds):
                tr, te = folds != f, folds == f
                sc = StandardScaler().fit(X[tr])
                model = RidgeCV(alphas=np.logspace(0, 6, 13)).fit(sc.transform(X[tr]), Yz[tr])
                pred[te] = model.predict(sc.transform(X[te]))
                if Xrt is not None:
                    pred_rt[te] = model.predict(sc.transform(Xrt[te]))
            out["inverse"][L][sname] = summary(ranks_of_truth(corr_rows(pred, Yz)), n)
            if Xrt is not None and sname == "qwen_name_L18":
                out["round_trip"][L] = summary(ranks_of_truth(corr_rows(pred_rt, Yz)), n)

    # 4. RSA with bootstrap over odorants and partial control for structure
    fly_rdms = {"input": rdm(P["actual"])} | {
        L: rdm(actual[L][:, actual[L].std(0) > 0]) for L in LEVELS
    }
    txt_rdms = {k: rdm(v, "zcorr") for k, v in spaces.items()}
    rng = np.random.default_rng(0)
    boots = [np.sort(rng.choice(n, n, replace=True)) for _ in range(200)]
    for L, DL in fly_rdms.items():
        out["rsa"][L] = {}
        for sname, DT in txt_rdms.items():
            rho = spearmanr(upper(DL), upper(DT)).statistic
            bs = []
            for b in boots:
                u = np.unique(b)
                if len(u) > 10:
                    bs.append(spearmanr(upper(DL[np.ix_(u, u)]), upper(DT[np.ix_(u, u)])).statistic)
            entry = {
                "rho": float(rho),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            }
            if sname.startswith("qwen"):
                entry["partial_given_morgan"] = partial_spearman(
                    upper(DL), upper(DT), upper(txt_rdms["morgan"])
                )
            out["rsa"][L][sname] = entry

    json.dump(out, open(DATA / "block_analysis.json", "w"), indent=1)
    fmt = lambda d: f"{d['top1'] * 100:5.1f}% {d['top5'] * 100:5.1f}% {d['median_rank']:5.0f}"
    print(
        f"\nForward (words -> fly): top-1, top-5, median rank of the true odorant among {n}; chance "
        f"{100 / n:.1f}% {500 / n:.1f}% {(n + 1) / 2:.0f}"
    )
    print(f"{'level':7s} {'ceiling':>20s} " + " ".join(f"{s:>20s}" for s in SOURCES))
    print(
        f"{'input':7s} {'':>20s} "
        + " ".join(f"{fmt(out['forward']['input'][s]):>20s}" for s in SOURCES)
    )
    for L in LEVELS:
        print(
            f"{L:7s} {fmt(out['ceiling'][L]):>20s} "
            + " ".join(f"{fmt(out['forward'][L][s]):>20s}" for s in SOURCES)
        )
    print("\nInverse (fly code -> space), same metrics:")
    print(f"{'level':7s} " + " ".join(f"{s:>20s}" for s in spaces))
    for L in out["inverse"]:
        print(f"{L:7s} " + " ".join(f"{fmt(out['inverse'][L][s]):>20s}" for s in spaces))
    print("\nRound trip word -> predicted input -> brain -> word (Qwen name space):")
    for L, d in out["round_trip"].items():
        print(f"  {L:7s} {fmt(d)}")
    print("\nRSA Spearman rho [95% CI] (partial | Morgan for Qwen):")
    for L, row in out["rsa"].items():
        print(
            f"{L:7s} "
            + " ".join(
                f"{s}: {d['rho']:+.3f} [{d['ci95'][0]:+.2f},{d['ci95'][1]:+.2f}]"
                + (f" p|M {d['partial_given_morgan']:+.3f}" if "partial_given_morgan" in d else "")
                for s, d in row.items()
            )
        )


if __name__ == "__main__":
    main()
