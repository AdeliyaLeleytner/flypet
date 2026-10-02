"""Second diagnostic: with excitatory LN outputs removed, who relays a single-glomerulus response to the
other glomeruli? Also: is the code odour-selective in the first tens of ms (before the spread)?"""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, numpy as np, pandas as pd

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.engine import Brain, StimInput

ann = C.annotations()
flyid2i, i2flyid = C.id_maps()
ct = ann.cell_type.astype(str)
nt = ann.top_nt.astype(str)
cc = ann.cell_class.astype(str)
sc = ann.super_class.astype(str)
orn = ann[(ann.cell_class == "olfactory") & (ct != "")]
by_glom = {
    g: [int(x) for x in orn.index[orn.cell_type.astype(str) == g]]
    for g in orn.cell_type.astype(str).unique()
}
idx = lambda mask: np.array([flyid2i[int(x)] for x in ann.index[mask]], dtype=np.int64)
is_upn = (
    ct.str.contains("PN")
    & (ann.super_class == "central")
    & ~ct.str.startswith("M_")
    & ct.str.contains("_")
)
UPN = idx(is_upn)
KC = idx(ann.cell_class == "Kenyon_Cell")
APL = idx(ct == "APL")
is_ln = (
    ct.str.contains("LN")
    & (ann.super_class == "central")
    & ~ct.str.contains("PN")
    & ~ct.str.contains("MBON")
)
LN_exc = idx(is_ln & (nt == "acetylcholine"))
b = Brain()
pre, post, w0 = b.i_pre, b._i_post, b.w0
tmap, cmap, smap, ntmap = ct.to_dict(), cc.to_dict(), sc.to_dict(), nt.to_dict()
type_of = np.array([tmap.get(i2flyid[i], "") for i in range(b.n)], dtype=object)
class_of = np.array(
    [(cmap.get(i2flyid[i], "") or smap.get(i2flyid[i], "")) for i in range(b.n)], dtype=object
)
glom_of_pn = lambda i: type_of[i].split("_")[0]


def apply(cfg):
    b.w_scale[:] = 1.0
    for m, f in cfg:
        b.w_scale[m] *= f
    b.syn.w[:] = b.w0 * b.w_scale * b.params["w_syn"]


def run_odor(gl, seed=1, rate=40, frac=0.5, dur=200):
    rng = np.random.default_rng(seed)
    ids = []
    for g in gl:
        pool = by_glom[g]
        ids += list(rng.choice(pool, max(1, int(frac * len(pool))), replace=False))
    return b.run([StimInput(ids, rate, key="odor")], duration_ms=dur, seed=seed)


def diagnose(tag):
    r = run_odor(["ORN_DM2"])
    first = {i: float(t[0]) * 1000 for i, t in r.trains.items()}
    act_pn = [i for i in UPN if i in first]
    cog = [i for i in act_pn if glom_of_pn(i) == "DM2"]
    non = [i for i in act_pn if glom_of_pn(i) != "DM2"]
    t_cog = np.median([first[i] for i in cog]) if cog else 0
    t_non = np.percentile([first[i] for i in non], 25) if non else 200
    print(
        f"[{tag}] active uPN {len(act_pn)}: cognate {len(cog)} @ {t_cog:.1f} ms, non-cognate {len(non)} (25th pct first spike {t_non:.1f} ms)"
    )
    # intermediaries: non-PN neurons spiking between cognate onset and the early non-cognate onset
    inter = [
        i
        for i, t0 in first.items()
        if t_cog <= t0 <= t_non and i not in set(UPN) and class_of[i] != "olfactory"
    ]
    df = pd.DataFrame(
        {
            "type": type_of[inter],
            "class": class_of[inter],
            "nt": [ntmap.get(i2flyid[i], "") for i in inter],
        }
    )
    print("  intermediaries by class:", df["class"].value_counts().head(8).to_dict())
    print(
        "  intermediaries by type (top 15):",
        df.groupby(["type", "nt"]).size().sort_values(ascending=False).head(15).to_dict(),
    )
    # early drive onto non-cognate PNs: presynaptic spikes before t_non+5ms, weighted by synapse count
    early_rate = np.zeros(b.n)
    for i, t in r.trains.items():
        early_rate[i] = np.sum(t * 1000 < t_non + 5)
    m = np.isin(post, non)
    d = np.abs(w0[m]) * early_rate[pre[m]]
    s = np.sign(w0[m])
    dd = pd.DataFrame({"type": type_of[pre[m]], "class": class_of[pre[m]], "drive": d, "sign": s})
    dd = dd[dd.drive > 0]
    g = dd.groupby(["class", "sign"])["drive"].sum()
    g = (g / g.sum()).sort_values(ascending=False)
    print("  early drive onto non-cognate PNs by class/sign:", g.head(8).round(3).to_dict())
    print(
        "  early exc drive top types:",
        dd[dd.sign > 0]
        .groupby("type")["drive"]
        .sum()
        .sort_values(ascending=False)
        .head(8)
        .round(0)
        .to_dict(),
    )


apply([])
diagnose("baseline")
m_lnexc = np.isin(pre, LN_exc)
apply([(m_lnexc, 0.0)])
diagnose("excLN out x0")
# --- short-window selectivity
jacc = lambda a, c: len(a & c) / max(1, len(a | c))
gA = ["ORN_DM2", "ORN_DM4", "ORN_VA2"]
gB = ["ORN_DA1", "ORN_DL3", "ORN_VA1v"]
m_aplkc = np.isin(pre, APL) & np.isin(post, KC)
for cfg, tag in [
    ([], "baseline"),
    ([(m_aplkc, 5.0)], "apl_kc x5"),
    ([(m_lnexc, 0.0), (m_aplkc, 5.0)], "excLN x0 + apl_kc x5"),
]:
    apply(cfg)
    for dur in (30, 50, 80):
        res = {}
        for name, gl, seed in [("A1", gA, 1), ("A2", gA, 2), ("B1", gB, 1)]:
            r = run_odor(gl, seed, dur=dur)
            v = r.rate_vector(b.n)
            res[name] = (
                set(np.flatnonzero(v[UPN] > 0).tolist()),
                set(np.flatnonzero(v[KC] > 0).tolist()),
            )
        pA, kA = res["A1"]
        pA2, kA2 = res["A2"]
        pB, kB = res["B1"]
        cog = np.mean([glom_of_pn(UPN[i]) in ("DM2", "DM4", "VA2") for i in pA]) if pA else 0
        print(
            f"{tag:24s} stim {dur:3d} ms | uPN act {len(pA):3d} cognate frac {cog:.2f} J_PN(A,B) {jacc(pA, pB):.2f} | KC act {len(kA):4d} ({100 * len(kA) / len(KC):4.1f}%) J_KC(A1,A2) {jacc(kA, kA2):.2f} J_KC(A,B) {jacc(kA, kB):.2f}",
            flush=True,
        )
