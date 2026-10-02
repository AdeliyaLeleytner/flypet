"""Why does one glomerulus activate all projection neurons? Trace the drive onto non-cognate PNs by
presynaptic cell type, then test targeted weight fixes for odour selectivity in PNs and KCs."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, numpy as np, pandas as pd

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.engine import Brain, StimInput
from flypet.catalog import STIMULI, MN9_RIGHT

ann = C.annotations()
flyid2i, i2flyid = C.id_maps()
ct = ann.cell_type.astype(str)
nt = ann.top_nt.astype(str)
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
MBON = idx(ann.cell_class == "MBON")
is_ln = (
    ct.str.contains("LN")
    & (ann.super_class == "central")
    & ~ct.str.contains("PN")
    & ~ct.str.contains("MBON")
)
LN = idx(is_ln)
LN_exc = idx(is_ln & (nt == "acetylcholine"))
LN_inh = idx(is_ln & nt.isin(["gaba", "glutamate"]))
MPN = idx(ct.str.startswith("M_") | (ct.str.contains("PN") & ct.str.startswith("M")))
print(
    f"uPN {len(UPN)}, LN {len(LN)} (ACh {len(LN_exc)}, GABA/Glu {len(LN_inh)}), multiglomerular PN {len(MPN)}, KC {len(KC)}"
)
b = Brain()
pre, post, w0 = b.i_pre, b._i_post, b.w0
tmap = ct.to_dict()
type_of = np.array([tmap.get(i2flyid[i], "") for i in range(b.n)], dtype=object)
glom_of_pn = lambda i: type_of[i].split("_")[0]


def run_odor(gl, seed=1, rate=40, frac=0.5, dur=200):
    rng = np.random.default_rng(seed)
    ids = []
    for g in gl:
        pool = by_glom[g]
        ids += list(rng.choice(pool, max(1, int(frac * len(pool))), replace=False))
    r = b.run([StimInput(ids, rate, key="odor")], duration_ms=dur, seed=seed)
    return r, r.rate_vector(b.n)


# --- 1. baseline: one glomerulus, who drives the non-cognate PNs?
r, v = run_odor(["ORN_DM2"])
act_pn = UPN[v[UPN] > 0]
cognate = [i for i in act_pn if glom_of_pn(i) == "DM2"]
noncog = [i for i in act_pn if glom_of_pn(i) != "DM2"]
first = {i: float(t[0]) * 1000 for i, t in r.trains.items()}
print(
    f"one glomerulus DM2: active uPN {len(act_pn)}/{len(UPN)}: cognate {len(cognate)} (first spike {np.median([first[i] for i in cognate]) if cognate else -1:.1f} ms), non-cognate {len(noncog)} (median first spike {np.median([first[i] for i in noncog]):.1f} ms)"
)
m = np.isin(post, noncog)
drive = np.abs(w0[m]) * v[pre[m]]
sign = np.sign(w0[m])
df = pd.DataFrame({"pre_type": type_of[pre[m]], "drive": drive, "sign": sign})
df["cls"] = [
    "LN"
    if "LN" in t and "PN" not in t
    else (
        "uPN"
        if "_" in t and "PN" in t and not t.startswith("M_")
        else ("ORN" if t.startswith("ORN") else t[:12])
    )
    for t in df.pre_type
]
g = df.groupby(["cls", "sign"])["drive"].sum().sort_values(ascending=False)
print("drive onto non-cognate PNs (sum |count| x presyn rate) by presynaptic class and sign:")
print((g / g.sum()).head(12).round(3).to_string())
print("top presynaptic types (exc):")
print(
    df[df.sign > 0]
    .groupby("pre_type")["drive"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .round(0)
    .to_string()
)
# --- 2. targeted fixes
masks = {
    "apl_kc": np.isin(pre, APL) & np.isin(post, KC),
    "kc_kc": np.isin(pre, KC) & np.isin(post, KC),
    "lnexc_out": np.isin(pre, LN_exc),
    "ln_out": np.isin(pre, LN),
    "upn_upn": np.isin(pre, UPN) & np.isin(post, UPN),
    "upn_ln": np.isin(pre, UPN) & np.isin(post, LN),
    "mpn_out": np.isin(pre, MPN),
    "lninh_out": np.isin(pre, LN_inh),
    "orn_ln": np.isin(pre, idx(ann.cell_class == "olfactory")) & np.isin(post, LN),
}
print({k: int(v.sum()) for k, v in masks.items()})


def apply(cfg):
    b.w_scale[:] = 1.0
    for k, f in cfg.items():
        b.w_scale[masks[k]] *= f
    b.syn.w[:] = b.w0 * b.w_scale * b.params["w_syn"]


jacc = lambda a, c: len(a & c) / max(1, len(a | c))
gA = ["ORN_DM2", "ORN_DM4", "ORN_VA2"]
gB = ["ORN_DA1", "ORN_DL3", "ORN_VA1v"]


def evaluate(cfg, tag):
    apply(cfg)
    rA, vA = run_odor(gA, 1)
    _, vA2 = run_odor(gA, 2)
    rB, vB = run_odor(gB, 1)
    pA, pA2, pB = (set(np.flatnonzero(x[UPN] > 0).tolist()) for x in (vA, vA2, vB))
    kA, kA2, kB = (set(np.flatnonzero(x[KC] > 0).tolist()) for x in (vA, vA2, vB))
    cogA = np.mean([glom_of_pn(UPN[i]) in ("DM2", "DM4", "VA2") for i in pA]) if pA else 0
    rs = b.run([StimInput(STIMULI["sugar"].resolve(), 150, key="sugar")], duration_ms=300, seed=1)
    print(
        f"{tag:40s} uPN act A {len(pA):3d} (cognate frac {cogA:.2f}) J_PN(A,A2) {jacc(pA, pA2):.2f} J_PN(A,B) {jacc(pA, pB):.2f} | KC% {100 * len(kA) / len(KC):4.1f} J_KC(A,A2) {jacc(kA, kA2):.2f} J_KC(A,B) {jacc(kA, kB):.2f} | MBON act {int((vA[MBON] > 0).sum()):2d} corr {np.corrcoef(vA[MBON], vB[MBON])[0, 1] if vA[MBON].std() > 0 and vB[MBON].std() > 0 else float('nan'):.2f} | total {len(rA.counts)} | MN9 {rs.rate(MN9_RIGHT):.0f}",
        flush=True,
    )


evaluate({}, "baseline")
evaluate({"apl_kc": 5}, "apl_kc x5")
for cfg, tag in [
    ({"lnexc_out": 0}, "exc LN out x0"),
    ({"lnexc_out": 0.3}, "exc LN out x.3"),
    ({"ln_out": 0.3}, "all LN out x.3"),
    ({"upn_upn": 0}, "uPN->uPN x0"),
    ({"mpn_out": 0}, "mPN out x0"),
    ({"orn_ln": 0.3}, "ORN->LN x.3"),
    ({"lnexc_out": 0, "mpn_out": 0}, "exc LN x0 + mPN x0"),
    ({"lnexc_out": 0, "upn_upn": 0, "mpn_out": 0}, "exc LN, uPN-uPN, mPN x0"),
    ({"lnexc_out": 0, "upn_upn": 0, "mpn_out": 0, "apl_kc": 5}, "same + apl_kc x5"),
    ({"lnexc_out": 0, "upn_upn": 0, "mpn_out": 0, "apl_kc": 5, "kc_kc": 0}, "same + kc_kc x0"),
    ({"lninh_out": 3}, "inh LN out x3"),
    ({"lninh_out": 3, "apl_kc": 5}, "inh LN x3 + apl_kc x5"),
]:
    evaluate(cfg, tag)
