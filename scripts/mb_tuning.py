"""Find a minimal weight adjustment that makes the mushroom body sparse and odour-selective,
without breaking the validated sugar -> MN9 pathway."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, itertools, numpy as np

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.engine import Brain, StimInput
from flypet.catalog import STIMULI, MN9_RIGHT

ann = C.annotations()
flyid2i, i2flyid = C.id_maps()
ct = ann.cell_type.astype(str)
orn = ann[(ann.cell_class == "olfactory") & (ct != "")]
by_glom = {
    g: [int(x) for x in orn.index[orn.cell_type.astype(str) == g]]
    for g in orn.cell_type.astype(str).unique()
}
ids_of = lambda mask: np.array([flyid2i[int(x)] for x in ann.index[mask]])
KC = ids_of(ann.cell_class == "Kenyon_Cell")
MBON = ids_of(ann.cell_class == "MBON")
APL = ids_of(ct == "APL")
PN = ids_of(ct.str.contains("PN") & (ann.super_class == "central") & ~ct.str.startswith("M_"))
b = Brain()
pre, post, w0 = b.i_pre, b._i_post, b.w0
masks = {
    "inh": w0 < 0,
    "kc_kc": np.isin(pre, KC) & np.isin(post, KC),
    "apl_kc": np.isin(pre, APL) & np.isin(post, KC),
    "pn_kc": np.isin(pre, PN) & np.isin(post, KC),
    "kc_out": np.isin(pre, KC) & ~np.isin(post, KC),
}
print({k: int(v.sum()) for k, v in masks.items()})


def apply(cfg):
    b.w_scale[:] = 1.0
    for k, f in cfg.items():
        b.w_scale[masks[k]] *= f
    b.syn.w[:] = b.w0 * b.w_scale * b.params["w_syn"]


gA = ["ORN_DM2", "ORN_DM4", "ORN_VA2", "ORN_DM1", "ORN_DL5", "ORN_VM5d"]
gB = ["ORN_DA1", "ORN_DL3", "ORN_VA1v", "ORN_VL1", "ORN_DC1", "ORN_D"]


def odor(gloms, seed, rate=40, frac=0.5, dur=200):
    rng = np.random.default_rng(seed)
    ids = []
    for g in gloms:
        pool = by_glom[g]
        ids += list(rng.choice(pool, max(1, int(frac * len(pool))), replace=False))
    r = b.run([StimInput(ids, rate, key="odor")], duration_ms=dur, seed=seed)
    v = r.rate_vector(b.n)
    return (
        set(np.flatnonzero(v[KC] > 0).tolist()),
        v[MBON],
        v[KC],
        v[PN],
        v[APL].max(),
        len(r.counts),
    )


jacc = lambda a, c: len(a & c) / max(1, len(a | c))


def evaluate(cfg, tag):
    apply(cfg)
    A1, mA, kA, pA, aplA, totA = odor(gA, 1)
    A2, _, _, _, _, _ = odor(gA, 2)
    B1, mB, kB, pB, aplB, totB = odor(gB, 1)
    r = b.run([StimInput(STIMULI["sugar"].resolve(), 150, key="sugar")], duration_ms=300, seed=1)
    mn9 = r.rate(MN9_RIGHT)
    print(
        f"{tag:34s} KC% A {100 * len(A1) / len(KC):5.1f} B {100 * len(B1) / len(KC):5.1f} | J(A1,A2) {jacc(A1, A2):.2f} J(A,B) {jacc(A1, B1):.2f} | PN act A {int((pA > 0).sum())} B {int((pB > 0).sum())} | MBON act A {int((mA > 0).sum())} corr(A,B) {np.corrcoef(mA, mB)[0, 1] if mA.std() > 0 and mB.std() > 0 else float('nan'):.2f} | APL {aplA:5.1f} | total A {totA} | sugar->MN9 {mn9:5.1f} Hz",
        flush=True,
    )


evaluate({}, "baseline")
for inh, kk, apl in itertools.product((1, 2, 3), (1, 0), (1, 5)):
    if (inh, kk, apl) == (1, 1, 1):
        continue
    evaluate({"inh": inh, "kc_kc": kk, "apl_kc": apl}, f"inh x{inh} kc_kc x{kk} apl_kc x{apl}")
print("--- with PN->KC halved")
for inh, kk, apl in [(1, 0, 5), (2, 0, 5), (3, 0, 5), (2, 0, 1)]:
    evaluate(
        {"inh": inh, "kc_kc": kk, "apl_kc": apl, "pn_kc": 0.5},
        f"inh x{inh} kc_kc x{kk} apl_kc x{apl} pn_kc x.5",
    )
