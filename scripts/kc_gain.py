"""With the antennal lobe fixed, find PN->KC gain / input strength giving sparse, odour-selective KCs."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, itertools, numpy as np

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.engine import Brain, StimInput

ann = C.annotations()
flyid2i, i2flyid = C.id_maps()
ct = ann.cell_type.astype(str)
nt = ann.top_nt.astype(str)
cc = ann.cell_class.astype(str)
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
KC = idx(cc == "Kenyon_Cell")
APL = idx(ct == "APL")
MBON = idx(cc == "MBON")
DAN = idx(cc == "DAN")
ALLN_ach = idx((cc == "ALLN") & (nt == "acetylcholine"))
ALLN_mono = idx((cc == "ALLN") & nt.isin(["dopamine", "serotonin", "octopamine"]))
PN_all = idx(cc == "ALPN")
b = Brain()
pre, post, w0 = b.i_pre, b._i_post, b.w0
m_ach = np.isin(pre, ALLN_ach) & (w0 > 0)
m_mono = np.isin(pre, ALLN_mono) & (w0 > 0)
m_dan = np.isin(pre, DAN)
m_pnkc = np.isin(pre, PN_all) & np.isin(post, KC)
m_aplkc = np.isin(pre, APL) & np.isin(post, KC)
print(
    f"PN->KC synapses {m_pnkc.sum()} (uPN->KC {(np.isin(pre, UPN) & np.isin(post, KC)).sum()}), per-pair count median {np.median(np.abs(w0[m_pnkc])):.0f}, mean {np.abs(w0[m_pnkc]).mean():.1f}"
)
base = [(m_ach, 0.0), (m_mono, -1.0), (m_dan, 0.0)]


def apply(cfg):
    b.w_scale[:] = 1.0
    for m, f in base + cfg:
        b.w_scale[m] *= f
    b.syn.w[:] = b.w0 * b.w_scale * b.params["w_syn"]


def run_odor(gl, seed=1, rate=40, frac=0.5, dur=200):
    rng = np.random.default_rng(seed)
    ids = []
    for g in gl:
        pool = by_glom[g]
        k = max(1, int(round(frac * len(pool))))
        ids += list(rng.choice(pool, k, replace=False)) if k < len(pool) else pool
    return b.run([StimInput(ids, rate, key="odor")], duration_ms=dur, seed=seed)


jacc = lambda a, c: len(a & c) / max(1, len(a | c))
gA = ["ORN_DM2", "ORN_DM4", "ORN_VA2"]
gB = ["ORN_DA1", "ORN_DL3", "ORN_VA1v"]
gC = ["ORN_DM2", "ORN_DM4", "ORN_DL3"]


def evaluate(cfg, rate, frac, tag, dur=200):
    apply(cfg)
    out = {}
    for name, gl, seed in [("A1", gA, 1), ("A2", gA, 2), ("B1", gB, 1), ("C1", gC, 1)]:
        r = run_odor(gl, seed, rate=rate, frac=frac, dur=dur)
        v = r.rate_vector(b.n)
        out[name] = (set(np.flatnonzero(v[KC] > 0).tolist()), v, len(r.counts))
    kA, vA, tot = out["A1"]
    kA2, _, _ = out["A2"]
    kB, vB, _ = out["B1"]
    kC, _, _ = out["C1"]
    pn = vA[UPN]
    kc = vA[KC]
    print(
        f"{tag:26s} ORN@{rate:3.0f}Hz frac {frac:.1f} | uPN act {int((pn > 0).sum()):3d} mean {pn[pn > 0].mean() if (pn > 0).any() else 0:5.1f} Hz | KC {100 * len(kA) / len(KC):5.2f}% ({len(kA):4d}) @{kc[kc > 0].mean() if len(kA) else 0:4.1f}Hz J(A1,A2) {jacc(kA, kA2):.2f} J(A,B) {jacc(kA, kB):.2f} J(A,C) {jacc(kA, kC):.2f} | MBON act {int((vA[MBON] > 0).sum()):2d} corr(A,B) {np.corrcoef(vA[MBON], vB[MBON])[0, 1] if vA[MBON].std() > 0 and vB[MBON].std() > 0 else float('nan'):.2f} | APL {vA[APL].max():4.0f} | total {tot}",
        flush=True,
    )


for pnkc in (1, 2, 3, 4):
    for rate, frac in [(40, 0.5), (100, 1.0), (150, 1.0)]:
        evaluate([(m_pnkc, float(pnkc))], rate, frac, f"pn_kc x{pnkc}")
print("--- with APL->KC scaling on top of pn_kc x3, ORN 100 Hz full")
for apl in (0.5, 2, 4):
    evaluate([(m_pnkc, 3.0), (m_aplkc, float(apl))], 100, 1.0, f"pn_kc x3 apl_kc x{apl}")
