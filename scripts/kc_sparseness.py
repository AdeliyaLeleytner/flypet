"""Is the mushroom body usable as a memory in this LIF model? Stimulate k glomeruli and measure
KC sparseness, MBON/APL/DAN activity, and overlap of KC codes between odours."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, numpy as np

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.engine import Brain, StimInput

ann = C.annotations()
flyid2i, i2flyid = C.id_maps()
ct = ann.cell_type.astype(str)
orn = ann[(ann.cell_class == "olfactory") & (ct != "")]
glom = sorted(orn.cell_type.astype(str).unique())
by_glom = {g: [int(x) for x in orn.index[orn.cell_type.astype(str) == g]] for g in glom}
kc_ids = [int(x) for x in ann.index[ann.cell_class == "Kenyon_Cell"]]
mbon_ids = [int(x) for x in ann.index[ann.cell_class == "MBON"]]
dan_ids = [int(x) for x in ann.index[ann.cell_class == "DAN"]]
apl_ids = [int(x) for x in ann.index[ct == "APL"]]
pn_ids = [int(x) for x in ann.index[ct.str.contains("PN") & (ann.super_class == "central")]]
b = Brain()
i_pre, w0 = b.i_pre, b.w0
dan_idx = np.array([flyid2i[i] for i in dan_ids])
m = np.isin(i_pre, dan_idx)
print(f"DAN output synapses: {m.sum()}, positive (excitatory) fraction {(w0[m] > 0).mean():.2f}")
kc_idx = np.array([flyid2i[i] for i in kc_ids])
mk = np.isin(i_pre, kc_idx)
print(
    f"KC output synapses: {mk.sum()}, excitatory fraction {(w0[mk] > 0).mean():.2f}; KC->KC entries {(mk & np.isin(b._i_post, kc_idx)).sum()}"
)


def probe(gloms, rate, dur=200, frac=1.0, seed=1, tag=""):
    rng = np.random.default_rng(seed)
    ids = []
    for g in gloms:
        pool = by_glom[g]
        k = max(1, int(round(frac * len(pool))))
        ids += list(rng.choice(pool, k, replace=False)) if k < len(pool) else pool
    r = b.run([StimInput(ids, rate, key="odor")], duration_ms=dur, seed=seed)
    rates = r.rate_vector(b.n)
    kc = rates[[flyid2i[i] for i in kc_ids]]
    mb = rates[[flyid2i[i] for i in mbon_ids]]
    dn = rates[[flyid2i[i] for i in dan_ids]]
    apl = rates[[flyid2i[i] for i in apl_ids]]
    pn = rates[[flyid2i[i] for i in pn_ids]]
    active_kc = set(np.flatnonzero(kc > 0).tolist())
    print(
        f"{tag:28s} ORN {len(ids):3d}@{rate:3.0f}Hz {dur}ms | PN active {int((pn > 0).sum()):3d}/{len(pn)} | KC active {len(active_kc):4d}/{len(kc)} ({100 * len(active_kc) / len(kc):4.1f}%), mean rate of active {kc[kc > 0].mean() if len(active_kc) else 0:5.1f} Hz | MBON active {int((mb > 0).sum()):2d}/96 max {mb.max():5.1f} | APL {apl.max():5.1f} Hz | DAN active {int((dn > 0).sum())} | total active {len(r.counts)}",
        flush=True,
    )
    return active_kc, mb


def jacc(a, b_):
    return len(a & b_) / max(1, len(a | b_))


gA = ["ORN_DM2", "ORN_DM4", "ORN_VA2", "ORN_DM1", "ORN_DL5", "ORN_VM5d"]
gB = ["ORN_DA1", "ORN_DL3", "ORN_VA1v", "ORN_VL1", "ORN_DC1", "ORN_D"]
gC = gA[:3] + gB[:3]
for k in (1, 3, 6):
    for rate in (30, 60):
        probe(gA[:k], rate, tag=f"A[{k}] @ {rate}")
print("--- fraction of ORNs per glomerulus 0.3, 6 glomeruli")
probe(gA, 40, frac=0.3, tag="A[6] frac .3 @40")
print("--- overlap between odours (6 glomeruli, 40 Hz, frac .5)")
A1, mbA = probe(gA, 40, frac=0.5, seed=1, tag="A seed1")
A2, _ = probe(gA, 40, frac=0.5, seed=2, tag="A seed2")
B1, mbB = probe(gB, 40, frac=0.5, seed=1, tag="B seed1")
C1, mbC = probe(gC, 40, frac=0.5, seed=1, tag="C=half A half B")
print(
    f"Jaccard KC sets: A1~A2 {jacc(A1, A2):.2f}  A1~B1 {jacc(A1, B1):.2f}  A1~C1 {jacc(A1, C1):.2f}  B1~C1 {jacc(B1, C1):.2f}"
)
print(
    f"MBON vectors corr: A~B {np.corrcoef(mbA, mbB)[0, 1]:.2f}, A~C {np.corrcoef(mbA, mbC)[0, 1]:.2f}"
)
