"""Biological corrections: antennal-lobe LNs inhibitory (their predicted dopamine/serotonin/ACh outputs are
treated as excitatory by the Shiu sign rule), dopaminergic neurons modulatory only. Measure odour selectivity,
KC sparseness and the three validated pathways."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, numpy as np

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.engine import Brain, StimInput
from flypet.catalog import STIMULI, MN9_RIGHT, READOUTS
from flypet import analysis as A

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
ALLN = idx(cc == "ALLN")
DAN = idx(cc == "DAN")
print(
    f"ALLN {len(ALLN)}: predicted transmitters",
    ann.loc[ann.cell_class == "ALLN", "top_nt"].astype(str).value_counts().to_dict(),
)
b = Brain()
pre, post, w0 = b.i_pre, b._i_post, b.w0
tmap = ct.to_dict()
type_of = np.array([tmap.get(i2flyid[i], "") for i in range(b.n)], dtype=object)
glom_of_pn = lambda i: type_of[i].split("_")[0]
m_alln_pos = np.isin(pre, ALLN) & (w0 > 0)
m_dan = np.isin(pre, DAN)
m_aplkc = np.isin(pre, APL) & np.isin(post, KC)
print(
    f"ALLN positive-sign output synapses {m_alln_pos.sum()} of {np.isin(pre, ALLN).sum()}; DAN outputs {m_dan.sum()}"
)


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


jacc = lambda a, c: len(a & c) / max(1, len(a | c))
gA = ["ORN_DM2", "ORN_DM4", "ORN_VA2"]
gB = ["ORN_DA1", "ORN_DL3", "ORN_VA1v"]
gC = ["ORN_DM2", "ORN_DM4", "ORN_DL3"]


def evaluate(cfg, tag):
    apply(cfg)
    out = {}
    for name, gl, seed in [("A1", gA, 1), ("A2", gA, 2), ("B1", gB, 1), ("C1", gC, 1)]:
        r = run_odor(gl, seed)
        v = r.rate_vector(b.n)
        out[name] = (
            set(np.flatnonzero(v[UPN] > 0).tolist()),
            set(np.flatnonzero(v[KC] > 0).tolist()),
            v[MBON],
            len(r.counts),
            v[APL].max(),
            v[KC][v[KC] > 0].mean() if (v[KC] > 0).any() else 0,
        )
    pA, kA, mA, totA, aplA, kcr = out["A1"]
    pA2, kA2, _, _, _, _ = out["A2"]
    pB, kB, mB, _, _, _ = out["B1"]
    pC, kC, _, _, _, _ = out["C1"]
    cog = np.mean([glom_of_pn(UPN[i]) in ("DM2", "DM4", "VA2") for i in pA]) if pA else 0
    s = []
    for key, read in [
        ("sugar", "proboscis_extension"),
        ("antenna_touch", "antennal_grooming"),
        ("looming", "escape_takeoff"),
    ]:
        st = STIMULI[key]
        r = b.run([StimInput(st.resolve(), st.default_rate_hz, key=key)], duration_ms=300, seed=1)
        bt = A.readout_table(r)[read]
        s.append(f"{bt['n_active']}/{bt['n_neurons']}@{bt['max_rate_hz']:.0f}")
    print(
        f"{tag:34s} uPN act {len(pA):3d} cog {cog:.2f} J_PN(A,A2) {jacc(pA, pA2):.2f} (A,B) {jacc(pA, pB):.2f} (A,C) {jacc(pA, pC):.2f} | KC {100 * len(kA) / len(KC):4.1f}% @{kcr:4.1f}Hz J_KC(A,A2) {jacc(kA, kA2):.2f} (A,B) {jacc(kA, kB):.2f} (A,C) {jacc(kA, kC):.2f} | MBON act {int((mA > 0).sum()):2d} corr(A,B) {np.corrcoef(mA, mB)[0, 1] if mA.std() > 0 and mB.std() > 0 else float('nan'):.2f} | APL {aplA:4.0f} | total {totA:5d} | sugar {s[0]} groom {s[1]} escape {s[2]}",
        flush=True,
    )


evaluate([], "baseline")
evaluate([(m_alln_pos, 0.0)], "ALLN exc out x0")
evaluate([(m_alln_pos, -1.0)], "ALLN exc out flipped to inh")
evaluate([(m_alln_pos, 0.0), (m_dan, 0.0)], "ALLN x0 + DAN out x0")
evaluate([(m_alln_pos, -1.0), (m_dan, 0.0)], "ALLN flip + DAN out x0")
evaluate([(m_alln_pos, -1.0), (m_dan, 0.0), (m_aplkc, 3.0)], "ALLN flip + DAN x0 + apl_kc x3")
evaluate([(m_alln_pos, 0.0), (m_dan, 0.0), (m_aplkc, 3.0)], "ALLN x0 + DAN x0 + apl_kc x3")
