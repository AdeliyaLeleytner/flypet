"""Make the valence MBONs respond to odours: KC->MBON gain and MBON->MBON inhibition grid; sharp compartments."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, numpy as np, pandas as pd

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C, corrections
from flypet.engine import Brain, StimInput
from flypet.mb import MushroomBody, OdorEncoder

ann = C.annotations()
ct = ann.cell_type.astype(str)
nt = ann.top_nt.astype(str)
cc = ann.cell_class.astype(str)
b = corrections.apply(Brain())
mb = MushroomBody(b)
enc = OdorEncoder(k_active=3, rate_hz=100.0, frac=1.0)
pre, post, w0 = b.i_pre, b._i_post, b.w0
base_scale = b.w_scale.copy()
MBON_gaba = np.array([i for i in mb.MBON if mb.mbon_nt[int(i)] == "gaba"])
MBON_glut = np.array([i for i in mb.MBON if mb.mbon_nt[int(i)] == "glutamate"])
MBON_ach = np.array([i for i in mb.MBON if mb.mbon_nt[int(i)] == "acetylcholine"])
m_kcmbon = np.isin(pre, mb.KC) & np.isin(post, mb.MBON)
m_gabambon_out = np.isin(pre, MBON_gaba)
m_mbon_mbon = np.isin(pre, mb.MBON) & np.isin(post, mb.MBON)
m_apl_mbon = np.isin(pre, np.array([b.flyid2i[int(x)] for x in ann.index[ct == "APL"]])) & np.isin(
    post, mb.MBON
)
print(
    f"KC->MBON {m_kcmbon.sum()}, GABA-MBON outputs {m_gabambon_out.sum()}, MBON->MBON {m_mbon_mbon.sum()}, APL->MBON {m_apl_mbon.sum()}"
)
# sharp compartments: PAM vs PPL1 fraction of direct DAN input
rows = []
for mbon in mb.MBON:
    dans, wts = mb.comp[int(mbon)]
    pam = sum(w for d, w in zip(dans, wts) if d in set(mb.PAM))
    ppl = sum(w for d, w in zip(dans, wts) if d in set(mb.PPL1))
    rows.append((mb.mbon_type[int(mbon)], mb.mbon_nt[int(mbon)][:4], round(pam, 2), round(ppl, 2)))
df = (
    pd.DataFrame(rows, columns=["type", "nt", "PAM", "PPL1"])
    .groupby(["type", "nt"])
    .mean()
    .round(2)
)
print("MBON type: PAM fraction / PPL1 fraction of DAN input:")
print(df.to_string())


def apply(extra):
    b.w_scale[:] = base_scale
    for m, f in extra:
        b.w_scale[m] *= f
    b.syn.w[:] = b.w0 * b.w_scale * b.params["w_syn"]


words = ["яблоко", "молоток", "книга"]


def report(tag):
    out = []
    for w in words:
        r = b.run([enc.stim(w, seed=7)], duration_ms=200, seed=7)
        v = r.rate_vector(b.n)
        g, a, gb = v[MBON_glut], v[MBON_ach], v[MBON_gaba]
        out.append(
            f"{w[:6]:6s} glut {g.sum():4.0f}Hz({int((g > 0).sum()):2d}) ach {a.sum():4.0f}Hz({int((a > 0).sum()):2d}) gaba {gb.sum():4.0f}Hz({int((gb > 0).sum()):2d}) KC {int((v[mb.KC] > 0).sum()):3d}"
        )
    print(f"{tag:34s} | " + " | ".join(out), flush=True)
    return v


report("corrections only")
for g in (2, 3, 4, 6):
    report(f"kc_mbon x{g}")
    apply([(m_kcmbon, float(g))])
    report(f"kc_mbon x{g}")
for g, inh in [(3, 0.5), (3, 0.0), (4, 0.5), (4, 0.0), (6, 0.0)]:
    apply([(m_kcmbon, float(g)), (m_gabambon_out, inh)])
    report(f"kc_mbon x{g} gabaMBON out x{inh}")
apply([(m_kcmbon, 4.0), (m_gabambon_out, 0.0)])
r = b.run([enc.stim("яблоко", seed=7)], duration_ms=200, seed=7)
v = r.rate_vector(b.n)
act = sorted(
    [
        (mb.mbon_type[int(i)], mb.mbon_nt[int(i)][:4], round(float(v[i])))
        for i in mb.MBON
        if v[i] > 0
    ],
    key=lambda x: -x[2],
)
print("MBONs active for яблоко at kc_mbon x4, gabaMBON x0:", act[:30])
