"""Mushroom-body anatomy from the annotations + connectivity: KCs, MBONs (with transmitter), DANs, and the
DAN->MBON / KC->MBON wiring that defines compartments for the plasticity rule."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, numpy as np, pandas as pd

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C

pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 300)
ann = C.annotations()
order, tiers = C.neuron_order()
print("tiers", tiers)
i_pre, i_post, w = C.connections()
flyid2i, i2flyid = C.id_maps()
idx_of = lambda ids: np.array([flyid2i[i] for i in ids])
kc = ann.index[ann.cell_class == "Kenyon_Cell"]
mbon = ann.index[ann.cell_class == "MBON"]
dan = ann.index[ann.cell_class == "DAN"]
print(f"KC {len(kc)}  MBON {len(mbon)}  DAN {len(dan)}")
print("KC types:", ann.loc[kc, "cell_type"].value_counts().to_dict())
print("\nMBON types x transmitter:")
print(
    pd.crosstab(
        ann.loc[mbon, "cell_type"].astype(str), ann.loc[mbon, "top_nt"].astype(str)
    ).to_string()
)
print("\nDAN types:", ann.loc[dan, "cell_type"].value_counts().to_dict())
# direct DAN -> MBON synapses (compartment co-innervation proxy)
kci, mboni, dani = set(idx_of(kc)), set(idx_of(mbon)), set(idx_of(dan))
pre_t = ann.cell_type.astype(str)
post_t = ann.cell_type.astype(str)
m_dan_mbon = np.isin(i_pre, list(dani)) & np.isin(i_post, list(mboni))
df = pd.DataFrame(
    {
        "dan": [pre_t[i2flyid[i]] for i in i_pre[m_dan_mbon]],
        "mbon": [post_t[i2flyid[i]] for i in i_post[m_dan_mbon]],
        "n": np.abs(w[m_dan_mbon]),
    }
)
tab = (
    df.groupby(["mbon", "dan"])["n"]
    .sum()
    .reset_index()
    .sort_values(["mbon", "n"], ascending=[True, False])
)
print("\nDAN->MBON direct synapses, top 3 DAN types per MBON type:")
for mb, g in tab.groupby("mbon"):
    print(f"  {mb:16s}", ", ".join(f"{r.dan}:{int(r.n)}" for r in g.head(3).itertuples()))
# KC -> MBON
m_kc_mbon = np.isin(i_pre, list(kci)) & np.isin(i_post, list(mboni))
print(
    f"\nKC->MBON synapse entries: {m_kc_mbon.sum()}  (sum of counts {int(np.abs(w[m_kc_mbon]).sum())})"
)
per_mbon = (
    pd.Series(np.abs(w[m_kc_mbon]))
    .groupby([post_t[i2flyid[i]] for i in i_post[m_kc_mbon]])
    .sum()
    .sort_values(ascending=False)
)
print("KC->MBON counts per MBON type (top 20):", per_mbon.head(20).astype(int).to_dict())
# DAN -> KC
m_dan_kc = np.isin(i_pre, list(dani)) & np.isin(i_post, list(kci))
print(
    f"DAN->KC entries: {m_dan_kc.sum()}; KC->KC entries: {(np.isin(i_pre, list(kci)) & np.isin(i_post, list(kci))).sum()}"
)
# olfactory glomeruli
orn = ann[ann.cell_class == "olfactory"]
print(
    "\nglomeruli (ORN types):",
    orn.cell_type.nunique(),
    sorted(orn.cell_type.astype(str).unique())[:60],
)
# APL / DPM
print(
    "\nAPL/DPM:",
    ann[ann.cell_type.astype(str).isin(["APL", "DPM"])][
        ["cell_type", "side", "top_nt"]
    ].to_string(),
)
