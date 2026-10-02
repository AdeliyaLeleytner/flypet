_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import pandas as pd, pickle, re

V = _ROOT + "/vendor/Drosophila_brain_model"
comp = pd.read_csv(f"{V}/Completeness_783.csv", index_col=0)
con = pd.read_parquet(f"{V}/Connectivity_783.parquet")
print("neurons:", len(comp), " completed:", comp["Completed"].sum())
print("connectivity rows:", len(con))
print(con.head(3))
print(con.dtypes)
w = con["Excitatory x Connectivity"]
print(
    "weight stats: min",
    w.min(),
    "max",
    w.max(),
    "abs>=5:",
    (w.abs() >= 5).sum(),
    "exc:",
    (w > 0).sum(),
    "inh:",
    (w < 0).sum(),
)
print(
    "unique pre:",
    con["Presynaptic_Index"].nunique(),
    "unique post:",
    con["Postsynaptic_Index"].nunique(),
)
ann = pd.read_csv(_ROOT + "/data/flywire_annotations_783.tsv", sep="\t", low_memory=False)
print("\nannotations rows:", len(ann), " in model:", ann["root_id"].isin(comp.index).sum())
print("\nflow:\n", ann["flow"].value_counts(dropna=False).to_string())
print("\nsuper_class:\n", ann["super_class"].value_counts(dropna=False).to_string())
sens = ann[
    ann["super_class"].isin(
        [
            "sensory",
            "ascending",
            "visual_projection",
            "visual_centrifugal",
            "motor",
            "descending",
            "endocrine",
        ]
    )
]
print(
    "\ncell_class by super_class (sensory-ish):\n",
    ann[ann.super_class == "sensory"]["cell_class"].value_counts(dropna=False).to_string(),
)
g = ann[ann.cell_class == "gustatory"]
print(
    "\ngustatory cell_type x side:\n",
    pd.crosstab(g["cell_type"].fillna("NA"), g["side"].fillna("NA")).to_string(),
)
print("\ngustatory cell_sub_class:\n", g["cell_sub_class"].value_counts(dropna=False).to_string())
for cc in [
    "mechanosensory",
    "olfactory",
    "thermosensory",
    "hygrosensory",
    "visual",
    "sensory_unknown",
    "taste",
    "auditory",
    "chemosensory",
]:
    s = ann[ann.cell_class == cc]
    if len(s):
        print(
            f"\n{cc} ({len(s)}): sub_class:",
            s["cell_sub_class"].value_counts(dropna=False).head(15).to_dict(),
        )
mot = ann[ann.super_class == "motor"]
print("\nmotor cell_type:\n", mot["cell_type"].value_counts(dropna=False).to_string())
print("\nmotor cell_class:\n", mot["cell_class"].value_counts(dropna=False).to_string())
dn = ann[ann.super_class == "descending"]
print(
    "\ndescending count",
    len(dn),
    " cell_class:",
    dn["cell_class"].value_counts(dropna=False).head(10).to_dict(),
)
print("descending cell_types sample:", sorted(dn["cell_type"].dropna().unique())[:60])
# notebook v630 ids -> exist in 783?
ids = {
    "MN9_L": 720575940660219265,
    "MN9_R": 720575940645521262,
    "aBN1": 720575940630907434,
    "DN1_1": 720575940616185531,
    "DN2_l": 720575940629806974,
}
for k, v in ids.items():
    print(
        k,
        v,
        "in783:",
        v in comp.index,
        "| ann:",
        ann.loc[ann.root_id == v, ["cell_type", "super_class", "side"]].values.tolist(),
    )
sugar = [
    720575940624963786,
    720575940630233916,
    720575940637568838,
    720575940638202345,
    720575940617000768,
    720575940630797113,
    720575940632889389,
    720575940621754367,
    720575940621502051,
    720575940640649691,
    720575940639332736,
    720575940616885538,
    720575940639198653,
    720575940620900446,
    720575940617937543,
    720575940632425919,
    720575940633143833,
    720575940612670570,
    720575940628853239,
    720575940629176663,
    720575940611875570,
]
print("sugar(630) present in 783:", sum(i in comp.index for i in sugar), "/", len(sugar))
print(
    ann[ann.root_id.isin(sugar)][
        ["root_id", "cell_class", "cell_sub_class", "cell_type", "side"]
    ].to_string()
)
print("\nsez_neurons.pickle:")
d = pickle.load(open(f"{V}/sez_neurons.pickle", "rb"))
print(type(d))
print(
    {
        k: (len(v) if hasattr(v, "__len__") else v)
        for k, v in (d.items() if isinstance(d, dict) else enumerate(d))
    }
)
print("\nsearch cell_type MN9 / proboscis:")
print(
    ann[ann.cell_type.fillna("").str.contains("MN9|MN1$|MN11|MN12", regex=True)][
        ["root_id", "cell_type", "super_class", "side"]
    ].to_string()
)
