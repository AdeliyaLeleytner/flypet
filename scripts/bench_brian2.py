"""Benchmark: build the Shiu et al. LIF model on FlyWire v783 with Brian2 and stimulate sugar GRNs."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, time

sys.path.insert(0, _ROOT + "/vendor/Drosophila_brain_model")
import pandas as pd, numpy as np
import brian2
from brian2 import ms, Hz, Network, prefs

print("brian2", brian2.__version__, "codegen target:", prefs.codegen.target, flush=True)
from model import default_params, create_model, poi, get_spk_trn

V = _ROOT + "/vendor/Drosophila_brain_model"
comp = pd.read_csv(f"{V}/Completeness_783.csv", index_col=0)
flyid2i = {j: i for i, j in enumerate(comp.index)}
i2flyid = {i: j for j, i in flyid2i.items()}
ann = pd.read_csv(
    _ROOT + "/data/flywire_annotations_783.tsv", sep="\t", low_memory=False
).set_index("root_id")
# sugar GRNs: try notebook list (v630), else annotations
sugar630 = [
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
sugar = [i for i in sugar630 if i in flyid2i]
print("sugar ids from notebook present:", len(sugar), flush=True)
if len(sugar) < 10:
    g = ann[
        (ann.cell_class == "gustatory")
        & ann.cell_type.fillna("").str.contains("sugar", case=False)
        & (ann.side == "right")
    ]
    sugar = [i for i in g.index if i in flyid2i]
    print(
        "sugar ids from annotations (right):",
        len(sugar),
        g.cell_type.value_counts().to_dict(),
        flush=True,
    )
params = dict(default_params)
params["t_run"] = 300 * ms
params["r_poi"] = 150 * Hz
t0 = time.time()
neu, syn, spk = create_model(f"{V}/Completeness_783.csv", f"{V}/Connectivity_783.parquet", params)
print(f"model built: {len(neu)} neurons, {len(syn)} synapses, {time.time() - t0:.1f}s", flush=True)
exc = [flyid2i[i] for i in sugar]
pois, neu = poi(neu, exc, [], params)
net = Network(neu, syn, spk, *pois)
t0 = time.time()
net.run(params["t_run"])
dt = time.time() - t0
print(
    f"run 300 ms biological: {dt:.1f}s wall  ({dt / 0.3:.1f} s per biological second)", flush=True
)
trains = get_spk_trn(spk)
print("active neurons:", len(trains), " total spikes:", sum(len(v) for v in trains.values()))
rows = []
for i, v in trains.items():
    fid = i2flyid[i]
    a = ann.loc[fid] if fid in ann.index else None
    rows.append(
        (
            fid,
            len(v) / 0.3,
            a.cell_type if a is not None else "",
            a.super_class if a is not None else "",
            a.cell_class if a is not None else "",
            a.side if a is not None else "",
        )
    )
df = pd.DataFrame(
    rows, columns=["root_id", "rate_Hz", "cell_type", "super_class", "cell_class", "side"]
).sort_values("rate_Hz", ascending=False)
pd.set_option("display.width", 200)
print(df.head(40).to_string())
print("\nby super_class (active):")
print(df.super_class.value_counts().to_string())
print("\nmotor/descending active:")
print(df[df.super_class.isin(["motor", "descending"])].to_string())
mn9 = ann[ann.cell_type.fillna("").str.fullmatch("MN9")]
print("\nMN9 ids:", mn9[["side"]].to_dict()["side"])
for fid in mn9.index:
    if fid in flyid2i:
        print("MN9", ann.loc[fid, "side"], "rate", len(trains.get(flyid2i[fid], [])) / 0.3, "Hz")
