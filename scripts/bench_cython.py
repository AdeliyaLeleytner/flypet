_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, time

sys.path.insert(0, _ROOT + "/vendor/Drosophila_brain_model")
import pandas as pd
from brian2 import ms, Hz, Network, prefs

prefs.codegen.target = "cython"
from model import default_params, create_model, poi, get_spk_trn

V = _ROOT + "/vendor/Drosophila_brain_model"
comp = pd.read_csv(f"{V}/Completeness_783.csv", index_col=0)
flyid2i = {j: i for i, j in enumerate(comp.index)}
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
params = dict(default_params)
params["t_run"] = 300 * ms
params["r_poi"] = 150 * Hz
t0 = time.time()
neu, syn, spk = create_model(f"{V}/Completeness_783.csv", f"{V}/Connectivity_783.parquet", params)
print(f"built {time.time() - t0:.1f}s", flush=True)
pois, neu = poi(neu, [flyid2i[i] for i in sugar if i in flyid2i], [], params)
net = Network(neu, syn, spk, *pois)
t0 = time.time()
net.run(100 * ms)
print(f"first 100 ms (incl. compile): {time.time() - t0:.1f}s", flush=True)
t0 = time.time()
net.run(200 * ms)
dt = time.time() - t0
print(f"next 200 ms: {dt:.1f}s -> {dt / 0.2:.1f} s per biological second", flush=True)
tr = get_spk_trn(spk)
print(
    "active",
    len(tr),
    "spikes",
    sum(len(v) for v in tr.values()),
    "MN9_R rate",
    len(tr.get(flyid2i[720575940660219265], [])) / 0.3,
)
