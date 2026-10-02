"""Build the MaleCNS descending -> ventral-nerve-cord subgraph used by flypet.vnc.

FlyWire v783 (our main model) stops at the neck: descending neurons are present but their axons
are cut, so a command like "DNp01 at 240 Hz" cannot be followed to a muscle. MaleCNS v1.0 contains
the brain *and* the VNC, and its type names match FlyWire's (column `flywireType`, and the sensory /
descending type names are identical), so the DN activity our brain model produces can be pushed into
the MaleCNS VNC by cell type.

Nodes: every descending neuron (drivers, kept separate) + every VNC neuron (simulated).
Edges: weight >= MIN_SYN, signed by the consensus neurotransmitter, same rule as Shiu et al.
       (acetylcholine +1, GABA / glutamate -1), monoamines set to 0 to match flypet.corrections
       ("dopamine acts only through the plasticity rule").

Inputs (downloaded once into data/malecns/):
  https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/
Output: data/vnc_malecns.npz + data/vnc_nodes.parquet
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "malecns"
BASE = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
FILES = {
    "annotations.feather": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters.feather": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights.feather": "connectome-weights-male-cns-v1.0-minconf-0.5-significant-only.feather",
}
OUT_NPZ = ROOT / "data" / "vnc_malecns.npz"
OUT_NODES = ROOT / "data" / "vnc_nodes.parquet"

MIN_SYN = 5  # Shiu et al. threshold
DRIVER_CLASSES = ("descending_neuron",)
VNC_CLASSES = (
    "vnc_intrinsic",
    "vnc_motor",
    "vnc_sensory",
    "vnc_efferent",
    "vnc_endocrine",
    "ascending_neuron",
    "sensory_ascending",
    "vnc_tbc",
    "vnc_sensory_tbc",
)
SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
    "histamine": -1,
    "unknown": 0,
}

# motor-neuron subclass codes -> what they move (MaleCNS annotation scheme)
MUSCLE_GROUPS = {
    "fl": "передняя нога",
    "ml": "средняя нога",
    "hl": "задняя нога",
    "wm": "крыло",
    "hm": "жужжальце",
    "nm": "шея",
    "ad": "брюшко",
    "pm": "хоботок",
    "am": "брюшко (LB)",
    "rm": "прочее",
    "xm": "прочее",
}
# the jump muscle sits under "wing muscles" in the MaleCNS scheme because it leaves through a
# mesothoracic nerve; it is the tergotrochanteral (thorax -> leg) muscle the giant fibre drives
MUSCLE_BY_TYPE = {"TTMn": "прыжок", "STTMm": "прыжок"}


def _s(df, col):
    """MaleCNS feather columns are pandas `str` dtype with float NaN mixed in."""
    return pd.Series([x if isinstance(x, str) else "" for x in df[col]], index=df.index)


def fetch():
    RAW.mkdir(parents=True, exist_ok=True)
    import urllib.request

    for local, remote in FILES.items():
        p = RAW / local
        if not p.exists():
            print(f"скачиваю {remote} …", flush=True)
            urllib.request.urlretrieve(f"{BASE}/{remote}", p)
    return {k: RAW / k for k in FILES}


def main():
    f = fetch()
    ann = pd.read_feather(f["annotations.feather"])
    sc = _s(ann, "superclass")
    keep = sc.isin(DRIVER_CLASSES + VNC_CLASSES)
    nodes = pd.DataFrame(
        {
            "body": ann.loc[keep, "bodyId"].to_numpy(dtype=np.int64),
            "superclass": sc[keep].to_numpy(),
            "type": _s(ann, "type")[keep].to_numpy(),
            "flywire_type": _s(ann, "flywireType")[keep].to_numpy(),
            "subclass": _s(ann, "subclass")[keep].to_numpy(),
            "side": _s(ann, "somaSide")[keep].to_numpy(),
            "neuromere": _s(ann, "somaNeuromere")[keep].to_numpy(),
            "nerve": _s(ann, "exitNerve")[keep].to_numpy(),
        }
    )
    nodes["is_driver"] = nodes.superclass.isin(DRIVER_CLASSES)
    # neurotransmitter -> sign of every outgoing synapse
    ntf = pd.read_feather(f["neurotransmitters.feather"])[
        ["body", "consensus_nt", "celltype_predicted_nt"]
    ]
    ntf = ntf.drop_duplicates("body").set_index("body")
    nt = _s(ntf, "consensus_nt")
    fallback = _s(ntf, "celltype_predicted_nt")  # consensus is "unclear" for ~5% of bodies
    nt = nt.where(nt.isin(list(SIGN)), fallback)
    nodes["nt"] = [x if isinstance(x, str) else "" for x in nt.reindex(nodes.body).to_numpy()]
    nodes["sign"] = nodes.nt.map(lambda x: SIGN.get(x, 0)).astype(np.int8)
    # drivers first, then VNC; muscle label for motor neurons
    nodes = nodes.sort_values(["is_driver", "body"], ascending=[False, True]).reset_index(drop=True)
    nodes["muscle"] = np.where(
        nodes.superclass == "vnc_motor", nodes.subclass.map(MUSCLE_GROUPS).fillna("прочее"), ""
    )
    nodes["muscle"] = np.where(
        nodes.type.isin(MUSCLE_BY_TYPE) & (nodes.superclass == "vnc_motor"),
        nodes.type.map(MUSCLE_BY_TYPE),
        nodes.muscle,
    )
    n_drv = int(nodes.is_driver.sum())
    print(
        f"узлов: {len(nodes)} (нисходящих-драйверов {n_drv}, ВНЦ {len(nodes) - n_drv}), "
        f"моторных {int((nodes.superclass == 'vnc_motor').sum())}"
    )
    print("  медиатор:", nodes.nt.value_counts().head(8).to_dict())

    pos = pd.Series(np.arange(len(nodes), dtype=np.int32), index=nodes.body.to_numpy())
    w = pd.read_feather(f["weights.feather"], columns=["body_pre", "body_post", "weight"])
    w = w[w.weight >= MIN_SYN]
    i = pos.reindex(w.body_pre.to_numpy()).to_numpy()
    j = pos.reindex(w.body_post.to_numpy()).to_numpy()
    m = ~(np.isnan(i) | np.isnan(j))
    i, j = i[m].astype(np.int32), j[m].astype(np.int32)
    wt = w.weight.to_numpy()[m]
    sign = nodes.sign.to_numpy()[i]
    m2 = (sign != 0) & (j >= n_drv)  # drivers are inputs only; modulators carry no fast synapse
    i, j, wt = i[m2], j[m2], wt[m2] * sign[m2]
    wt = np.clip(wt, -32767, 32767).astype(np.int16)
    print(
        f"связей >= {MIN_SYN} синапсов внутри подграфа: {len(wt)} "
        f"(из них от нисходящих {int((i < n_drv).sum())}), возбуждающих {int((wt > 0).sum())}"
    )

    OUT_NODES.parent.mkdir(parents=True, exist_ok=True)
    nodes.to_parquet(OUT_NODES)
    np.savez_compressed(
        OUT_NPZ,
        body=nodes.body.to_numpy(),
        n_drivers=np.int32(n_drv),
        i_pre=i,
        i_post=j,
        w=wt,
        min_syn=np.int32(MIN_SYN),
    )
    print(f"сохранено: {OUT_NPZ.name} ({OUT_NPZ.stat().st_size / 1e6:.1f} МБ), {OUT_NODES.name}")


if __name__ == "__main__":
    sys.exit(main())
