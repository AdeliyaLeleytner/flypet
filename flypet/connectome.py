"""Load the FlyWire v783 connectome files used by the Shiu et al. model plus the Schlegel et al. annotations.

Model index order (see `neuron_order`): neurons that can receive external input come first, in tiers,
so the engine can draw Poisson input only for a short contiguous block.
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("FLYPET_ROOT", Path(__file__).resolve().parent.parent))
VENDOR = ROOT / "vendor" / "Drosophila_brain_model"
PATH_COMP = VENDOR / "Completeness_783.csv"
PATH_CON = VENDOR / "Connectivity_783.parquet"
PATH_CON_NPZ = ROOT / "data" / "connectivity_783.npz"
PATH_ANN = ROOT / "data" / "flywire_annotations_783.tsv"
ANN_COLS = [
    "root_id",
    "flow",
    "super_class",
    "cell_class",
    "cell_sub_class",
    "cell_type",
    "hemibrain_type",
    "side",
    "top_nt",
    "ito_lee_hemilineage",
    "synonyms",
]
LOOM_TRACK_TYPES = {"LC4", "LPLC2", "LC11", "LC10a", "LC10b", "LC10c", "LC10d"}


@lru_cache(maxsize=1)
def completeness() -> pd.DataFrame:
    return pd.read_csv(PATH_COMP, index_col=0)


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    ann = pd.read_csv(PATH_ANN, sep="\t", usecols=ANN_COLS, low_memory=False).set_index("root_id")
    ann = ann[ann.index.isin(completeness().index)]
    for c in ann.columns:
        ann[c] = ann[c].fillna("").astype("category")
    return ann


def _tier(r) -> int:
    """0..2 for stimulable neurons (small groups first), 3 for everything else."""
    cc, sub, sc, ct, flow = r.cell_class, r.cell_sub_class, r.super_class, r.cell_type, r.flow
    if cc in ("gustatory", "thermosensory", "hygrosensory") or (
        cc == "mechanosensory" and sub != "eye bristle"
    ):
        return 0
    if (
        cc == "olfactory"
        or (cc == "mechanosensory" and sub == "eye bristle")
        or (sc == "visual_projection" and ct in LOOM_TRACK_TYPES)
    ):
        return 1
    if (
        flow == "afferent"
        or sc in ("sensory", "sensory_ascending", "visual_projection")
        or cc == "DAN"
    ):
        return 2  # DANs are included so reward/punishment can be delivered "optogenetically"
    return 3


@lru_cache(maxsize=1)
def neuron_order():
    """Returns (root_ids in model order, tier_ends) where tier_ends[k] is the end index of tier k (k = 0..2)."""
    comp = completeness()
    ann = annotations()
    tiers = np.full(len(comp), 3, dtype=np.int8)
    pos = comp.index.get_indexer(ann.index)
    tiers[pos] = np.array([_tier(r) for r in ann.itertuples()], dtype=np.int8)
    stable = np.argsort(tiers, kind="stable")
    order = comp.index.to_numpy()[stable].astype(np.int64)
    t = tiers[stable]
    tier_ends = [int(np.searchsorted(t, k, side="right")) for k in range(3)]
    return order, tier_ends


@lru_cache(maxsize=1)
def id_maps():
    order, _ = neuron_order()
    flyid2i = {int(fid): i for i, fid in enumerate(order)}
    i2flyid = {i: fid for fid, i in flyid2i.items()}
    return flyid2i, i2flyid


def _build_npz():
    con = pd.read_parquet(
        PATH_CON, columns=["Presynaptic_ID", "Postsynaptic_ID", "Excitatory x Connectivity"]
    )
    flyid2i, _ = id_maps()
    pre = con["Presynaptic_ID"].map(flyid2i).to_numpy(dtype=np.int32)
    post = con["Postsynaptic_ID"].map(flyid2i).to_numpy(dtype=np.int32)
    w = con["Excitatory x Connectivity"].to_numpy().astype(np.int16)
    PATH_CON_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PATH_CON_NPZ, order=neuron_order()[0], i_pre=pre, i_post=post, w=w)


@lru_cache(maxsize=1)
def connections():
    """(i_pre int32, i_post int32, w int16) in model index order. w = signed synapse count (Shiu et al.)."""
    if not PATH_CON_NPZ.exists():
        _build_npz()
    z = np.load(PATH_CON_NPZ)
    order, _ = neuron_order()
    if not np.array_equal(z["order"], order):
        _build_npz()
        z = np.load(PATH_CON_NPZ)
    return z["i_pre"], z["i_post"], z["w"]


def select(
    cell_class: str | None = None,
    cell_sub_class: str | None = None,
    cell_type: str | list[str] | None = None,
    super_class: str | None = None,
    side: str | None = None,
    regex: bool = False,
) -> list[int]:
    """Return FlyWire root IDs matching the given annotation fields."""
    ann = annotations()
    m = pd.Series(True, index=ann.index)
    if super_class:
        m &= ann.super_class == super_class
    if cell_class:
        m &= ann.cell_class == cell_class
    if cell_sub_class:
        m &= ann.cell_sub_class == cell_sub_class
    if cell_type is not None:
        if regex:
            m &= ann.cell_type.astype(str).str.fullmatch(cell_type)
        elif isinstance(cell_type, str):
            m &= ann.cell_type == cell_type
        else:
            m &= ann.cell_type.isin(list(cell_type))
    if side:
        m &= ann.side == side
    return [int(x) for x in ann.index[m]]


def describe(root_id: int) -> dict:
    ann = annotations()
    if root_id not in ann.index:
        return {
            "root_id": root_id,
            "cell_type": "",
            "cell_class": "",
            "cell_sub_class": "",
            "super_class": "",
            "side": "",
            "nt": "",
        }
    r = ann.loc[root_id]
    return {
        "root_id": int(root_id),
        "cell_type": str(r.cell_type),
        "cell_class": str(r.cell_class),
        "cell_sub_class": str(r.cell_sub_class),
        "super_class": str(r.super_class),
        "side": str(r.side),
        "nt": str(r.top_nt),
    }
