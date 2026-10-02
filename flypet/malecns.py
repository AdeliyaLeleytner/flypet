"""The whole male central nervous system (MaleCNS v1.0: brain and ventral nerve cord) as a second fly for the engine.

Same interface as flypet.connectome (annotations, neuron_order, id_maps, connections), so `Brain(conn=malecns)` runs
the Shiu et al. leaky integrate-and-fire model on the male instead of the FlyWire female. Rules, chosen to match the
female model where the data allow:
  * neurons: every body with a superclass in the MaleCNS annotation (glia, orphans and fragments are left out);
  * connections: every edge of the "significant-only" weights table, all synapse counts (the female model also uses
    edges with a single synapse);
  * sign of a presynaptic neuron from its consensus transmitter (cell-type prediction when unclear): acetylcholine
    +1; GABA, glutamate and histamine -1; serotonin and octopamine +1 as in Shiu et al.; dopamine 0, the equivalent
    of the female's dan_modulatory correction; unknown 0. The female's other corrections (antennal-lobe local
    neurons, mushroom-body gains) are not applied here.
Input tiers put the neurons courtship experiments drive first: leg and wing pheromone receptors (ppk), auditory
Johnston's-organ neurons, cVA receptors (ORN_DA1), then LC10, LC4 and LPLC2 visual projection neurons, then pC1/P1, descending
neurons and the remaining sensory neurons.
Inputs: data/malecns/{annotations,neurotransmitters,weights}.feather (scripts/fetch_data.py vnc).
Cache: data/malecns_cns.npz.
"""

from __future__ import annotations
import os
from functools import lru_cache
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("FLYPET_ROOT", Path(__file__).resolve().parent.parent))
RAW = ROOT / "data" / "malecns"
CACHE = ROOT / "data" / "malecns_cns.npz"
SIGN = {
    "acetylcholine": 1,
    "gaba": -1,
    "glutamate": -1,
    "histamine": -1,
    "serotonin": 1,
    "octopamine": 1,
    "dopamine": 0,
    "unknown": 0,
}
COLS = [
    "bodyId",
    "superclass",
    "class",
    "subclass",
    "type",
    "flywireType",
    "receptorType",
    "somaSide",
    "dimorphism",
    "fruDsx",
    "exitNerve",
]


def _str(s: pd.Series) -> pd.Series:
    """MaleCNS feather text columns mix str and float NaN."""
    return pd.Series([x if isinstance(x, str) else "" for x in s], index=s.index)


@lru_cache(maxsize=1)
def annotations() -> pd.DataFrame:
    a = pd.read_feather(RAW / "annotations.feather", columns=COLS)
    a = a.set_index("bodyId")
    for c in a.columns:
        a[c] = _str(a[c])
    return a[a.superclass != ""]


def _tier(r) -> int:
    t, rec, sc = r.type, r.receptorType, r.superclass
    if "ppk" in rec or t.startswith(("JO-A", "JO-B")) or t == "ORN_DA1":
        return 0
    if t.startswith("LC10") or t in (
        "LC4",
        "LPLC2",
    ):  # courtship tracking (LC10) and looming, as in the female model
        return 1
    if t.startswith("pC1") or sc in (
        "descending_neuron",
        "cb_sensory",
        "vnc_sensory",
        "sensory_ascending",
    ):
        return 2
    return 3


@lru_cache(maxsize=1)
def neuron_order():
    """(body ids in model order, tier_ends) where tier_ends[k] is the end index of input tier k (k = 0..2)."""
    a = annotations()
    tiers = np.array([_tier(r) for r in a.itertuples()], dtype=np.int8)
    order = np.lexsort((a.index.to_numpy(), tiers))
    t = tiers[order]
    return a.index.to_numpy()[order].astype(np.int64), [
        int(np.searchsorted(t, k, side="right")) for k in range(3)
    ]


@lru_cache(maxsize=1)
def id_maps():
    order, _ = neuron_order()
    body2i = {int(b): i for i, b in enumerate(order)}
    return body2i, {i: b for b, i in body2i.items()}


def signs() -> pd.Series:
    nt = pd.read_feather(
        RAW / "neurotransmitters.feather", columns=["body", "consensus_nt", "celltype_predicted_nt"]
    )
    nt = nt.drop_duplicates("body").set_index("body")
    cons, cell = _str(nt.consensus_nt), _str(nt.celltype_predicted_nt)
    best = cons.where(cons.isin(list(SIGN)), cell)
    return best.map(lambda x: SIGN.get(x, 0)).astype(np.int8)


def _build():
    import pyarrow.feather as pf

    order, _ = neuron_order()
    t = pf.read_table(
        RAW / "weights.feather", columns=["body_pre", "body_post", "weight"], memory_map=True
    )
    pre, post, w = (t.column(c).to_numpy() for c in ("body_pre", "body_post", "weight"))
    pos = pd.Index(order)
    ip, jp = pos.get_indexer(pre), pos.get_indexer(post)
    keep = (ip >= 0) & (jp >= 0) & (ip != jp)
    sign = signs().reindex(order).fillna(0).to_numpy(np.int8)
    ip, jp = ip[keep].astype(np.int32), jp[keep].astype(np.int32)
    ws = (np.minimum(w[keep], 32767) * sign[ip]).astype(np.int16)
    nz = ws != 0  # dopamine and unknown transmitters carry no fast signal
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, order=order, i_pre=ip[nz], i_post=jp[nz], w=ws[nz])


@lru_cache(maxsize=1)
def connections():
    """(i_pre int32, i_post int32, w int16) in model order; w = signed synapse count."""
    order, _ = neuron_order()
    if not CACHE.exists():
        _build()
    z = np.load(CACHE)
    if not np.array_equal(z["order"], order):
        _build()
        z = np.load(CACHE)
    return z["i_pre"], z["i_post"], z["w"]


def select(
    prefix: str = "", receptor: str = "", superclass: str = "", exact: bool = False
) -> list[int]:
    """Body ids by type (prefix, or the whole name with exact=True), receptor substring and/or superclass."""
    a = annotations()
    m = np.ones(len(a), bool)
    if prefix:
        m &= (a.type == prefix).to_numpy() if exact else a.type.str.startswith(prefix).to_numpy()
    if receptor:
        m &= a.receptorType.str.contains(receptor, regex=False).to_numpy()
    if superclass:
        m &= (a.superclass == superclass).to_numpy()
    return [int(b) for b in a.index[m]]
