"""Language-model <-> fly olfactory brain translation: shared helpers.

The fly side is the corrected FlyWire LIF model driven through DoOR glomerular responses. To keep
odorants comparable, the main analyses use a complete block of DoOR (every odorant measured on the
same glomeruli); unmeasured glomeruli would otherwise stay silent and leak each odorant's coverage mask.
"""

from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

from .odor import DoorOdor

ROOT = Path(__file__).resolve().parents[1]
SKIP = {"water", "mineral oil", "paraffin oil", "solvent", "sfr"}
SEEDS = (11, 23, 47)


def door_names(door: DoorOdor) -> list[str]:
    return [
        door.key2name[k]
        for k in door.resp.index
        if k in door.key2name and door.key2name[k].lower() not in SKIP
    ]


def complete_block(
    door: DoorOdor, n_glomeruli: int = 23, min_measured: int = 10
) -> tuple[list[str], list[str]]:
    """Deterministic peeling: drop the row or column with the largest missing fraction until the target
    glomerulus count is reached, then keep odorants measured on all of them."""
    names = door_names(door)
    V = np.array([door.vector(n) for n in names])
    M = ~np.isnan(V)
    rows, cols = np.flatnonzero(M.sum(1) >= min_measured), np.arange(M.shape[1])
    while len(cols) > n_glomeruli:
        sub = M[np.ix_(rows, cols)]
        rmiss, cmiss = 1 - sub.mean(1), 1 - sub.mean(0)
        if rmiss.max() >= cmiss.max():
            rows = np.delete(rows, rmiss.argmax())
        else:
            cols = np.delete(cols, cmiss.argmax())
    full = rows[M[np.ix_(rows, cols)].all(1)]
    return [names[i] for i in full], [door.glomeruli[j] for j in cols]


def glomerular_matrix(door: DoorOdor, names: list[str], glomeruli: list[str]) -> np.ndarray:
    idx = [door.glomeruli.index(g) for g in glomeruli]
    return np.array([door.vector(n)[idx] for n in names], dtype=float)


def populations(brain) -> dict[str, np.ndarray]:
    """Model indices of the populations used along the olfactory hierarchy."""
    from . import connectome as C

    ann = C.annotations()
    cc, ct, sc = ann.cell_class.astype(str), ann.cell_type.astype(str), ann.super_class.astype(str)
    idx = lambda mask: np.array(
        sorted(brain.flyid2i[int(x)] for x in ann.index[mask]), dtype=np.int64
    )
    lh = (cc.isin(["LHLN", "LHCENT"]) | ct.str.match(r"^LH(AV|PV|AD|PD|CENT|LN)")) & (cc != "ALPN")
    return {
        "ORN": idx(cc == "olfactory"),
        "PN": idx(cc == "ALPN"),
        "KC": idx(cc == "Kenyon_Cell"),
        "LH": idx(lh),
        "MBON": idx(cc == "MBON"),
        "DAN": idx(cc == "DAN"),
        "DN": idx(sc == "descending"),
    }


# ---------------------------------------------------------------- parallel simulation
_W: dict = {}


def _init_worker(rate_max: float, min_response: float, scale: float, duration_ms: float):
    from . import corrections
    from .engine import Brain

    brain = corrections.apply(Brain())
    _W.update(
        brain=brain, door=DoorOdor(rate_max, min_response), scale=scale, duration_ms=duration_ms
    )


def _run(job):
    """job = (key, {glomerulus: response 0..1}, seed). Returns key, sparse model indices, spike counts."""
    from .engine import StimInput

    key, glom, seed = job
    door, brain = _W["door"], _W["brain"]
    inputs = [
        StimInput(door.gl2ids[g], door.rate_max * float(v) * _W["scale"], key=f"{key}:{g}")
        for g, v in glom.items()
        if v >= door.min_response and door.gl2ids.get(g)
    ]
    res = brain.run(inputs, duration_ms=_W["duration_ms"], seed=int(seed))
    idx = np.fromiter(res.counts.keys(), dtype=np.int32, count=len(res.counts))
    cnt = np.fromiter(res.counts.values(), dtype=np.int32, count=len(res.counts))
    return key, idx, cnt


def simulate(
    jobs: list,
    n_workers: int = 6,
    rate_max: float = 150.0,
    min_response: float = 0.05,
    scale: float = 1.0,
    duration_ms: float = 250.0,
    log_every: int = 100,
    checkpoint: Path | None = None,
) -> dict:
    """Run jobs across worker processes, each holding its own Brain. Returns {key: (idx, counts)}.
    With a checkpoint path, finished results are saved every log_every jobs and reused on restart."""
    out, t0 = {}, time.time()
    if checkpoint is not None and Path(checkpoint).exists():
        out = load_sims(Path(checkpoint))
        print(f"  resuming: {len(out)} results from {checkpoint}", flush=True)
    todo = [j for j in jobs if j[0] not in out]
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        n_workers, initializer=_init_worker, initargs=(rate_max, min_response, scale, duration_ms)
    ) as pool:
        for i, (key, idx, cnt) in enumerate(pool.imap_unordered(_run, todo, chunksize=1), 1):
            out[key] = (idx, cnt)
            if i % log_every == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} simulations, {time.time() - t0:.0f} s", flush=True)
                if checkpoint is not None:
                    save_sims(Path(checkpoint), out, {"partial": True})
    return out


def save_sims(path: Path, sims: dict, meta: dict):
    keys = sorted(sims)
    lens = np.array([len(sims[k][0]) for k in keys], dtype=np.int64)
    np.savez_compressed(
        path,
        keys=np.array(keys),
        offsets=np.concatenate([[0], np.cumsum(lens)]),
        idx=np.concatenate([sims[k][0] for k in keys]) if keys else np.zeros(0, np.int32),
        counts=np.concatenate([sims[k][1] for k in keys]) if keys else np.zeros(0, np.int32),
        meta=np.array(repr(meta)),
    )


def load_sims(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    o = z["offsets"]
    return {
        str(k): (z["idx"][o[i] : o[i + 1]], z["counts"][o[i] : o[i + 1]])
        for i, k in enumerate(z["keys"])
    }


def rates(sim: tuple, pop: np.ndarray, duration_s: float = 0.25) -> np.ndarray:
    """Dense firing rates (Hz) of one population from a sparse simulation record."""
    idx, cnt = sim
    lookup = np.full(int(pop.max()) + 1, -1, dtype=np.int64)
    lookup[pop] = np.arange(len(pop))
    v = np.zeros(len(pop), dtype=np.float32)
    ok = idx <= pop.max()
    pos = lookup[idx[ok]]
    v[pos[pos >= 0]] = cnt[ok][pos >= 0] / duration_s
    return v
