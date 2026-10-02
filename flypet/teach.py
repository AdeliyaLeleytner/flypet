"""Teaching a connectome fly: an environment where a teacher designs a training protocol and the fly
learns only through its own dopamine-gated KC->MBON plasticity (flypet.mb), exactly as in case C.

A protocol is a short list of trials: an odour alone, or an odour with reward (PAM) or punishment (PPL1).
After every trial the batch update is applied, including odour-only trials, whose own dopamine recruitment
(KC->DAN) can change memory. The fly is then probed on every panel odour without learning; the score says
how well the change in MBON valence matches a target (+1 approach more, -1 avoid more, 0 leave unchanged).
"""

from __future__ import annotations

import multiprocessing as mp
import re
import time
from dataclasses import dataclass, field

import numpy as np

REINFORCEMENTS = ("reward", "punish", "none")
ACTION_RE = re.compile(
    r"^\s*(?:\d+[.)]\s*)?(TRAIN|EXPOSE)\s+(.+?)(?:\s*(?:\+|WITH)\s*(REWARD|PUNISH(?:MENT)?|NOTHING|NONE))?\s*$",
    re.I,
)


@dataclass
class Task:
    panel: list[str]
    targets: dict[str, int]
    budget: int = 6
    delta: float = 0.15
    name: str = ""

    def describe(self) -> str:
        want = {1: "approach MORE than now", -1: "avoid MORE than now", 0: "stay UNCHANGED"}
        return "\n".join(f"- {o}: {want[self.targets[o]]}" for o in self.panel)


@dataclass
class Outcome:
    actions: list[tuple[str, str]]
    valence: dict[str, float]
    naive: dict[str, float]
    success: float
    reward: float
    per_odour: dict[str, float] = field(default_factory=dict)
    invalid_lines: int = 0


def parse_protocol(text: str, panel: list[str], budget: int) -> tuple[list[tuple[str, str]], int]:
    """Lines like 'TRAIN ethyl acetate + REWARD', 'TRAIN pentanol WITH PUNISHMENT', 'EXPOSE hexanal'.
    Unknown odours or malformed lines are counted as invalid; lines beyond the budget are dropped."""
    lower = {p.lower(): p for p in panel}
    actions, invalid = [], 0
    for line in text.strip().splitlines():
        line = line.strip().strip("`*-").strip()
        if not line:
            continue
        m = ACTION_RE.match(line)
        if not m:
            invalid += 1
            continue
        verb, odour, reinf = (
            m.group(1).upper(),
            m.group(2).strip().strip("\"'").lower(),
            (m.group(3) or "").lower(),
        )
        if odour not in lower:
            invalid += 1
            continue
        r = (
            "none"
            if verb == "EXPOSE" or reinf in ("", "nothing", "none")
            else ("reward" if reinf == "reward" else "punish")
        )
        actions.append((lower[odour], r))
    return actions[:budget], invalid + max(0, len(actions) - budget)


def score(
    task: Task, valence: dict[str, float], naive: dict[str, float]
) -> tuple[float, float, dict[str, float]]:
    """success: fraction of panel odours meeting their target; reward: mean of a clipped graded version in [-1, 1]."""
    ok, graded = [], {}
    for o in task.panel:
        d, t = valence[o] - naive[o], task.targets[o]
        if t == 0:
            ok.append(abs(d) < task.delta)
            graded[o] = 1.0 - min(2.0, abs(d) / task.delta)
        else:
            ok.append(t * d >= task.delta)
            graded[o] = float(np.clip(t * d / task.delta, -1.0, 1.0))
    return float(np.mean(ok)), float(np.mean(list(graded.values()))), graded


# ---------------------------------------------------------------- the real (Brian2) fly, one per worker
_F: dict = {}


def _init_fly(
    odor_scale: float,
    rate_max: float,
    min_response: float,
    reinforcement_hz: float,
    duration_ms: float,
):
    from . import corrections
    from .engine import Brain
    from .mb import MushroomBody
    from .odor import DoorOdor

    brain = corrections.apply(Brain())
    _F.update(
        brain=brain,
        mb=MushroomBody(brain),
        door=DoorOdor(rate_max, min_response),
        scale=odor_scale,
        reinf_hz=reinforcement_hz,
        duration_ms=duration_ms,
    )


def _trial(odour: str, reinf: str, seed: int, learn: bool):
    mb, door = _F["mb"], _F["door"]
    inputs = door.stim(odour, scale=_F["scale"])
    if reinf == "reward":
        inputs.append(mb.reward(_F["reinf_hz"]))
    elif reinf == "punish":
        inputs.append(mb.punishment(_F["reinf_hz"]))
    res = _F["brain"].run(inputs, duration_ms=_F["duration_ms"], seed=int(seed))
    log = mb.learn(res, note=f"{odour}/{reinf}") if learn else None
    return res, log


def _episode(job):
    """job = (key, actions, panel, train_seed, eval_seeds). Fresh memory, run the protocol, probe the panel."""
    key, actions, panel, train_seed, eval_seeds = job
    mb = _F["mb"]
    mb.reset()
    logs = []
    for step, (odour, reinf) in enumerate(actions):
        _, log = _trial(odour, reinf, train_seed + step, learn=True)
        logs.append({k: log[k] for k in ("n_synapses_changed", "dan_active", "kc_active")})
    val = {}
    for o in panel:
        val[o] = float(
            np.mean([mb.valence(_trial(o, "none", s, learn=False)[0]).score for s in eval_seeds])
        )
    return key, val, logs, mb.mem.copy() if key.startswith("keepmem:") else None


def _dopamine(res):
    """Thresholded compartment dopamine per MBON, exactly as MushroomBody.learn computes it."""
    mb = _F["mb"]
    v = res.rate_vector(_F["brain"].n)
    out = np.zeros(len(mb.MBON))
    for k, m in enumerate(mb.MBON):
        dans, wts = mb.comp[int(m)]
        if len(dans):
            level = float(np.minimum(1.0, (v[dans] * wts).sum() / mb.dan_scale_hz))
            out[k] = level if level >= mb.dan_threshold else 0.0
    return out


def _measure(job):
    """Naive-memory trial without learning. job = (key, odour, reinf, seed) -> KC rates, dopamine, MBON rates."""
    key, odour, reinf, seed = job
    mb = _F["mb"]
    mb.reset()
    res, _ = _trial(odour, reinf, seed, learn=False)
    v = res.rate_vector(_F["brain"].n)
    return key, {
        "kc": v[mb.KC].astype(np.float32),
        "dop": _dopamine(res),
        "mbon": v[mb.MBON].astype(np.float64),
    }


def _episode_full(job):
    """Like _episode, but also returns the final memory factors and every probe's MBON rate vector."""
    key, actions, panel, train_seed, eval_seeds = job
    mb = _F["mb"]
    mb.reset()
    for step, (odour, reinf) in enumerate(actions):
        _trial(odour, reinf, train_seed + step, learn=True)
    mem, rates, val = mb.mem.copy(), {}, {}
    for o in panel:
        vs = []
        for s in eval_seeds:
            res, _ = _trial(o, "none", s, learn=False)
            rates[(o, s)] = res.rate_vector(_F["brain"].n)[mb.MBON].astype(np.float64)
            vs.append(mb.valence(res).score)
        val[o] = float(np.mean(vs))
    return key, {"valence": val, "mem": mem, "rates": rates}


def _export_spec(_):
    from .teach_surrogate import MBSpec

    mb = _F["mb"]
    return MBSpec.from_mb(mb), np.asarray(mb.KC)


class FlyPool:
    """Pool of worker processes, each holding its own whole-brain fly."""

    def spec(self):
        return self.pool.apply(_export_spec, (None,))

    def measure(self, jobs: list) -> dict:
        return dict(self.pool.imap_unordered(_measure, jobs))

    def episodes_full(self, jobs: list, log_every: int = 0) -> dict:
        out, t0 = {}, time.time()
        for i, (key, rec) in enumerate(self.pool.imap_unordered(_episode_full, jobs), 1):
            out[key] = rec
            if log_every and (i % log_every == 0 or i == len(jobs)):
                print(f"  {i}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
        return out

    def __init__(self, n_workers: int = 4, protocol: dict | None = None):
        from .experiments import DEFAULT_PROTOCOL, load_protocol

        p = protocol or load_protocol(DEFAULT_PROTOCOL)
        self.args = (
            p["odor_scale"],
            p["odor_rate_max_hz"],
            p["odor_min_response"],
            p["reinforcement_rate_hz"],
            p["duration_ms"],
        )
        self.pool = mp.get_context("spawn").Pool(
            n_workers, initializer=_init_fly, initargs=self.args
        )

    def episodes(self, jobs: list, log_every: int = 0) -> dict:
        out, t0 = {}, time.time()
        for i, (key, val, logs, mem) in enumerate(self.pool.imap_unordered(_episode, jobs), 1):
            out[key] = {"valence": val, "logs": logs, "mem": mem}
            if log_every and (i % log_every == 0 or i == len(jobs)):
                print(f"  {i}/{len(jobs)} episodes, {time.time() - t0:.0f} s", flush=True)
        return out

    def close(self):
        self.pool.close()
        self.pool.join()


def run_protocols(
    pool: FlyPool,
    task: Task,
    protocols: dict[str, list[tuple[str, str]]],
    naive: dict[str, float],
    eval_seeds=(101, 102, 103),
    train_seed: int = 5000,
) -> dict[str, Outcome]:
    jobs = [(k, acts, task.panel, train_seed, eval_seeds) for k, acts in protocols.items()]
    res = pool.episodes(jobs)
    out = {}
    for k, acts in protocols.items():
        s, r, g = score(task, res[k]["valence"], naive)
        out[k] = Outcome(acts, res[k]["valence"], naive, s, r, g)
    return out
