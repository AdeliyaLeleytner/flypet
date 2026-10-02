"""Talking to a fly with a hidden past: closed-loop tasks on the whole-brain model.

Before the agent meets it, each fly was conditioned (0-2 panel odours, each paired with sugar (PAM) or shock (PPL1)
1-3 times, batch policy as in case C, random order and seeds). The agent acts one line at a time:

    SMELL <odour>                      probe without learning; returns the MBON approach valence
    TRAIN <odour> + REWARD | PUNISH    a conditioning trial (learning applied)        [erase only]
    EXPOSE <odour>                     the odour alone (learning applied)             [erase only]
    ANSWER + one line per odour        diagnose: "<odour>: +" (sugar), "-" (shock) or "0" (not trained)
    FINISH                             erase: stop and be scored

Scores (balanced, because most odours are untrained and "do nothing" would otherwise look good)
    diagnose  mean of: fraction of trained odours labelled correctly, fraction of untrained odours labelled 0
              (only the second when nothing was trained). Also reported: all seven right, fraction of seven right.
    erase     mean of: fraction of trained odours whose final valence (three fresh probe seeds) is within `delta`
              of the untrained fly, and the same fraction for untrained odours (only the second when nothing was
              trained). Also reported: all seven within delta, fraction of seven.

The simulator is stateless per call: a worker receives an episode's KC->MBON memory factors, runs one trial and
returns the result. Episodes live in Env; `step` is thread-safe per episode, `step_many` runs a batch in parallel.
"""

from __future__ import annotations

import itertools
import re
import threading
from dataclasses import dataclass, field

import numpy as np

from . import teach as T

PANEL = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
]  # fixed before any agent ran

ACTION = re.compile(
    r"^\s*(SMELL|EXPOSE|TRAIN)\s+(.+?)(?:\s*(?:\+|WITH)\s*(REWARD|PUNISH(?:MENT)?))?\s*\.?\s*$",
    re.I,
)
NUMBER = re.compile(r"[-+]?\d*\.\d+|[-+]?\d+")
LABEL = {
    "+": "+",
    "-": "-",
    "−": "-",
    "0": "0",
    "sugar": "+",
    "reward": "+",
    "rewarded": "+",
    "shock": "-",
    "punish": "-",
    "punishment": "-",
    "punished": "-",
    "none": "0",
    "not trained": "0",
    "untrained": "0",
    "nothing": "0",
}


@dataclass
class Episode:
    eid: str
    task: str
    seed: int
    history: list  # [(odour, "reward"|"punish", n_trials)]
    budget: int
    mem: np.ndarray | None
    used: int = 0
    done: bool = False
    transcript: list = field(default_factory=list)  # [{"action": str, "observation": str, ...}]
    result: dict | None = None
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def steps_left(self):
        return self.budget - self.used


def sample_history(panel, rng, p_k=(0.1, 0.45, 0.45)):
    k = int(rng.choice(3, p=p_k))
    odours = [str(o) for o in rng.choice(panel, size=k, replace=False)] if k else []
    return [
        (o, "reward" if rng.random() < 0.5 else "punish", int(rng.integers(1, 4))) for o in odours
    ]


def truth_labels(history, panel):
    lab = {o: "0" for o in panel}
    for o, r, _ in history:
        lab[o] = "+" if r == "reward" else "-"
    return lab


def parse_answer(text, panel):
    names = {p.lower(): p for p in panel}
    pred = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        o = left.strip().strip("-*`> ").lower()
        lab = LABEL.get(right.strip().strip(".`*() ").lower())
        if o in names and lab:
            pred[names[o]] = lab
    return pred


# ---------------------------------------------------------------- stateless worker calls (inside FlyPool workers)
def _load(mem):
    mb = T._F["mb"]
    if mem is None:
        mb.reset()
    else:
        mb.mem = np.asarray(mem, dtype=np.float32).copy()
        mb._push()
    return mb


def w_apply(job):
    """(key, mem|None, [(odour, reinf, seed)]) -> (key, new mem)."""
    key, mem, trials = job
    mb = _load(mem)
    for o, r, s in trials:
        T._trial(o, r, s, learn=True)
    return key, mb.mem.copy()


def w_probe(job):
    """(key, mem|None, odour, seed) -> (key, approach valence)."""
    key, mem, odour, seed = job
    mb = _load(mem)
    res, _ = T._trial(odour, "none", seed, learn=False)
    return key, float(mb.valence(res).score)


class Env:
    def __init__(
        self,
        pool: T.FlyPool,
        panel=PANEL,
        naive=None,
        budget=12,
        delta=0.15,
        eval_seeds=(301, 302, 303),
    ):
        self.pool, self.panel = pool, list(panel)
        self.budget, self.delta, self.eval_seeds = budget, delta, tuple(eval_seeds)
        if naive is None:  # the untrained fly on the scoring seeds
            vals = dict(
                pool.pool.imap_unordered(
                    T_probe_proxy,
                    [((o, s), None, o, s) for o in self.panel for s in self.eval_seeds],
                )
            )
            naive = {o: float(np.mean([vals[(o, s)] for s in self.eval_seeds])) for o in self.panel}
        self.naive = dict(naive)
        self.episodes: dict[str, Episode] = {}
        self._lock = threading.Lock()
        self._ids = itertools.count()

    # -- lifecycle
    def reset(self, task, seed, eid=None):
        return self.reset_many([(task, seed, eid)])[0]

    def reset_many(self, specs):
        """specs: [(task, seed, eid|None)]. Hidden histories are a deterministic function of the seed."""
        eps, jobs = [], []
        for task, seed, eid in specs:
            rng = np.random.default_rng(int(seed))
            hist = sample_history(self.panel, rng)
            trials = [(o, r) for o, r, n in hist for _ in range(n)]
            order = rng.permutation(len(trials))
            seeds = rng.integers(10_000, 90_000, size=len(trials))
            with self._lock:
                eid = eid or f"{task}-{seed}-{next(self._ids)}"
                if eid in self.episodes:
                    raise ValueError(f"episode id {eid} already exists")
            ep = Episode(eid, task, int(seed), hist, self.budget, None)
            eps.append(ep)
            jobs.append(
                (ep.eid, None, [(trials[i][0], trials[i][1], int(s)) for i, s in zip(order, seeds)])
            )
        mems = dict(self.pool.pool.imap_unordered(T_apply_proxy, jobs))
        with self._lock:
            for ep in eps:
                ep.mem = mems[ep.eid]
                self.episodes[ep.eid] = ep
        return eps

    # -- acting
    def parse(self, text):
        m = ACTION.match(text.strip().strip("`*").strip())
        if not m:
            return None
        verb, odour, reinf = (
            m.group(1).upper(),
            m.group(2).strip().strip("\"'").lower(),
            (m.group(3) or "").lower(),
        )
        names = {p.lower(): p for p in self.panel}
        if odour not in names or (verb == "TRAIN" and not reinf):
            return None
        return (
            verb,
            names[odour],
            ("reward" if reinf.startswith("reward") else "punish") if verb == "TRAIN" else "none",
        )

    def _plan(self, ep, text):
        """Validate one action. Returns ('obs', text) for immediate replies, or ('sim', job) for a simulation."""
        first = text.strip().splitlines()[0].strip().strip("`*").upper() if text.strip() else ""
        if ep.done:
            return "obs", "The experiment is over."
        if first.startswith("ANSWER") or first.startswith("FINISH") or first.startswith("DONE"):
            return "finish", text
        act = self.parse(text.strip().splitlines()[0])
        allowed = (
            "SMELL <odour>"
            if ep.task == "diagnose"
            else "SMELL <odour>, TRAIN <odour> + REWARD|PUNISH, EXPOSE <odour>"
        )
        end = "ANSWER" if ep.task == "diagnose" else "FINISH"
        if act is None:
            return "obs", f"Not understood. Use {allowed}, or {end}."
        verb, odour, reinf = act
        if ep.task == "diagnose" and verb != "SMELL":
            return "obs", "In this task you may only SMELL odours, then ANSWER."
        if ep.used >= ep.budget:
            return "obs", f"No actions left. {end} now."
        seed = int(np.random.default_rng((ep.seed, ep.used, 7)).integers(100_000, 900_000))
        ep.used += 1
        if verb == "SMELL":
            return "sim", ("probe", (ep.eid, ep.mem, odour, seed), f"SMELL {odour}")
        return "sim", (
            "apply",
            (ep.eid, ep.mem, [(odour, reinf, seed)]),
            f"TRAIN {odour} + {reinf.upper()}" if verb == "TRAIN" else f"EXPOSE {odour}",
        )

    def step(self, eid, text):
        return self.step_many([(eid, text)])[eid]

    def step_many(self, items):
        """items: [(eid, action text)] -> {eid: observation}. At most one item per episode per call."""
        out, probes, applies, labels, finishes = {}, [], [], {}, []
        for eid, text in items:
            ep = self.episodes[eid]
            kind, payload = self._plan(ep, text)
            if kind == "obs":
                out[eid] = payload
                ep.transcript.append({"action": text.strip(), "observation": payload})
            elif kind == "finish":
                finishes.append((eid, payload))
            else:
                what, job, label = payload
                labels[eid] = label
                (probes if what == "probe" else applies).append(job)
        for eid, v in self.pool.pool.imap_unordered(T_probe_proxy, probes) if probes else []:
            ep = self.episodes[eid]
            obs = f"The fly's approach valence for {labels[eid][6:]}: {v:+.2f} ({ep.steps_left()} actions left)"
            out[eid] = obs
            ep.transcript.append({"action": labels[eid], "observation": obs, "value": v})
        for eid, mem in self.pool.pool.imap_unordered(T_apply_proxy, applies) if applies else []:
            ep = self.episodes[eid]
            ep.mem = mem
            obs = f"Done ({ep.steps_left()} actions left)."
            out[eid] = obs
            ep.transcript.append({"action": labels[eid], "observation": obs})
        if finishes:
            res = self.finish_many(finishes)
            for eid, _ in finishes:
                out[eid] = "Recorded. The experiment is over."
                self.episodes[eid].transcript.append(
                    {"action": dict(finishes)[eid].strip(), "observation": out[eid]}
                )
        return out

    def finish_many(self, items):
        res, probes = {}, []
        for eid, text in items:
            ep = self.episodes[eid]
            ep.done = True
            truth = truth_labels(ep.history, self.panel)
            trained = [o for o in self.panel if truth[o] != "0"]
            if ep.task == "diagnose":
                pred = parse_answer(text, self.panel)
                right = {o: pred.get(o) == truth[o] for o in self.panel}
                cr = float(np.mean([right[o] for o in trained])) if trained else float("nan")
                ur = float(np.mean([right[o] for o in self.panel if o not in trained]))
                ep.result = {
                    "score": 0.5 * (cr + ur) if trained else ur,
                    "exact": all(right.values()),
                    "acc7": float(np.mean(list(right.values()))),
                    "conditioned_right": cr,
                    "untouched_right": ur,
                    "pred": pred,
                    "truth": truth,
                    "n_actions": ep.used,
                }
                res[eid] = ep.result
            else:
                probes += [((eid, o, s), ep.mem, o, s) for o in self.panel for s in self.eval_seeds]
        if probes:
            vals = dict(self.pool.pool.imap_unordered(T_probe_proxy, probes))
            for eid, _ in items:
                ep = self.episodes[eid]
                if ep.task != "erase":
                    continue
                truth = truth_labels(ep.history, self.panel)
                trained = [o for o in self.panel if truth[o] != "0"]
                final = {
                    o: float(np.mean([vals[(eid, o, s)] for s in self.eval_seeds]))
                    for o in self.panel
                }
                ok = {o: abs(final[o] - self.naive[o]) < self.delta for o in self.panel}
                rs = float(np.mean([ok[o] for o in trained])) if trained else float("nan")
                kp = float(np.mean([ok[o] for o in self.panel if o not in trained]))
                ep.result = {
                    "score": 0.5 * (rs + kp) if trained else kp,
                    "exact": all(ok.values()),
                    "acc7": float(np.mean(list(ok.values()))),
                    "restored": rs,
                    "kept": kp,
                    "final": final,
                    "truth": truth,
                    "n_actions": ep.used,
                }
                res[eid] = ep.result
        return res


def w_history(job):
    """(key, [(odour, reinf, seed)], [(odour, probe seed)]) -> (key, [valence per probe]). Train once, probe many."""
    key, trials, probes = job
    mb = _load(None)
    for o, r, s in trials:
        T._trial(o, r, s, learn=True)
    mem = mb.mem.copy()
    out = []
    for o, s in probes:
        _load(mem)
        res, _ = T._trial(o, "none", s, learn=False)
        out.append(float(mb.valence(res).score))
    return key, out


def all_histories(panel):
    """Every history sample_history can produce: 1 + 42 + 756 = 799 for seven odours."""
    from itertools import combinations, product

    out = [[]]
    for k in (1, 2):
        for odours in combinations(panel, k):
            for spec in product(
                [(r, n) for r in ("reward", "punish") for n in (1, 2, 3)], repeat=k
            ):
                out.append([(o, r, n) for o, (r, n) in zip(odours, spec)])
    return out


def history_prior(h, panel, p_k=(0.1, 0.45, 0.45)):
    k = len(h)
    n_hist = {0: 1, 1: len(panel) * 6, 2: len(panel) * (len(panel) - 1) // 2 * 36}[k]
    return p_k[k] / n_hist


def T_apply_proxy(job):  # module-level names so spawn workers can unpickle them
    return w_apply(job)


def T_probe_proxy(job):
    return w_probe(job)


# ---------------------------------------------------------------- what the agent reads
def describe(ep_or_task, env: Env | None = None, budget: int | None = None) -> str:
    task = ep_or_task.task if isinstance(ep_or_task, Episode) else ep_or_task
    budget = budget or (ep_or_task.budget if isinstance(ep_or_task, Episode) else 12)
    naive = env.naive if env else {}
    ref = "\n".join(f"- {o}: {v:+.2f}" for o, v in naive.items())
    common = (
        "You are working with a fruit fly in an olfactory learning experiment. Its response to an odour is measured as an "
        "approach valence: positive means it tends to approach the odour, higher means it likes it more. Pairing an odour "
        "with sugar raises the valence for that odour; pairing it with an electric shock lowers it. Repeated measurements "
        "of the same odour differ by a few hundredths.\n\n"
        "Before you got this fly it may have been trained: up to two of the odours below were each paired with sugar or "
        "with shock, one to three times. You do not know which, if any.\n\n"
        f"Valence of a typical untrained fly:\n{ref}\n\n"
    )
    if task == "diagnose":
        return common + (
            f"Your task: find out how this fly was trained. You have {budget} measurements. Each turn, write exactly one line:\n"
            "SMELL <odour>\n"
            "When you are ready (or out of measurements), write ANSWER on the first line followed by one line per odour:\n"
            "<odour>: +   (paired with sugar)\n<odour>: -   (paired with shock)\n<odour>: 0   (not trained)"
        )
    return common + (
        f"Your task: make this fly respond to every odour like the typical untrained fly again, within {env.delta if env else 0.15:.2f}. "
        f"You have {budget} actions. Each turn, write exactly one line:\n"
        "SMELL <odour>                  (measure; no learning)\n"
        "TRAIN <odour> + REWARD         (pair with sugar)\n"
        "TRAIN <odour> + PUNISH         (pair with shock)\n"
        "EXPOSE <odour>                 (the odour alone)\n"
        "Write FINISH when you are done; the fly is then tested on every odour."
    )


def render_transcript(ep: Episode) -> str:
    return "\n".join(f"> {t['action']}\n{t['observation']}" for t in ep.transcript)
