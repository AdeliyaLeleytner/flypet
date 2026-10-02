"""Teaching tasks, a vectorised surrogate for search, and reference teachers (random, textbook, beam-search oracle)."""

from __future__ import annotations

import pickle
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy import sparse

from .teach import REINFORCEMENTS, Task, score


def make_tasks(panel: list[str], n_random: int = 20, budget: int = 6, seed: int = 0) -> list[Task]:
    tasks = []
    for o in panel:
        for t in (1, -1):
            tasks.append(
                Task(
                    panel,
                    {p: (t if p == o else 0) for p in panel},
                    budget,
                    name=f"{'+' if t > 0 else '-'}{o}",
                )
            )
    for a, b in permutations(panel, 2):
        tasks.append(
            Task(
                panel,
                {p: (1 if p == a else -1 if p == b else 0) for p in panel},
                budget,
                name=f"+{a} -{b}",
            )
        )
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        pick = rng.choice(len(panel), size=3, replace=False)
        signs = rng.choice([-1, 1], size=3)
        tg = {p: 0 for p in panel}
        for i, s in zip(pick, signs):
            tg[panel[i]] = int(s)
        tasks.append(
            Task(
                panel,
                tg,
                budget,
                name=" ".join(f"{'+' if tg[p] > 0 else '-'}{p}" for p in panel if tg[p]),
            )
        )
    seen, unique = set(), []
    for t in (
        tasks
    ):  # the random draws repeat two target vectors; keep the first copy (74 tasks for seed 0)
        key = tuple(t.targets[p] for p in panel)
        if key not in seen:
            seen.add(key)
            unique.append(t)
    return unique


class FastSurrogate:
    """Same model as teach_surrogate.Surrogate, vectorised over panel probes."""

    def __init__(self, path: Path):
        d = pickle.load(open(path, "rb"))
        spec, kc_index = d["spec"], d["kc_index"]
        self.panel, self.naive, self.budget = d["panel"], d["naive"], d["budget"]
        self.eval_seeds, self.train_seed = tuple(d["eval_seeds"]), d["train_seed"]
        pos = np.full(int(max(kc_index.max(), spec.syn_pre.max())) + 1, -1)
        pos[kc_index] = np.arange(len(kc_index))
        pre = pos[spec.syn_pre]
        mpos = np.full(int(spec.mbon.max()) + 1, -1)
        mpos[spec.mbon] = np.arange(len(spec.mbon))
        post = mpos[spec.syn_post]
        n, m = len(pre), len(spec.mbon)
        self.P = sparse.csr_matrix((np.ones(n), (np.arange(n), post)), shape=(n, m))
        self.sign, self.slopes = spec.sign, d["slopes"]
        self.eta, self.floor, self.kc_scale = spec.eta, spec.floor, spec.kc_scale_hz
        self.probes = [(o, s) for o in self.panel for s in self.eval_seeds]
        self.U = np.stack(
            [spec.base_w * d["probe_kc"][(o, s)][pre] for o, s in self.probes]
        )  # probes x syn
        self.D0 = np.asarray((self.U @ self.P))  # probes x mbon
        self.R0 = np.stack([d["probe_mbon"][p] for p in self.probes])
        self.change = {}
        for (o, r, sd), kc in d["trial_kc"].items():
            a = np.minimum(1.0, kc[pre] / self.kc_scale)
            self.change[(o, r, sd)] = self.eta * a * d["trial_dop"][(o, r, sd)][post]
        self.n_syn = n

    def train(self, mem, odour, reinf, step):
        c = self.change[(odour, reinf, self.train_seed + step)]
        t = c > 0
        mem = mem.copy()
        mem[t] = np.maximum(self.floor, mem[t] * (1.0 - c[t]))
        return mem

    def valences(self, mem) -> dict[str, float]:
        D = np.asarray((self.U * mem) @ self.P)
        R = np.maximum(0.0, self.R0 + self.slopes * (D - self.D0))
        ap, av = R[:, self.sign > 0].sum(1), R[:, self.sign < 0].sum(1)
        v = (ap - av) / (ap + av + 1.0)
        k = len(self.eval_seeds)
        return {o: float(v[i * k : (i + 1) * k].mean()) for i, o in enumerate(self.panel)}

    def run(self, actions) -> dict[str, float]:
        mem = np.ones(self.n_syn)
        for step, (o, r) in enumerate(actions):
            mem = self.train(mem, o, r, step)
        return self.valences(mem)

    def evaluate(self, task: Task, actions) -> tuple[float, float]:
        s, r, _ = score(task, self.run(actions), self.naive)
        return s, r


def textbook(task: Task) -> list[tuple[str, str]]:
    """Pair every 'approach more' odour with reward and every 'avoid more' odour with punishment, round-robin."""
    todo = [(o, "reward" if t > 0 else "punish") for o, t in task.targets.items() if t != 0]
    return [todo[k % len(todo)] for k in range(task.budget)] if todo else []


def random_protocol(task: Task, rng) -> list[tuple[str, str]]:
    acts = [(o, r) for o in task.panel for r in REINFORCEMENTS]
    return [acts[j] for j in rng.integers(len(acts), size=task.budget)]


def beam_topk(
    sur: FastSurrogate, task: Task, width: int = 48, k: int = 16
) -> list[tuple[list, float, float]]:
    """The k best distinct protocols (any length <= budget) seen during the beam search, by the same key."""
    acts = [(o, r) for o in task.panel for r in REINFORCEMENTS]
    seen, pool = set(), []
    beams = [([], np.ones(sur.n_syn))]
    for step in range(task.budget):
        cand = []
        for seq, mem in beams:
            for a in acts:
                m2 = sur.train(mem, a[0], a[1], step)
                s, r, _ = score(task, sur.valences(m2), sur.naive)
                cand.append((s + 0.05 * r, r, s, seq + [a], m2))
        cand.sort(key=lambda x: -x[0])
        for c in cand[: 4 * k]:
            key = tuple(c[3])
            if key not in seen:
                seen.add(key)
                pool.append((c[3], c[1], c[2], c[0]))
        beams = [(c[3], c[4]) for c in cand[:width]]
    pool.sort(key=lambda x: -x[3])
    return [(p, r, s) for p, r, s, _ in pool[:k]]


def beam_search(sur: FastSurrogate, task: Task, width: int = 48) -> tuple[list, float, float]:
    """Best protocol of length <= budget under the surrogate: success first, graded reward as a tie-breaker."""
    acts = [(o, r) for o in task.panel for r in REINFORCEMENTS]
    naive_mem = np.ones(sur.n_syn)
    s0, r0, _ = score(task, sur.valences(naive_mem), sur.naive)
    best = ([], r0, s0)
    beams = [([], naive_mem)]
    for step in range(task.budget):
        cand = []
        for seq, mem in beams:
            for a in acts:
                m2 = sur.train(mem, a[0], a[1], step)
                s, r, _ = score(task, sur.valences(m2), sur.naive)
                cand.append((s + 0.05 * r, r, s, seq + [a], m2))
        cand.sort(key=lambda x: -x[0])
        if cand[0][2] + 0.05 * cand[0][1] > best[2] + 0.05 * best[1]:
            best = (cand[0][3], cand[0][1], cand[0][2])
        beams = [(c[3], c[4]) for c in cand[:width]]
    return best
