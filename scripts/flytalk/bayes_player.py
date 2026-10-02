#!/usr/bin/env python3
"""Bayes-optimal-ish diagnose player: the ceiling for what can be learned about a fly from 12 measurements.

precompute (machine 2, own worker pool): every history the sampler can produce (799 for seven odours), two replicate
flies each (random trial order and seeds, as in real episodes), every odour probed on four seeds -> per history and
odour a Gaussian (mean, SD over the 8 values, SD floored at 0.015).
play (through the FlyTalk server, seeds 0-59): posterior over histories; next SMELL = odour with the largest posterior
predictive variance; after the budget, ANSWER each odour with its posterior-marginal most probable label.
The likelihood has a 1% uniform outlier component so a wrong table entry cannot zero out the true history.
"""

from __future__ import annotations
import argparse, json, sys, time, urllib.request
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from flypet import flytalk as F

OUT = ROOT / "data/flytalk_20260926"
PANEL = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
]


def precompute(workers, reps=2, probe_seeds=(401, 402, 403, 404)):
    from flypet.teach import FlyPool

    hists = F.all_histories(PANEL)
    jobs = []
    for hi, h in enumerate(hists):
        for r in range(reps):
            rng = np.random.default_rng((hi, r, 17))
            trials = [(o, re) for o, re, n in h for _ in range(n)]
            order = rng.permutation(len(trials))
            seeds = rng.integers(10_000, 90_000, size=len(trials))
            jobs.append(
                (
                    (hi, r),
                    [(trials[i][0], trials[i][1], int(s)) for i, s in zip(order, seeds)],
                    [(o, s) for o in PANEL for s in probe_seeds],
                )
            )
    pool = FlyPool(workers)
    vals = np.zeros((len(hists), reps, len(PANEL), len(probe_seeds)))
    t0 = time.time()
    for i, ((hi, r), out) in enumerate(pool.pool.imap_unordered(F.w_history, jobs), 1):
        vals[hi, r] = np.array(out).reshape(len(PANEL), len(probe_seeds))
        if i % 100 == 0 or i == len(jobs):
            print(f"  {i}/{len(jobs)} replicate flies, {time.time() - t0:.0f} s", flush=True)
    pool.close()
    np.savez(
        OUT / "bayes_table.npz",
        vals=vals,
        histories=np.array([json.dumps(h) for h in hists]),
        panel=np.array(PANEL),
    )


class Player:
    def __init__(self, table_path):
        z = np.load(table_path)
        self.hists = [json.loads(h) for h in z["histories"]]
        v = z["vals"].reshape(len(self.hists), -1, len(PANEL), z["vals"].shape[-1])
        v = v.transpose(0, 2, 1, 3).reshape(len(self.hists), len(PANEL), -1)
        self.mu, self.sd = v.mean(-1), np.maximum(v.std(-1), 0.015)
        self.logprior = np.log([F.history_prior(h, PANEL) for h in self.hists])
        lab = [F.truth_labels(h, PANEL) for h in self.hists]
        self.labels = np.array(
            [[{"+": 0, "-": 1, "0": 2}[l[o]] for o in PANEL] for l in lab]
        )  # hist x odour

    def loglik(self, o, x):
        g = np.exp(-0.5 * ((x - self.mu[:, o]) / self.sd[:, o]) ** 2) / (
            self.sd[:, o] * np.sqrt(2 * np.pi)
        )
        return np.log(0.99 * g + 0.01 * 0.5)

    def choose(self, lp):
        w = np.exp(lp - lp.max())
        w /= w.sum()
        m = w @ self.mu
        var = w @ (self.sd**2 + self.mu**2) - m**2
        return int(np.argmax(var))

    def answer(self, lp):
        w = np.exp(lp - lp.max())
        w /= w.sum()
        probs = np.stack(
            [(w[:, None] * (self.labels == c)).sum(0) for c in range(3)], 1
        )  # odour x label
        return {o: "+-0"[int(np.argmax(probs[i]))] for i, o in enumerate(PANEL)}, probs


def post(url, path, obj):
    req = urllib.request.Request(
        url + path, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def play(url, n, table, threads=20):
    info = json.loads(urllib.request.urlopen(url + "/info", timeout=60).read())
    assert info["panel"] == PANEL, info["panel"]
    P = Player(table)

    def one(seed):
        ep = post(url, "/reset", {"task": "diagnose", "seed": seed})
        lp = P.logprior.copy()
        for _ in range(info["budget"]):
            o = P.choose(lp)
            r = post(url, "/step", {"eid": ep["eid"], "action": f"SMELL {PANEL[o]}"})
            x = float(r["observation"].split(":")[1].split("(")[0])
            lp = lp + P.loglik(o, x)
        lab, probs = P.answer(lp)
        post(
            url,
            "/step",
            {
                "eid": ep["eid"],
                "action": "ANSWER\n" + "\n".join(f"{o}: {l}" for o, l in lab.items()),
            },
        )
        res = post(url, "/result", {"eid": ep["eid"]})
        return {
            "seed": seed,
            **res["result"],
            "history": res["history"],
            "transcript": res["transcript"],
            "map_history": P.hists[int(np.argmax(lp))],
        }

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(
        threads
    ) as ex:  # flies are independent; the server runs episodes in parallel
        rows = sorted(ex.map(one, range(n)), key=lambda r: r["seed"])
    json.dump(rows, open(OUT / "bayes_player.json", "w"), indent=1)
    print(
        f"Bayes player, diagnose, n={n}: score {np.mean([r['score'] for r in rows]):.3f}  exact {np.mean([r['exact'] for r in rows]):.3f}  "
        f"trained right {np.nanmean([r['conditioned_right'] for r in rows]):.3f}  untrained right {np.mean([r['untouched_right'] for r in rows]):.3f}"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["precompute", "play"])
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--url", default="http://127.0.0.1:8800")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--threads", type=int, default=20)
    a = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "precompute":
        precompute(a.workers)
    else:
        play(a.url, a.n, OUT / "bayes_table.npz", a.threads)


if __name__ == "__main__":
    main()
