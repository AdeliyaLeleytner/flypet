#!/usr/bin/env python3
"""FlyTalk smoke test: one diagnose episode in-process, checked against recorded values.

Starts a small worker pool, measures the untrained fly on the scoring seeds, plays fly 5 (ethyl acetate paired once and acetic acid twice with sugar) with
three measurements and a fixed answer, and compares every number with the values recorded when the benchmark was built
(Brian2 2.10.1, 96-worker server). Exit code 0 means the local install reproduces the benchmark exactly.

    python scripts/flytalk/smoke_test.py --workers 2
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# recorded on the benchmark server (GET /info and a replay of seed 5)
EXPECTED_NAIVE = {
    "ethyl acetate": 0.19391385714213052,
    "methyl acetate": 0.17135003209114075,
    "1-hexanol": 0.29850619037946063,
    "2-heptanone": 0.26589176058769226,
    "hexanal": 0.35133497913678485,
    "acetic acid": 0.15213294327259064,
    "linalool": 0.37562177578608197,
}
RECORD_PATH = Path(__file__).with_name("smoke_expected.json")
ACTIONS = ["SMELL ethyl acetate", "SMELL acetic acid", "SMELL linalool"]
ANSWER = "ANSWER\n" + "\n".join(
    f"{o}: {'+' if o in ('ethyl acetate', 'acetic acid') else 0}" for o in EXPECTED_NAIVE
)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument(
        "--record", action="store_true", help="write the expected values instead of checking them"
    )
    a = ap.parse_args(argv)
    from flypet.flytalk import Env, PANEL
    from flypet.teach import FlyPool

    t0 = time.time()
    pool = FlyPool(a.workers)
    env = Env(pool, PANEL)
    t_init = time.time() - t0
    naive_err = max(abs(env.naive[o] - EXPECTED_NAIVE[o]) for o in PANEL)
    t1 = time.time()
    ep = env.reset("diagnose", 5)
    obs = [env.step(ep.eid, a_) for a_ in ACTIONS]
    t_steps = time.time() - t1
    env.step(ep.eid, ANSWER)
    got = {
        "history": ep.history,
        "values": [t["value"] for t in ep.transcript if "value" in t],
        "score": ep.result["score"],
    }
    print(
        f"workers {a.workers}: start-up and untrained fly {t_init:.0f} s; reset + {len(ACTIONS)} measurements {t_steps:.0f} s"
    )
    print(f"untrained fly max |difference| from the benchmark: {naive_err:.2e}")
    for o in obs:
        print("  " + o)
    if a.record:
        RECORD_PATH.write_text(json.dumps(got, indent=1))
        print(f"recorded {RECORD_PATH}")
        return 0
    exp = json.loads(RECORD_PATH.read_text())
    ok = (
        naive_err < 1e-9
        and [list(h) for h in got["history"]] == [list(h) for h in exp["history"]]
        and all(abs(x - y) < 1e-9 for x, y in zip(got["values"], exp["values"]))
        and abs(got["score"] - exp["score"]) < 1e-12
    )
    print("PASS" if ok else f"FAIL: got {got}, expected {exp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
