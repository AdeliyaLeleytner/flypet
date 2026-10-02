"""Check the persistent engine reproduces the known responses; measure time and memory."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, time, os, subprocess, resource

sys.path.insert(0, _ROOT + "")
from flypet.catalog import STIMULI, MN9_RIGHT
from flypet.engine import Brain, StimInput
from flypet import analysis as A

maxrss = lambda: resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
rss = lambda: int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 2**20
t0 = time.time()
b = Brain()
print(
    f"build {b.build_s}s total {time.time() - t0:.1f}s, n={b.n}, tiers={b.tier_ends}, syn={b.n_syn}, rss {rss():.2f} GB (peak {maxrss():.2f})",
    flush=True,
)
for name, spec in [
    ("sugar", ["sugar"]),
    ("sugar", ["sugar"]),
    ("sugar+bitter", ["sugar", "bitter"]),
    ("antenna_touch", ["antenna_touch"]),
    ("looming", ["looming"]),
    ("smell_vinegar", ["smell_vinegar"]),
    ("light", ["light"]),
    ("silence", []),
    ("sugar", ["sugar"]),
]:
    inputs = [StimInput(STIMULI[k].resolve(), STIMULI[k].default_rate_hz, key=k) for k in spec]
    r = b.run(inputs, duration_ms=300, seed=1)
    s = A.summarize(r)
    beh = {
        k: v["max_rate_hz"]
        for k, v in s["behaviours"].items()
        if v["n_active"] and k not in ("other_motor", "descending_all")
    }
    print(
        f"{name:14s} wall {r.wall_s:5.1f}s active {s['n_active_downstream']:5d} spikes {s['n_spikes_total']:6d} MN9_R {r.rate(MN9_RIGHT):5.1f}Hz | {beh} | rss {rss():.2f} GB",
        flush=True,
    )
# silencing check: silence MN9's strongest presynaptic partner? just silence sugar GRNs themselves -> MN9 should drop
r = b.run(
    [StimInput(STIMULI["sugar"].resolve(), 150, key="sugar")],
    duration_ms=300,
    seed=1,
    silence=STIMULI["sugar"].resolve(),
)
print(
    f"sugar with GRN output silenced: MN9_R {r.rate(MN9_RIGHT):.1f} Hz (expect 0); then normal again:",
    end=" ",
)
r = b.run([StimInput(STIMULI["sugar"].resolve(), 150, key="sugar")], duration_ms=300, seed=1)
print(f"{r.rate(MN9_RIGHT):.1f} Hz")
