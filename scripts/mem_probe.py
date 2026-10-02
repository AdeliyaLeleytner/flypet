_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, os, threading, time, subprocess, gc

sys.path.insert(0, _ROOT + "")
from flypet.catalog import STIMULI
from flypet.engine import Brain, StimInput

rss = lambda: int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 2**20
peak = [0.0]
stop = [False]


def sampler():
    while not stop[0]:
        peak[0] = max(peak[0], rss())
        time.sleep(0.05)


threading.Thread(target=sampler, daemon=True).start()
b = Brain()
print(f"built rss {rss():.2f} GB", flush=True)
for arg in sys.argv[1:]:
    k, _, rate = arg.partition(":")
    rate = float(rate) if rate else STIMULI[k].default_rate_hz
    peak[0] = rss()
    t0 = time.time()
    r = b.run([StimInput(STIMULI[k].resolve(), rate, key=k)], duration_ms=300, seed=1)
    gc.collect()
    print(
        f"{k:14s} wall {r.wall_s:4.1f}s spikes {sum(r.counts.values()):6d} active {len(r.counts):5d} | rss after {rss():.2f} GB, peak during {peak[0]:.2f} GB, monitor spikes {int(b.mon.num_spikes)}",
        flush=True,
    )
stop[0] = True
