_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, os, tracemalloc, subprocess

sys.path.insert(0, _ROOT + "")
from brian2 import ms, Hz
from flypet.catalog import STIMULI
from flypet.engine import Brain, StimInput

rss = lambda: int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())])) / 2**20
b = Brain()
k = "smell_vinegar"
ids = [b.flyid2i[i] for i in STIMULI[k].resolve() if i in b.flyid2i]
b._flush()
b.neu.r_stim[ids] = 60 * Hz
b.neu.rfc[ids] = 0 * ms
b.stim_ops[1].active = True
tracemalloc.start(10)
print("before run rss", round(rss(), 2), flush=True)
b.net.run(300 * ms)
print(
    "after run rss",
    round(rss(), 2),
    "tracemalloc current MB",
    round(tracemalloc.get_traced_memory()[0] / 2**20),
    "peak MB",
    round(tracemalloc.get_traced_memory()[1] / 2**20),
    flush=True,
)
snap = tracemalloc.take_snapshot()
for st in snap.statistics("traceback")[:6]:
    print(f"--- {st.size / 2**20:.0f} MB, {st.count} blocks")
    for line in st.traceback.format()[-4:]:
        print("   ", line)
