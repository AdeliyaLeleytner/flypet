"""Two checks for the FLM analysis on the FlyWire v783 connectome.
1. FLM-style graph: unsigned, incoming-normalised W; input = random ±c[bin]. Pre-activation scale per neuron
   = 0.4 * sqrt(sum_j W_ij^2) (RMS of c is 1). If this is << 1, tanh is linear and the graph is a linear filter.
2. Degree-preserving rewiring of our LIF model (permute postsynaptic targets; keeps every neuron's in/out degree
   and the presynaptic sign): do sugar -> MN9, JO -> grooming, looming -> giant fibre survive?
"""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, time, numpy as np

sys.path.insert(0, _ROOT + "")
from flypet import connectome as C
from flypet.catalog import STIMULI, READOUTS, MN9_RIGHT
from flypet.engine import Brain, StimInput
from flypet import analysis as A

i_pre, i_post, w = C.connections()
n = len(C.completeness())
print(f"connections {len(w)}, neurons {n}, mean unique in-degree {len(w) / n:.0f}")
# --- check 1: FLM-style row-normalised unsigned W
cnt = np.abs(w).astype(np.float64)
in_sum = np.bincount(i_post, weights=cnt, minlength=n)
wn = cnt / np.maximum(in_sum[i_post], 1)  # W_ij = C_ij / sum_k C_ik
row_l2 = np.sqrt(np.bincount(i_post, weights=wn**2, minlength=n))
has_in = in_sum > 0
pre_scale = 0.4 * row_l2[has_in]
print(
    f"FLM-style pre-activation scale 0.4*||W_i||_2: median {np.median(pre_scale):.3f}, 90th pct {np.percentile(pre_scale, 90):.3f}, 99th {np.percentile(pre_scale, 99):.3f}, max {pre_scale.max():.3f}"
)
print(
    f"fraction of neurons with scale < 0.2 (tanh within 1.3% of linear): {(pre_scale < 0.2).mean():.3f}"
)
# simulate one FLM step on our graph with a random hashed input to measure state RMS
rng = np.random.default_rng(0)
bins = rng.integers(0, 128, n)
sign = rng.choice([-1.0, 1.0], n)
c = rng.standard_normal(128)
c /= np.sqrt(np.mean(c * c))
drive = 0.4 * c[bins] * sign
from scipy import sparse

W = sparse.csr_matrix((wn, (i_post, i_pre)), shape=(n, n))
x = np.tanh(W @ drive)
print(
    f"one FLM step on FlyWire: state RMS {np.sqrt(np.mean(x**2)):.4f}, max |x| {np.abs(x).max():.3f}, tanh nonlinearity (RMS of x - preact) {np.sqrt(np.mean((x - (W @ drive)) ** 2)):.2e}"
)


# --- check 2: degree-preserving rewiring of the LIF model
def responses(brain, tag):
    out = {}
    for key, read in [
        ("sugar", "proboscis_extension"),
        ("antenna_touch", "antennal_grooming"),
        ("looming", "escape_takeoff"),
    ]:
        st = STIMULI[key]
        r = brain.run(
            [StimInput(st.resolve(), st.default_rate_hz, key=key)], duration_ms=300, seed=1
        )
        s = A.summarize(r)
        b = s["behaviours"][read]
        out[key] = (
            b["n_active"],
            b["n_neurons"],
            b["max_rate_hz"],
            s["n_active_downstream"],
            s["n_spikes_total"],
        )
        print(
            f"  [{tag}] {key:14s} -> {read}: {b['n_active']}/{b['n_neurons']} active, max {b['max_rate_hz']} Hz | downstream active {s['n_active_downstream']}, spikes {s['n_spikes_total']}",
            flush=True,
        )
    return out


print("--- intact connectome")
brain = Brain()
intact = responses(brain, "intact")
print("--- degree-preserving rewiring (postsynaptic targets permuted, signs and degrees kept)")
perm = np.random.default_rng(1).permutation(len(brain._i_post))
brain._i_post = brain._i_post[
    perm
]  # same multiset of targets -> same in-degrees; per-pre out-degree unchanged
brain._build()
shuffled = responses(brain, "rewired")
