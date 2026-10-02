"""Run a set of stimuli through the whole-brain model and print the behaviour readouts."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, json, time

sys.path.insert(0, _ROOT + "")
from flypet.engine import Brain, StimInput
from flypet.catalog import STIMULI
from flypet import analysis as A

brain = Brain()
print(f"brain ready: {brain.n} neurons, {brain.n_syn} synapses", flush=True)
runs = [
    ("sugar", [("sugar", None)]),
    ("bitter", [("bitter", None)]),
    ("sugar+bitter", [("sugar", None), ("bitter", None)]),
    ("water", [("water", None)]),
    ("antenna_touch", [("antenna_touch", None)]),
    ("looming", [("looming", None)]),
    ("smell_vinegar", [("smell_vinegar", None)]),
    ("light", [("light", None)]),
    ("silence", []),
]
results = {}
for name, spec in runs:
    inputs = [
        StimInput(STIMULI[k].resolve(side=side), STIMULI[k].default_rate_hz, key=k)
        for k, side in spec
    ]
    res = brain.run(inputs, duration_ms=300, seed=1)
    s = A.summarize(res)
    results[name] = s
    print(
        f"\n===== {name}: wall {s['wall_s']}s, downstream active {s['n_active_downstream']}, spikes {s['n_spikes_total']}, by class {s['active_by_super_class']}",
        flush=True,
    )
    print(A.behaviours_text(s), flush=True)
    print(
        "top downstream:",
        [(r["cell_type"], r["side"][:1], r["rate_hz"]) for r in s["top_downstream"][:10]],
        flush=True,
    )
json.dump(
    results, open(_ROOT + "/data/demo_stimuli_results.json", "w"), indent=1, ensure_ascii=False
)
print("\nsaved data/demo_stimuli_results.json")
