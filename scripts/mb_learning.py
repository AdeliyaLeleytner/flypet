"""Two flies with different experience: does dopamine-gated KC->MBON plasticity give word-specific,
persistent, generalising valence? Controls: untrained fly, unpaired reward, save/load."""

_ROOT = str(__import__("pathlib").Path(__file__).resolve().parents[1])  # repository root
import sys, time, json, numpy as np

sys.path.insert(0, _ROOT + "")
from flypet.engine import Brain, StimInput
from flypet import corrections
from flypet.mb import MushroomBody, OdorEncoder
from flypet.catalog import STIMULI, MN9_RIGHT
from flypet import analysis as A

b = corrections.apply(Brain())
mb = MushroomBody(b)
enc = OdorEncoder(k_active=3, rate_hz=100.0, frac=1.0)
print(
    f"corrections {b.corrections}; KC->MBON synapses {len(mb.syn_idx)}; PAM {len(mb.PAM)} PPL1 {len(mb.PPL1)}; MBON signs +{sum(1 for s in mb.sign.values() if s > 0)} -{sum(1 for s in mb.sign.values() if s < 0)} 0:{sum(1 for s in mb.sign.values() if s == 0)}"
)
# sanity: validated pathways with corrections
for key, read in [
    ("sugar", "proboscis_extension"),
    ("antenna_touch", "antennal_grooming"),
    ("looming", "escape_takeoff"),
]:
    st = STIMULI[key]
    r = b.run([StimInput(st.resolve(), st.default_rate_hz, key=key)], duration_ms=300, seed=1)
    t = A.readout_table(r)[read]
    print(
        f"  sanity {key:14s} -> {read}: {t['n_active']}/{t['n_neurons']} max {t['max_rate_hz']} Hz"
    )
words = {"W1": "яблоко", "W2": "молоток", "G1": "груша", "G2": "гвоздь", "N": "книга"}
for k, w in words.items():
    print(f"  odour {k} {w:8s} -> glomeruli {enc.glomerular_code(w)}")
print(
    "  glomerular overlap: W1~G1 %.2f  W2~G2 %.2f  W1~W2 %.2f  W1~N %.2f"
    % (
        enc.similarity(words["W1"], words["G1"]),
        enc.similarity(words["W2"], words["G2"]),
        enc.similarity(words["W1"], words["W2"]),
        enc.similarity(words["W1"], words["N"]),
    )
)


def test(label):
    row = {}
    for k, w in words.items():
        r = b.run([enc.stim(w, seed=7)], duration_ms=200, seed=7)
        val = mb.valence(r)
        row[k] = val
    print(
        f"{label:26s} "
        + " | ".join(
            f"{k}:{words[k][:6]:6s} ap {v.approach_hz:5.0f} av {v.avoid_hz:5.0f} s {v.score:+.2f}"
            for k, v in row.items()
        ),
        flush=True,
    )
    return row


def kc_of(w, seed=7):
    return mb.kc_code(b.run([enc.stim(w, seed=seed)], duration_ms=200, seed=seed))


jacc = lambda a, c: len(a & c) / max(1, len(a | c))
kW1, kW1b, kG1, kW2, kN = (
    kc_of("яблоко"),
    kc_of("яблоко", 8),
    kc_of("груша"),
    kc_of("молоток"),
    kc_of("книга"),
)
print(
    f"  KC code: |яблоко| {len(kW1)} J(яблоко,яблоко') {jacc(kW1, kW1b):.2f} J(яблоко,груша) {jacc(kW1, kG1):.2f} J(яблоко,молоток) {jacc(kW1, kW2):.2f} J(яблоко,книга) {jacc(kW1, kN):.2f}"
)
base = test("untrained (fly C)")
r = b.run([enc.stim("яблоко", seed=7)], duration_ms=200, seed=7)
print("  top MBONs for яблоко (untrained):", mb.valence(r).top)


def train(pairs, label):
    mb.reset()
    for w, us in pairs:
        for trial in range(3):
            inputs = [enc.stim(w, seed=10 + trial)] + (
                [mb.reward()] if us == "+" else [mb.punishment()] if us == "-" else []
            )
            r = b.run(inputs, duration_ms=200, seed=10 + trial)
            e = mb.learn(r, note=f"{label}:{w}{us}{trial}")
        print(
            f"  [{label}] {w}{us}: last trial changed {e['n_synapses_changed']} synapses, mean change {e['mean_change']:.2f}, KC active {e['kc_active']}, DAN active {e['dan_active']}, MBONs with dopamine {e['mbons_with_dopamine']}"
        )
    return test(f"{label}")


A_ = train([("яблоко", "+"), ("молоток", "-")], "fly A: яблоко+ молоток-")
mb.save(_ROOT + "/data/memory_flyA.npz")
B_ = train([("яблоко", "-"), ("молоток", "+")], "fly B: яблоко- молоток+")
print("--- unpaired control: same odours and same dopamine, never together")
mb.reset()
for trial in range(3):
    mb.learn(
        b.run([enc.stim("яблоко", seed=10 + trial)], duration_ms=200, seed=10 + trial),
        "unpaired odour",
    )
    mb.learn(b.run([mb.reward()], duration_ms=200, seed=20 + trial), "unpaired reward")
U_ = test("unpaired (odour, reward apart)")
print("--- persistence: reset, load fly A from disk")
mb.reset()
base2 = test("after reset (=untrained)")
mb.load(_ROOT + "/data/memory_flyA.npz")
L_ = test("fly A loaded from disk")
print("\nSUMMARY score (approach-avoid)/(sum+1):")
print(f"{'':26s}" + "".join(f"{words[k]:>10s}" for k in words))
for label, row in [
    ("untrained", base),
    ("fly A яблоко+", A_),
    ("fly B яблоко-", B_),
    ("unpaired", U_),
    ("fly A reloaded", L_),
]:
    print(f"{label:26s}" + "".join(f"{row[k].score:+10.2f}" for k in words))
