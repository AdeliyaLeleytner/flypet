#!/usr/bin/env python3
"""Supervision for a hidden-state-to-continuous-neural-drive bridge.

Targets are measured DoOR input profiles and explicit dopamine input profiles.
At inference, the writer receives hidden vectors, not chemical/action labels.
"""

from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from flypet import connectome as C
from flypet.odor import DoorOdor
from flypet.neural_records import file_hash, write_json

ODORS = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
    "geosmin",
]
TEMPLATES = [
    ("train", "Give the fly {intensity} exposure to {odor}. {reinforcement}"),
    ("train", "Let me smell {odor}, at {intensity} intensity. {reinforcement}"),
    ("train", "Present {intensity} {odor} to the simulated fly. {reinforcement}"),
    ("train", "The sensory input should be {intensity} {odor}. {reinforcement}"),
    ("train", "Подай мухе {odor}; интенсивность: {intensity}. {reinforcement}"),
    ("train", "Experiment: {intensity} {odor}. {reinforcement}"),
    ("validation", "Offer a {intensity} sample of {odor}. {reinforcement}"),
    (
        "development_test",
        "I would like the fly to experience {odor} with a {intensity} signal. {reinforcement}",
    ),
]
REINFORCEMENTS = [
    ("none", "No reward or punishment.", 0, 0),
    ("reward", "Pair it with a sugar reward.", 60, 0),
    ("punish", "Pair it with an aversive punishment.", 0, 60),
]
INTENSITIES = [("faint", 0.4), ("moderate", 1.0), ("strong", 1.4)]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    source = Path("data/latent_runtime_20260927_v2")
    roots = np.load(source / "input_root_ids.npy")
    positions = {int(v): i for i, v in enumerate(roots)}
    door = DoorOdor()
    names = []
    columns = []
    for glom in door.glomeruli:
        ids = [positions[int(i)] for i in door.gl2ids[glom] if int(i) in positions]
        if not ids:
            continue
        col = np.zeros(len(roots), np.float32)
        col[ids] = 1
        columns.append(col)
        names.append("glomerulus:" + glom)
    ann = C.annotations()
    for kind in ("PAM", "PPL1"):
        ids = ann.index[
            (ann.cell_class.astype(str) == "DAN") & ann.cell_type.astype(str).str.startswith(kind)
        ]
        col = np.zeros(len(roots), np.float32)
        col[[positions[int(i)] for i in ids if int(i) in positions]] = 1
        columns.append(col)
        names.append("dopamine:" + kind)
    basis = np.stack(columns, axis=1)
    if not np.all(basis.sum(axis=1) == 1):
        raise ValueError("Every writable neuron must have exactly one anatomical channel")
    profiles = {}
    for odor in ODORS:
        full = np.zeros(len(roots), np.float32)
        for inp in door.stim(odor):
            full[[positions[int(i)] for i in inp.ids]] = inp.rate_hz
        rates = (basis.T @ full) / basis.sum(axis=0)
        if not np.allclose(basis @ rates, full):
            raise ValueError("Channel map does not reproduce reference input")
        profiles[odor] = rates
    rows = []
    targets = []

    def add(text, split, target, group):
        rows.append({"id": f"w{len(rows):05d}", "text": text, "split": split, "group": group})
        targets.append(target.astype(np.float32))

    for odor in ODORS:
        for intensity, scale in INTENSITIES:
            for reward, text, pam, ppl in REINFORCEMENTS:
                target = profiles[odor] * scale
                target[-2:] = [pam, ppl]
                for j, (split, template) in enumerate(TEMPLATES):
                    add(
                        template.format(intensity=intensity, odor=odor, reinforcement=text),
                        split,
                        target,
                        f"single-{odor}-{intensity}-{reward}-t{j}",
                    )
    for i, odor in enumerate(ODORS):
        other = ODORS[(i + 1) % len(ODORS)]
        for fraction in (0.25, 0.5, 0.75):
            mixture = f"a mixture of {round(100 * fraction)}% {odor} and {round(100 * (1 - fraction))}% {other}"
            for reward, text, pam, ppl in REINFORCEMENTS:
                target = fraction * profiles[odor] + (1 - fraction) * profiles[other]
                target[-2:] = [pam, ppl]
                for j, (split, template) in enumerate(TEMPLATES):
                    if i == 0 or (i in (2, 5) and fraction == 0.5):
                        split = "transfer_test"
                    add(
                        template.format(intensity="moderate", odor=mixture, reinforcement=text),
                        split,
                        target,
                        f"mix-{i}-{fraction}-{reward}-t{j}",
                    )
    for odor in ODORS:
        target = profiles[odor] * 0.7
        for split, template in TEMPLATES:
            add(
                template.format(
                    intensity="light (70% of normal strength)",
                    odor=odor,
                    reinforcement="No reward or punishment.",
                ),
                "transfer_test",
                target,
                f"unseen-strength-{odor}",
            )
    for odor in ODORS:
        for split, template in [
            ("train", "How do you respond to {odor}?"),
            ("train", "Let us find out how {odor} affects you."),
            ("validation", "Try {odor} and describe your response."),
            ("development_test", "What happens in your brain when you smell {odor}?"),
        ]:
            add(template.format(odor=odor), split, profiles[odor], f"query-{odor}-{split}")
    neutral = [
        ("train", "Describe your current neural response."),
        ("train", "How are you responding now?"),
        ("train", "Что изменилось в твоей реакции?"),
        ("train", "Observe your current state without a new stimulus."),
        ("train", "Tell me about the current activity."),
        ("train", "Do you have any response to describe?"),
        ("train", "No new stimulus; report the current state."),
        ("train", "Pause the external stimulation."),
        ("validation", "Tell me how you react now."),
        ("development_test", "What can you say about your present response?"),
    ]
    for split, text in neutral:
        for prefix in (
            "",
            "Please: ",
            "For now: ",
            "Next: ",
            "A question: ",
            "I want to know: ",
            "Observe only. ",
            "Without giving a new stimulus: ",
        ):
            add(prefix + text, split, np.zeros(len(names)), f"neutral-{len(rows)}")
    (a.output / "samples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    )
    np.savez_compressed(
        a.output / "targets.npz", rates_hz=np.stack(targets), basis=basis, input_root_ids=roots
    )
    matrix = np.stack(targets)
    seen = {v.tobytes() for r, v in zip(rows, matrix) if r["split"] == "train"}
    transfer_overlap = sum(
        v.tobytes() in seen for r, v in zip(rows, matrix) if r["split"] == "transfer_test"
    )
    if transfer_overlap:
        raise ValueError("Transfer targets overlap training")
    manifest = {
        "status": "complete",
        "scope": "familiar odorants; separate held wording and unseen mixture/strength target tests",
        "samples": len(rows),
        "channels": names,
        "counts": {
            s: sum(r["split"] == s for r in rows)
            for s in ["train", "validation", "development_test", "transfer_test"]
        },
        "transfer_target_train_overlap": transfer_overlap,
        "inference_inputs": "frozen LLM intermediate hidden states only",
        "reference": "measured DoOR profiles plus declared PAM/PPL1 inputs",
        "input_root_ids_file_sha256": file_hash(source / "input_root_ids.npy"),
        "files": {name: file_hash(a.output / name) for name in ["samples.jsonl", "targets.npz"]},
    }
    write_json(a.output / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ["samples", "counts"]}))


if __name__ == "__main__":
    main()
