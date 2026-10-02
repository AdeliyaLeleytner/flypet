#!/usr/bin/env python3
"""Richer neural-to-language training data: two observations, anatomy and QA.

All inspected calibration observations are training only. Validation/test are
the pre-fixed, newly generated mixture recipe groups. Targets are supervision;
neural inputs contain no recipe text, computed valence, memory weights or labels.
"""

from pathlib import Path
import argparse, json, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from flypet import connectome as C
from flypet.latent_reader_data import read_phase, target_caption
from flypet.neural_records import file_hash, write_json, array_hash

QUESTIONS = {
    "audit": "Report the current approach_hz, avoid_hz and valence as JSON. Valence is (approach_hz-avoid_hz)/(approach_hz+avoid_hz+1), positive when approach exceeds avoidance. Use one decimal for rates and three for valence.",
    "summary": "Describe your current neural response in a short sentence.",
    "dynamics": "How did the neural activity develop during this observation?",
    "odor": "Which familiar odor pattern is reflected in the current neural activity?",
    "change": "How did the response change relative to the reference observation?",
    "population": "Which observed neuron population is most active per neuron?",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    sources = [
        ("calibration", Path("data/latent_runtime_20260927_v2"), True),
        ("mixed", Path("data/latent_mixed_20260927_v1"), False),
    ]
    roots = np.load(sources[0][1] / "root_ids.npy")
    ports = np.load(sources[0][1] / "input_model_indices.npy")
    ann = C.annotations().reindex(roots)
    classes = ann.cell_class.astype(str).fillna("unknown").to_numpy()
    names = sorted(set(classes))
    class_ids = np.array([names.index(x) for x in classes])
    eligible = np.ones(len(roots), bool)
    eligible[ports] = False
    group_sizes = np.bincount(class_ids[eligible], minlength=len(names)).astype(np.float32)
    usage = np.zeros(len(roots), np.int64)
    episodes = []
    source_manifests = {}
    for tag, folder, all_train in sources:
        manifest = json.loads((folder / "manifest.json").read_text())
        if (
            manifest["status"] != "complete"
            or not json.loads((folder / "verification.json").read_text())["ok"]
        ):
            raise ValueError("Incomplete source")
        if not np.array_equal(roots, np.load(folder / "root_ids.npy")) or not np.array_equal(
            ports, np.load(folder / "input_model_indices.npy")
        ):
            raise ValueError("Anatomical order differs")
        source_manifests[tag] = file_hash(folder / "manifest.json")
        for dest in sorted((folder / "episodes").iterdir()):
            spec = json.loads((dest / "episode.json").read_text())
            split = "train" if all_train else spec["split"]
            for filename in ["neural.npz", "episode.json"]:
                rel = str((dest / filename).relative_to(folder))
                if file_hash(dest / filename) != manifest["files"][rel]:
                    raise ValueError("Source hash mismatch")
            episodes.append((tag, dest, spec, split))
            if split == "train":
                with np.load(dest / "neural.npz", allow_pickle=False) as z:
                    for phase in (0, 4, 5):
                        start = z["phase_end_tick"][phase - 1] if phase else 0
                        end = z["phase_end_tick"][phase]
                        mask = (z["spike_tick"] >= start) & (z["spike_tick"] < end)
                        usage += np.bincount(z["spike_neuron_index"][mask], minlength=len(roots))
    mandatory = np.flatnonzero(eligible & np.isin(classes, ["MBON", "ALPN"]))
    candidates = np.flatnonzero(eligible & (usage > 0) & ~np.isin(np.arange(len(roots)), mandatory))
    ranked = candidates[np.argsort(-usage[candidates], kind="stable")]
    selected = np.sort(np.r_[mandatory, ranked[: 4096 - len(mandatory)]]).astype(np.int32)
    if len(mandatory) > 4096:
        raise ValueError("Mandatory population exceeds fine-feature budget")
    nt = ann.top_nt.astype(str).to_numpy()
    ct = ann.cell_type.astype(str).to_numpy()
    mbon = classes == "MBON"
    ap = np.flatnonzero(mbon & ((nt == "acetylcholine") | (ct == "MBON11")))
    av = np.flatnonzero(mbon & (nt == "glutamate"))
    metadata = []
    vocab = {}
    for field in ["cell_class", "top_nt", "side"]:
        values = ann.iloc[selected][field].astype(str).tolist()
        vocab[field] = sorted(set(values))
        lookup = {x: i for i, x in enumerate(vocab[field])}
        metadata.append([lookup[x] for x in values])
    fine = []
    pop = []
    targets = []
    state_rows = []
    qa = []
    for number, (tag, dest, spec, split) in enumerate(episodes):
        with np.load(dest / "neural.npz", allow_pickle=False) as z:
            neural = {k: z[k] for k in z.files}
        baseline = len(state_rows)
        local_values = {}
        for phase in (0, 4, 5):
            index = len(state_rows)
            end = neural["phase_end_tick"][phase]
            start = neural["phase_end_tick"][phase - 1] if phase else 0
            choose = (neural["spike_tick"] >= start) & (neural["spike_tick"] < end)
            ids = neural["spike_neuron_index"][choose]
            ticks = neural["spike_tick"][choose] - start
            counts = np.bincount(ids, minlength=len(roots))
            duration = (end - start) * spec["dt_ms"] / 1000
            approach = float(counts[ap].sum() / duration)
            avoid = float(counts[av].sum() / duration)
            valence = (approach - avoid) / (approach + avoid + 1)
            local_values[phase] = (approach, avoid, valence)
            fine.append(read_phase(neural, phase, selected, dt_ms=spec["dt_ms"]))
            allowed = eligible[ids]
            bins = np.minimum(4, ticks[allowed] * 5 // (end - start))
            groups = class_ids[ids[allowed]]
            bins_counts = np.bincount(groups * 5 + bins, minlength=len(names) * 5).reshape(
                len(names), 5
            )
            values = np.zeros((len(names), 9), np.float32)
            values[:, :5] = np.log1p(
                bins_counts / np.maximum(group_sizes[:, None], 1) / (duration / 5)
            )
            means = bins_counts.sum(axis=1) / np.maximum(group_sizes, 1) / duration
            values[:, 5] = np.log1p(means)
            for column, raw in [
                (6, (neural["voltage_mv"][phase] + 52) / 7),
                (7, neural["synaptic_mv"][phase] / 20),
                (8, neural["refractory_remaining_ms"][phase] / 2.2),
            ]:
                pooled = np.bincount(
                    class_ids[eligible], weights=raw[eligible], minlength=len(names)
                ) / np.maximum(group_sizes, 1)
                values[:, column] = (
                    np.sign(pooled) * np.log1p(abs(pooled)) if column == 7 else pooled
                )
            pop.append(values)
            targets.append([approach, avoid, valence])
            row = {
                "id": f"{tag}-{spec['episode']:04d}-p{phase}",
                "state_index": index,
                "baseline_index": baseline,
                "phase": phase,
                "split": split,
                "family": tag + "/" + spec["family"],
            }
            state_rows.append(row)
            contributing = int(counts[ap].sum() + counts[av].sum())
            direction = (
                "approach" if approach > avoid else "avoidance" if avoid > approach else "balanced"
            )
            summary = (
                f"My current neural response leans toward {direction}; it is based on {contributing} MBON output spikes."
                if direction != "balanced"
                else f"My approach-related and avoidance-related outputs are balanced in this observation, with {contributing} output spikes."
            )
            if contributing == 0:
                summary = "There are no MBON output spikes in this observation, so there is no approach or avoidance signal to describe."
            if tag == "calibration":
                odor = spec["odor_a"] if phase in (0, 4) else spec["odor_b"]
            else:
                odor = (
                    f"a mixture of {spec['odors'][0]} and {spec['odors'][1]}"
                    if phase in (0, 4)
                    else spec["odors"][2]
                )
            total = bins_counts.sum(axis=0)
            peak = int(np.argmax(total))
            dynamics = (
                f"Activity peaked in time bin {peak + 1} of five. There were {int(total[0])} downstream spikes in the first bin and {int(total[-1])} in the last."
                if total.sum()
                else "The observed downstream neurons produced no spikes during this observation."
            )
            top = int(np.argmax(means))
            population = (
                f"The {names[top]} population has the highest mean firing rate per observed neuron, about {means[top]:.1f} Hz."
                if means.max() > 0
                else "No observed neuron population is firing in this observation."
            )
            answers = {
                "audit": target_caption([approach, avoid, valence]),
                "summary": summary,
                "dynamics": dynamics,
                "odor": f"The neural pattern corresponds to {odor}.",
                "population": population,
            }
            if phase == 4:
                delta = valence - local_values[0][2]
                trend = (
                    "toward approach"
                    if delta > 0
                    else "toward avoidance"
                    if delta < 0
                    else "not at all"
                )
                answers["change"] = (
                    f"The measured balance changed from {local_values[0][2]:+.2f} to {valence:+.2f}, shifting {trend} relative to the reference observation."
                )
            for kind, answer in answers.items():
                qa.append({**row, "kind": kind, "question": QUESTIONS[kind], "answer": answer})
        if (number + 1) % 128 == 0:
            print(json.dumps({"episodes_prepared": number + 1, "total": len(episodes)}), flush=True)
    fine = np.stack(fine)
    pop = np.stack(pop)
    targets = np.asarray(targets, np.float32)
    train = np.array([i for i, r in enumerate(state_rows) if r["split"] == "train"])
    # One scale per physical channel, shared across neurons. Rare MBON values do
    # not saturate merely because that particular neuron was quiet in training.
    mean = fine[train].mean(axis=(0, 1))
    std = np.maximum(fine[train].std(axis=(0, 1)), 0.1)
    pmean = pop[train].mean(axis=0)
    pstd = np.maximum(pop[train].std(axis=0), 0.1)
    x = np.clip((fine - mean) / std, -20, 20).astype(np.float32)
    px = np.clip((pop - pmean) / pstd, -20, 20).astype(np.float32)
    aux = px[:, :, [5, 6, 7]].reshape(len(px), -1)
    # A pair of model inputs is the unit of duplicate checking.
    seen = {}
    duplicates = []
    for i in sorted(
        range(len(state_rows)),
        key=lambda k: {"train": 0, "validation": 1, "development_test": 2}[state_rows[k]["split"]],
    ):
        base_idx = state_rows[i]["baseline_index"]
        key = array_hash(
            np.concatenate((x[i].ravel(), x[base_idx].ravel(), px[i].ravel(), px[base_idx].ravel()))
        )
        if key in seen and state_rows[seen[key]]["split"] != state_rows[i]["split"]:
            duplicates.append(state_rows[i]["id"])
            state_rows[i]["split"] = "duplicate_control"
        else:
            seen.setdefault(key, i)
    for q in qa:
        q["split"] = state_rows[q["state_index"]]["split"]
    np.savez_compressed(
        a.output / "features.npz",
        x=x,
        population_x=px,
        auxiliary=aux,
        targets=targets,
        neuron_indices=selected,
        neuron_root_ids=roots[selected],
        metadata=np.asarray(metadata, np.int64).T,
        mean=mean,
        std=std,
        population_mean=pmean,
        population_std=pstd,
        population_ids=class_ids,
        population_sizes=group_sizes,
        input_model_indices=ports,
    )
    (a.output / "states.jsonl").write_text("".join(json.dumps(r) + "\n" for r in state_rows))
    (a.output / "qa.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in qa)
    )
    report = {
        "status": "complete",
        "scope": "paired neural observations; new mixture recipe-group evaluation with familiar chemicals",
        "state_counts": {
            s: sum(r["split"] == s for r in state_rows)
            for s in ["train", "validation", "development_test", "duplicate_control"]
        },
        "qa_counts": {
            s: sum(r["split"] == s for r in qa)
            for s in ["train", "validation", "development_test", "duplicate_control"]
        },
        "question_types": QUESTIONS,
        "fine_neurons": len(selected),
        "mandatory_mbon_alpn": len(mandatory),
        "population_names": names,
        "metadata_vocabulary": vocab,
        "auxiliary_features": int(aux.shape[1]),
        "source_manifests": source_manifests,
        "duplicate_exclusions": duplicates,
        "normalization": "train-only shared fine-feature channel scaling plus per-population scaling",
        "writable_inputs_excluded": True,
        "memory_weights_in_input": False,
        "history_text_in_input": False,
        "files": {
            name: file_hash(a.output / name)
            for name in ["features.npz", "states.jsonl", "qa.jsonl"]
        },
        "preparation_code_sha256": file_hash(Path(__file__)),
    }
    write_json(a.output / "manifest.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in [
                    "status",
                    "state_counts",
                    "qa_counts",
                    "fine_neurons",
                    "auxiliary_features",
                ]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
