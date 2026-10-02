"""Validation-only bridge sensitivity, using frozen token embeddings without loading the full LM."""

import argparse, json, hashlib, io
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from flypet.physiology_language import PhysiologicalLanguageBridge, MODEL_ID, MODEL_REVISION
from flypet.physiology_memory_task import OBJECTS, LOCATIONS
from scripts.train_physiology_language import selection


def main(a):
    torch.set_num_threads(4)
    root = Path(a.results)
    report = json.loads((root / "report.json").read_text())
    c = report["config"]
    snapshot = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        )
    )
    index = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(snapshot / index[key], framework="pt", device="cpu") as f:
        weight = f.get_tensor(key)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION, local_files_only=True)
    events = {}
    for obj in OBJECTS:
        for loc in LOCATIONS:
            text = f"The {obj} moved to the {loc} box."
            ids = tok(text, add_special_tokens=False)["input_ids"]
            events[text] = weight[ids].float().mean(0).cuda()
    del weight
    graph = np.load(Path(a.data) / "graph/graph.npz")
    bridge = (
        PhysiologicalLanguageBridge(
            graph,
            2560,
            report["embedding_std"],
            tokens=c["prefix_tokens"],
            hidden=c["hidden"],
            ticks_per_event=c["ticks_per_event"],
            checkpoint_steps=c["checkpoint_steps"],
            physics_dtype=c["physics_dtype"],
            surrogate_scale=c["surrogate_scale"],
            max_drive_mv=c["max_drive_mv"],
        )
        .cuda()
        .eval()
    )
    rows = selection(
        [json.loads(x) for x in (Path(a.data) / "task/validation.jsonl").read_text().splitlines()],
        48,
    )
    result = {}
    feature = []
    hook = bridge.decoder[0].register_forward_pre_hook(
        lambda m, args: feature.append(args[0].detach().cpu().clone())
    )
    for name in ["initial"] if a.initial_only else ["trainable_core", "frozen_core"]:
        ckpt = root / "initial.pt" if name == "initial" else root / name / "best.pt"
        if not ckpt.exists():
            continue
        checkpoint_bytes = ckpt.read_bytes()
        bridge.load_trainable_state(
            torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
        )
        with torch.no_grad():
            currents = bridge.max_drive_mv * torch.sigmoid(
                bridge.encoder(torch.stack(list(events.values())).float())
            )
            input_diagnostic = {
                "min_mv": float(currents.min()),
                "max_mv": float(currents.max()),
                "mean_std_across_events_mv": float(currents.std(0).mean()),
                "fraction_near_bounds": float(
                    (
                        (currents < 0.01 * bridge.max_drive_mv)
                        | (currents > 0.99 * bridge.max_drive_mv)
                    )
                    .float()
                    .mean()
                ),
            }
        states = {}
        stats = []
        with torch.no_grad():
            for row in rows:
                k = (row["group"], row["assignment"])
                if k in states:
                    continue
                x = torch.stack([events[e] for e in row["history"]])[None]
                feature.clear()
                prefix, s = bridge(x)
                states[k] = (prefix.cpu(), feature[-1])
                stats.append(float(s["state_rms"]))
        selected = list(states)[:4]
        selected_rows = [
            next(row for row in rows if (row["group"], row["assignment"]) == k) for k in selected
        ]
        with torch.no_grad():
            x = torch.stack(
                [torch.stack([events[e] for e in row["history"]]) for row in selected_rows]
            )
            feature.clear()
            batched_prefix, _ = bridge(x)
            batched_features = feature[-1]
        batch_check = {
            "n": len(selected),
            "prefix_max_difference": max(
                float((batched_prefix[i].cpu() - states[k][0][0]).abs().max())
                for i, k in enumerate(selected)
            ),
            "feature_max_difference": max(
                float((batched_features[i] - states[k][1][0]).abs().max())
                for i, k in enumerate(selected)
            ),
        }
        pairs = []
        for (g, assignment), (prefix, features) in states.items():
            if assignment:
                continue
            other, ff = states[(g, 1)]
            pairs.append(
                {
                    "group": g,
                    "feature_rms_difference": float((features - ff).square().mean().sqrt()),
                    "prefix_rms_difference": float((prefix - other).square().mean().sqrt()),
                    "bf16_identical_fraction": float(
                        (prefix.bfloat16() == other.bfloat16()).float().mean()
                    ),
                    "bf16_bit_identical": bool(torch.equal(prefix.bfloat16(), other.bfloat16())),
                }
            )
        result[name] = {
            "checkpoint_sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
            "validation_only": True,
            "input_currents": input_diagnostic,
            "batch4_vs_single": batch_check,
            "state_rms_min_max": [min(stats), max(stats)],
            "pairs": pairs,
        }
    hook.remove()
    (
        root
        / (
            "initial_prefix_sensitivity.json"
            if a.initial_only
            else "validation_prefix_sensitivity.json"
        )
    ).write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                k: {
                    "batch4_vs_single": v["batch4_vs_single"],
                    "input_currents": v["input_currents"],
                    "state_rms": v["state_rms_min_max"],
                    "identical_pairs": sum(x["bf16_bit_identical"] for x in v["pairs"]),
                    "n_pairs": len(v["pairs"]),
                    "prefix_delta_range": [
                        min(x["prefix_rms_difference"] for x in v["pairs"]),
                        max(x["prefix_rms_difference"] for x in v["pairs"]),
                    ],
                }
                for k, v in result.items()
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--initial-only", action="store_true")
    main(p.parse_args())
