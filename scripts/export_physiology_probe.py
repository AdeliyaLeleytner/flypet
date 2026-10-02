"""Export INITIAL physiological states on train/validation only for an information diagnostic."""

import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from safetensors import safe_open
from flypet.physiology_language import PhysiologicalLanguageBridge, MODEL_ID, MODEL_REVISION
from flypet.physiology_memory_task import OBJECTS, LOCATIONS


def main(a):
    torch.set_num_threads(4)
    root = Path(a.results)
    report = json.loads((root / "report.json").read_text())
    c = report["config"]
    snap = Path(
        snapshot_download(
            MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
            allow_patterns=["*.safetensors", "model.safetensors.index.json"],
        )
    )
    index = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    key = "model.embed_tokens.weight"
    with safe_open(snap / index[key], framework="pt", device="cpu") as f:
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
    bridge.load_trainable_state(
        torch.load(root / "initial.pt", map_location="cpu", weights_only=True)
    )
    saved = []
    hook = bridge.decoder[0].register_forward_pre_hook(
        lambda m, args: saved.append(args[0].detach().cpu().numpy().copy())
    )
    arrays = {}
    start = time.monotonic()
    with torch.no_grad():
        for split in ["train", "validation"]:
            rows = [
                json.loads(x)
                for x in (Path(a.data) / "task" / f"{split}.jsonl").read_text().splitlines()
            ]
            unique = {tuple(row["history"]): None for row in rows}
            histories = list(unique)
            for offset in range(0, len(histories), a.batch_size):
                hh = histories[offset : offset + a.batch_size]
                x = torch.stack([torch.stack([events[e] for e in history]) for history in hh])
                saved.clear()
                if a.settle_ticks:
                    currents = bridge.max_drive_mv * torch.sigmoid(bridge.encoder(x.float()))
                    drive = (
                        currents.repeat_interleave(bridge.ticks_per_event, dim=1)
                        .transpose(0, 1)
                        .contiguous()
                    )
                    drive = torch.cat(
                        [drive, drive.new_zeros(a.settle_ticks, len(x), drive.shape[-1])]
                    )
                    state = bridge.core(drive, bridge.input_indices, bridge.output_indices)
                    features = (
                        torch.cat((state["voltage_mv"] / 7.0, state["synaptic_mv"] / 20.0), -1)
                        .cpu()
                        .numpy()
                    )
                else:
                    bridge(x)
                    features = saved[-1]
                for h, feature in zip(hh, features):
                    unique[h] = feature
            arrays[split + "_x"] = np.stack([unique[tuple(row["history"])] for row in rows])
            arrays[split + "_y"] = np.asarray([LOCATIONS.index(row["answer"]) for row in rows])
            arrays[split + "_q"] = np.asarray([OBJECTS.index(row["query_object"]) for row in rows])
            arrays[split + "_last"] = np.asarray([row["query_is_last_mentioned"] for row in rows])
            arrays[split + "_group"] = np.asarray([row["group"] for row in rows])
    hook.remove()
    stem = "initial_state_probe" + (f"_tail{a.settle_ticks}" if a.settle_ticks else "")
    np.savez_compressed(root / (stem + ".npz"), **arrays)
    (root / (stem + "_metadata.json")).write_text(
        json.dumps(
            {
                "seconds": time.monotonic() - start,
                "source": "initial.pt",
                "features": "final physiological v/7,g/20 on output ports; no text embedding bypass",
                "splits": ["train", "validation"],
                "batch_size": a.batch_size,
                "settle_ticks": a.settle_ticks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--results", required=True)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--settle-ticks", type=int, default=0)
    main(p.parse_args())
