#!/usr/bin/env python3
"""Export verified own-training artifacts to safe tensor bundles for inference."""

from pathlib import Path
import argparse, json, shutil, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from safetensors.torch import save_file
from flypet.neural_records import file_hash, write_json


def verify_result(folder):
    marker = folder.parent / "SHA256.json"
    if not marker.is_file():
        raise ValueError("Verified result hash manifest is required")
    hashes = json.loads(marker.read_text())
    for relative, want in hashes.items():
        path = folder.parent / relative
        if file_hash(path) != want:
            raise ValueError("Training artifact hash mismatch")
    report = json.loads((folder / "report.json").read_text())
    if report.get("status") != "complete":
        raise ValueError("Incomplete training result")
    return report


def load_numpy_checkpoint(path):
    # The training checkpoint uses only tensors, primitive values and a numeric
    # NumPy metadata array. Public inference bundles use safetensors instead.
    import numpy._core.multiarray

    allowed = [np.ndarray, np.dtype, numpy._core.multiarray._reconstruct, type(np.dtype("int64"))]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reader", type=Path)
    p.add_argument("--writer", type=Path)
    p.add_argument("--dopamine-head", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "model": "Qwen/Qwen3-4B",
        "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
    }
    if a.writer:
        report = verify_result(a.writer)
        choice = min(
            ["ridge", "mlp"], key=lambda k: report["metrics"]["validation"][k]["port_mae_hz"]
        )
        shutil.copy2(a.writer / "preprocessing.npz", a.output / "writer_preprocessing.npz")
        if choice == "ridge":
            with np.load(a.writer / "ridge.npz", allow_pickle=False) as z:
                tensors = {k: torch.from_numpy(z[k].copy()).contiguous() for k in z.files}
            config = {"kind": "ridge"}
        else:
            checkpoint = torch.load(a.writer / "writer.pt", map_location="cpu", weights_only=True)
            tensors = {k: v.contiguous() for k, v in checkpoint["state_dict"].items()}
            config = {k: checkpoint[k] for k in ["dim", "channels", "hidden"]}
            config["kind"] = "mlp"
        save_file(tensors, str(a.output / "writer.safetensors"))
        config.update(
            pooling=report["pooling"],
            middle_layer=report["middle_layer"],
            max_rate_hz=300,
            channels=report["channels"],
            base_adapter_disabled=True,
        )
        if a.dopamine_head:
            if choice != "ridge":
                raise ValueError("Dopamine calibration requires its ridge sensory map")
            head_report = json.loads((a.dopamine_head / "report.json").read_text())
            if head_report.get("status") != "complete":
                raise ValueError("Incomplete dopamine calibration")
            provenance = json.loads((a.dopamine_head / "provenance.json").read_text())
            for name, digest in provenance["fit_files"].items():
                if file_hash(a.writer / name) != digest:
                    raise ValueError("Dopamine head was trained with a different writer")
            for name, digest in provenance["head_files"].items():
                if file_hash(a.dopamine_head / name) != digest:
                    raise ValueError("Dopamine head hash mismatch")
            shutil.copy2(
                a.dopamine_head / "dopamine_head.safetensors",
                a.output / "dopamine_head.safetensors",
            )
            config["dopamine_head"] = json.loads((a.dopamine_head / "config.json").read_text())
            manifest["dopamine_report_sha256"] = file_hash(a.dopamine_head / "report.json")
        write_json(a.output / "writer_config.json", config)
        manifest["writer_report_sha256"] = file_hash(a.writer / "report.json")
    if a.reader:
        report = verify_result(a.reader)
        checkpoint = load_numpy_checkpoint(a.reader / "reader.pt")
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in checkpoint["state_dict"].items()},
            str(a.output / "reader.safetensors"),
        )
        shutil.copy2(a.reader / "preprocessing.npz", a.output / "reader_preprocessing.npz")
        shutil.copy2(a.reader / "dataset_manifest.json", a.output / "reader_dataset_manifest.json")
        paired = "n_populations" in checkpoint
        fields = ["embedding_dim", "embedding_std", "n_neurons", "channels"]
        config = {k: checkpoint[k] for k in fields}
        config["metadata"] = np.asarray(checkpoint["metadata"]).tolist()
        config.update(
            kind="paired_dialogue" if paired else "numeric",
            tokens=checkpoint.get("tokens", 16),
            width=checkpoint.get("width", 128),
        )
        config["instruction_augmented"] = bool(report.get("config", {}).get("paraphrases", False))
        config["separate_context"] = bool(checkpoint.get("separate_context", False))
        if paired:
            config.update(
                n_populations=checkpoint["n_populations"], auxiliary_dim=checkpoint["auxiliary_dim"]
            )
            if checkpoint.get("architecture") in ("focused", "codec"):
                config.update(
                    {
                        k: checkpoint[k]
                        for k in ["architecture", "mbon_positions", "extra_mbon_tokens", "system"]
                    }
                )
                if checkpoint["architecture"] == "codec":
                    config.update({k: checkpoint[k] for k in ["codec_tokens", "codec_width"]})
            if (a.reader / "lora").exists():
                shutil.copytree(a.reader / "lora", a.output / "reader_lora")
                config["lora_directory"] = "reader_lora"
        else:
            config["question"] = (
                "Read the supplied neural state of a simulated fruit fly. Report one JSON object with exactly "
                'these numeric fields: "approach_hz", "avoid_hz", "valence". The first two are summed MBON '
                "population firing rates; valence is their normalized balance, between -1 and 1. "
                "Use one decimal place for rates and three for valence. Return JSON only."
            )
        write_json(a.output / "reader_config.json", config)
        manifest["reader_report_sha256"] = file_hash(a.reader / "report.json")
    manifest["files"] = {
        str(x.relative_to(a.output)): file_hash(x) for x in a.output.rglob("*") if x.is_file()
    }
    write_json(a.output / "manifest.json", manifest)
    print(json.dumps({"output": str(a.output), "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()
