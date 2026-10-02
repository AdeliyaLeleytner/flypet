#!/usr/bin/env python3
"""Attach a source-grounded model card to a verified local inference bundle."""

import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.neural_records import file_hash, write_json, verify_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "reader_report", "writer_report", "out"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    p.add_argument("--fresh-report", type=Path)
    p.add_argument("--twins-report", type=Path)
    a = p.parse_args()
    manifest = verify_bundle(a.bundle)
    a.out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(a.bundle, a.out, dirs_exist_ok=True)
    reader = json.loads(a.reader_report.read_text())
    writer = json.loads(a.writer_report.read_text())
    config = json.loads((a.bundle / "reader_config.json").read_text())
    if reader["status"] != "complete" or writer["status"] != "complete":
        raise ValueError("Incomplete model report")
    values = reader["metrics"]["development_test/neural"]
    donor = reader["metrics"]["development_test/donor"]
    core = values["mae_valid"]
    comparison = donor["mae_valid"]
    extras = ""
    qualification = "Experimental checkpoint; final release qualification is pending."
    reports = a.out / "reports"
    reports.mkdir()
    for name, path in [
        ("reader", a.reader_report),
        ("writer", a.writer_report),
        ("fresh_reader", a.fresh_report),
        ("fresh_twins", a.twins_report),
    ]:
        if path:
            shutil.copy2(path, reports / (name + ".json"))
    if a.fresh_report:
        fresh = json.loads(a.fresh_report.read_text())
        if not fresh["panel_gate_pass"]:
            qualification = "Experimental checkpoint; not all declared release checks passed. See the per-class results and gates in the fresh report."
        extras = (
            "\n## Fresh evaluation\n\nThe final panel contains "
            + str(fresh["recipe_groups"])
            + " new recipe groups, with two noise seeds per group. "
            "Its preprocessing is fixed from training. Model selection was locked before reading outcomes.\n\n"
            f"Current-valence MAE (including invalid-answer penalties): **{fresh['current_mae_with_invalid_penalty']:.3f}**. "
            f"Donor-state control: **{fresh['donor_mae_with_invalid_penalty']:.3f}**. "
            f"Change MAE: **{fresh['change']['mae_with_invalid_penalty']:.3f}**. "
            f"Summary balanced direction accuracy: **{fresh['summary_balanced_accuracy']:.1%}**.\n\n"
            "Full counts, invalid replies, controls and clustered uncertainty: [fresh report](reports/fresh_reader.json).\n"
        )
        extras += "\n| Measured current direction | Cases | Accuracy |\n|---|---:|---:|\n"
        for sign, name in [("-1", "Avoidance"), ("0", "Exactly neutral"), ("1", "Approach")]:
            if sign in fresh["summary_by_sign"]:
                row = fresh["summary_by_sign"][sign]
                extras += f"| {name} | {row['n']} | {row['accuracy']:.1%} |\n"
    if a.twins_report:
        twins = json.loads(a.twins_report.read_text())
        extras += f"\nFresh live twins: {twins['correct_sign']}/{twins['n']} summary directions matched the measured signed readout. See [the full report](reports/fresh_twins.json).\n"
        if a.fresh_report and fresh["panel_gate_pass"] and twins["gate_pass"]:
            qualification = "Passed the declared fresh-panel and live-twins release checks."
    card = f"""---
license: other
license_name: component-specific-licenses
license_link: LICENSES.md
base_model: Qwen/Qwen3-4B
library_name: pytorch
language:
- en
tags:
- neuroscience
- drosophila
- neural-interface
- lora
- safetensors
---

# Flypet Neural Link · Qwen3-4B bridges

Learned bridges between a continuing FlyWire brain simulation and the internal representations of Qwen3-4B.
This bundle contains the neural reader, a small language adapter, a continuous writer, and preprocessing
fitted on the training data. Download the base language model separately at the pinned revision below.

**Status:** {qualification}

## Mechanism

- Text is encoded into contextual middle/final Qwen hidden states. Learned maps produce continuous ORN and dopamine input rates.
- A private Brian2 worker carries the fly's neural and synaptic state between interactions.
- Neural observations become soft tokens for Qwen. Current-response questions use the current observation alone; comparison questions can use reference, current and difference representations. No stimulus labels, exposure-history text or memory weights are supplied to the reader.
- The codec variant retains global neural tokens and adds a pretrained MBON representation, an internal hidden representation and its reference/current difference. The three scalar auxiliary outputs are not inputs to language generation.
- The writer disables the reader's LoRA while obtaining its contextual states. A model lock protects the adapter switch.

Architecture: `{config.get("architecture", "paired")}`. Base: `Qwen/Qwen3-4B`, revision `{manifest["revision"]}`.
The simulator uses 138,639 FlyWire neurons; the reader samples 4,096 downstream neurons, including all 96 MBONs and 685 ALPNs,
plus summaries of 51 anatomical populations. This is compressed observation, not a complete brain restart checkpoint.

## Development results

These development data were inspected during model repair. They are not the final held-out result.

| Current readout | Neural input | Donor-state input |
|---|---:|---:|
| Valid JSON | {values["valid_fraction"]:.1%} | {donor["valid_fraction"]:.1%} |
| Approach-rate MAE | {core["approach_hz"]:.1f} Hz | {comparison["approach_hz"]:.1f} Hz |
| Avoidance-rate MAE | {core["avoid_hz"]:.1f} Hz | {comparison["avoid_hz"]:.1f} Hz |
| Normalized MBON-balance MAE | {core["valence"]:.3f} | {comparison["valence"]:.3f} |

See [the reader report](reports/reader.json) and [the writer report](reports/writer.json).
{extras}
## Use

Install Flypet's language-model dependencies, fetch the checked simulation inputs, and point the live app at this folder:

```bash
python scripts/fetch_data.py brain
python -m flypet.latent_web --bundle /path/to/this/bundle
```

The model is loaded through `flypet.latent_inference.LatentModels`; this is not a standalone `AutoModelForCausalLM` checkpoint.
The web app runs at `http://127.0.0.1:8775`. CUDA and Apple MPS are supported by the inference code; each published host still needs its own runtime check.

## Scope

The writer covers eight familiar odorants, mixtures, intensities and reinforcement requests. The reader's intended scope is current neural response and change from a reference; qualification is stated above.
Exact odor identification, fine temporal-bin reconstruction, arbitrary sensory semantics and subjective experience are not established.
The approach/avoidance balance is a model-specific MBON readout, not measured animal behavior. Its spike counts refer to the contributing signed MBON subset.
The fly animation is illustrative and does not simulate a physical body. Sessions and synaptic memories are temporary and isolated.

The base LLM and interface weights remain fixed during an ordinary visitor session. The fly's mushroom-body synapses can learn online.
Continuous-agent SFT/GRPO is a separate experiment and must not be inferred from this reader/writer bundle.

## Artifacts and provenance

Inference tensors use safetensors; preprocessing uses numeric NPZ with pickle disabled. `manifest.json` verifies the inference files.
Model repair kept failed checkpoints and donor controls. The writer's dopamine calibration was checked by its actual effect in the full simulator.
Source code, training scripts, data licenses and reproduction instructions are part of the Flypet release.
"""
    (a.out / "README.md").write_text(card)
    (a.out / "LICENSES.md").write_text("""# Component-specific terms

The Qwen3 base model keeps its upstream Apache 2.0 license and is downloaded separately.
The learned bridge and adapter tensor parameters are released under Apache 2.0, consistent with the project's model artifacts.
Flypet source code is MIT. Anatomical neuron identifiers, class mappings and simulation-derived preprocessing retain the FlyWire data terms (CC BY-NC 4.0).
DoOR source response tables and adapted odor profiles retain CC BY-SA 4.0. Bundling these components does not remove their individual terms.
See THIRD_PARTY_NOTICES.md for sources, modifications and citations. This bundle does not grant broader rights to the underlying data.
""")
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", a.out / "THIRD_PARTY_NOTICES.md")
    write_json(
        a.out / "CARD_PROVENANCE.json",
        {
            "reader_report_sha256": file_hash(a.reader_report),
            "writer_report_sha256": file_hash(a.writer_report),
            "inference_manifest_sha256": file_hash(a.bundle / "manifest.json"),
            "publication_performed": False,
        },
    )
    manifest["files"] = {
        p.relative_to(a.out).as_posix(): file_hash(p)
        for p in a.out.rglob("*")
        if p.is_file() and p.name != "manifest.json"
    }
    write_json(a.out / "manifest.json", manifest)
    print(
        json.dumps({"output": str(a.out), "fresh_evaluation_included": a.fresh_report is not None})
    )


if __name__ == "__main__":
    main()
