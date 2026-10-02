#!/usr/bin/env python3
"""Build an allowlisted private-preview Space folder. Does not upload or publish."""

import argparse, json, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet.neural_records import file_hash, write_json, verify_bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    manifest = verify_bundle(a.bundle)
    a.out.mkdir(parents=True, exist_ok=False)
    modules = [
        "__init__",
        "engine",
        "connectome",
        "mb",
        "odor",
        "corrections",
        "neural_runtime",
        "neural_records",
        "latent_observation",
        "latent_reader_data",
        "latent_bridge",
        "latent_dialogue",
        "latent_questions",
        "neural_codec",
        "latent_writer",
        "latent_inference",
        "latent_sessions",
        "latent_web",
    ]
    paths = [f"flypet/{name}.py" for name in modules] + [
        "flypet/static/latent.html",
        "scripts/fetch_data.py",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ]
    for name in paths:
        target = a.out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copytree(a.bundle, a.out / "bundle")
    shutil.copy2(ROOT / "release/latent.Dockerfile", a.out / "Dockerfile")
    shutil.copy2(ROOT / "release/latent-requirements.txt", a.out / "requirements-space.txt")
    data = json.loads((ROOT / "data/manifest.json").read_text())
    data["files"] = [r for r in data["files"] if r["group"] == "brain"]
    data["derived"] = [r for r in data["derived"] if r["group"] == "brain"]
    (a.out / "data").mkdir()
    write_json(a.out / "data/manifest.json", data)
    (a.out / "README.md").write_text("""---
title: Flypet Neural Link
emoji: 🪰
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---

# Flypet Neural Link

A learned two-way interface between a continuing FlyWire brain simulation and Qwen3-4B.
Two independent flies, continuous language-to-neuron stimulation, neural-token replies,
and a controlled state-swap comparison. Sessions and synaptic memory are temporary.

This folder is a private preview build. Publication is pending author review.
The model uses simulated neural activity; answers do not establish animal experiences.
The animation visualizes activity and does not simulate a physical body.

Requires a CUDA GPU. Start with one L4 for the measured deployment check. Smaller GPUs
and other numeric formats require a separate runtime check before claiming support.
No private pet memory, training logs or credentials are included. Source data and models
are downloaded from pinned upstream versions during the build; see THIRD_PARTY_NOTICES.md.
""")
    files = {
        str(path.relative_to(a.out)): file_hash(path) for path in a.out.rglob("*") if path.is_file()
    }
    # Inspect the finished export before publishing it elsewhere.
    import importlib.util

    spec = importlib.util.spec_from_file_location("release_scan", ROOT / "release/check.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    findings = []
    for name in files:
        path = a.out / name
        if scanner.is_text(path):
            text = path.read_text()
            if name == "Dockerfile":
                text = text.replace("/home/user/", "/container-user/")
            findings.extend(scanner.scan(text, name))
        else:
            findings.extend(scanner.scan_binary(path, name))
    write_json(
        a.out / "BUILD_REPORT.json",
        {
            "status": "ready_for_private_runtime_check" if not findings else "scan_failed",
            "bundle_manifest_sha256": file_hash(a.bundle / "manifest.json"),
            "files": files,
            "leak_findings": findings,
            "publication_performed": False,
        },
    )
    print(json.dumps({"files": len(files), "leak_findings": findings, "output": str(a.out)}))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
