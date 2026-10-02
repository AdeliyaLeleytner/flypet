#!/usr/bin/env python3
"""Allowlist a bounded final-evaluation worker package; never publishes artifacts."""

import argparse, json, tarfile, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from flypet.neural_records import file_hash, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bundle", "reader_report", "selection", "sealed", "teacher", "out"):
        p.add_argument("--" + name.replace("_", "-"), dest=name, type=Path, required=True)
    p.add_argument("--train-agent", action="store_true")
    a = p.parse_args()
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
        "response_semantics",
        "neural_codec",
        "latent_writer",
        "latent_inference",
        "latent_sessions",
        "latent_policy",
        "latent_agent_env",
        "latent_agent_rollouts",
    ]
    scripts = [
        "collect_calibration",
        "train_reader_repair",
        "evaluate_sealed_reader",
        "check_live_twins",
        "run_agent_job",
        "prepare_agent_features",
        "train_agent_sft",
        "train_agent_grpo",
    ]
    tests = ["latent_policy", "latent_agent_features", "latent_agent_rollouts", "latent_dialogue"]
    files = (
        [f"flypet/{n}.py" for n in modules]
        + [f"scripts/latent/{n}.py" for n in scripts]
        + [f"tests/test_{n}.py" for n in tests]
    )
    files += [
        "data/connectivity_783.npz",
        "data/flywire_annotations_783.tsv",
        "vendor/Drosophila_brain_model/Completeness_783.csv",
        "vendor/Drosophila_brain_model/LICENSE",
    ] + [
        "data/door/" + name
        for name in ["door_mappings.csv", "door_response_matrix.csv", "odor.csv"]
    ]
    items = {name: Path(name) for name in files}
    items["reader_report.json"] = a.reader_report
    items["selection.json"] = a.selection
    for prefix, folder in [
        ("bundle", a.bundle),
        ("data/sealed_reader", a.sealed),
        ("data/teacher", a.teacher),
    ]:
        for path in folder.rglob("*"):
            if path.is_file() and not path.is_symlink():
                items[prefix + "/" + str(path.relative_to(folder))] = path
    manifest = {name: file_hash(path) for name, path in items.items()}
    write_json(a.out / "source_manifest.json", manifest)
    with tarfile.open(a.out / "source.tar.gz", "w:gz") as archive:
        for name, path in items.items():
            archive.add(path, arcname=name)
        archive.add(a.out / "source_manifest.json", arcname="source_manifest.json")
    command = ["python", "-u", "scripts/latent/run_agent_job.py"]
    if not a.train_agent:
        command.append("--evaluation-only")
    write_json(a.out / "command.json", command)
    write_json(
        a.out / "package_report.json",
        {
            "files": len(items),
            "bytes": (a.out / "source.tar.gz").stat().st_size,
            "train_agent": a.train_agent,
            "bundle_manifest_sha256": file_hash(a.bundle / "manifest.json"),
            "selection_sha256": file_hash(a.selection),
        },
    )
    print(
        json.dumps(
            {
                "files": len(items),
                "package_mb": (a.out / "source.tar.gz").stat().st_size / 1e6,
                "train_agent": a.train_agent,
            }
        )
    )


if __name__ == "__main__":
    main()
