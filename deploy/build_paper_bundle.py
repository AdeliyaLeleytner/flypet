#!/usr/bin/env python3
"""Build an allowlisted anonymous reviewer archive; never copy raw base datasets."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import pickletools
import re
import shutil
import tarfile
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CODE_PATTERNS = (
    "flypet/*.py",
    "flypet/static/*",
    "tests/test_*.py",
    "tests/*.cjs",
    "scripts/run_paper_cases.py",
    "scripts/run_language_cases.py",
    "scripts/run_language_panel.py",
    "scripts/run_nl_panel.py",
    "scripts/gen_odor_data.py",
    "scripts/build_vnc.py",
    "scripts/train_projector.py",
    "scripts/plot_paper_cases.py",
    "scripts/audit_language_panel.py",
    "scripts/analyze_memory_resolution.py",
    "deploy/build_paper_bundle.py",
    "deploy/stage_paper_data.py",
    "deploy/recover_odor_classes.py",
    "deploy/export_hf_garden.py",
    "paper/requirements-*.txt",
    "paper/portability-receipt.json",
    "paper/build_receipt.json",
    "paper/reader_results/*.json",
    "paper/reader_results/seed_*/*.json",
    "paper/reader_results/*.md",
    "paper/analysis/*.md",
    "paper/analysis/*.json",
    "paper/analysis/*.csv",
    "paper/cases/*.json",
    "paper/cases/*.md",
    "paper/main.tex",
    "paper/references.bib",
    "paper/*.sty",
    "paper/*.bst",
    "paper/*.py",
    "paper/sections/*.tex",
    "paper/tables/*.tex",
    "paper/tables/*.json",
    "paper/figures/*.pdf",
    "paper/figures/*.png",
    "paper/figures/*.svg",
    "paper/figures/*.json",
    "paper/figures/*.csv",
    "paper/build/main.pdf",
    "vendor/Drosophila_brain_model/LICENSE",
)
RUN_SUFFIXES = {".json", ".jsonl", ".csv", ".npz", ".npy"}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".csv",
    ".tex",
    ".txt",
    ".bib",
    ".html",
    ".js",
    ".cjs",
    ".svg",
    ".sty",
    ".bst",
}
PRIVATE_PATH = re.compile(r"/(?:Users|home|private/tmp)/[^\s/'\"<>]+")
KEY_MATERIAL = re.compile(
    r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?:sk-ant-|sk-proj-)[A-Za-z0-9_-]{12,}"
)
FORBIDDEN_PARTS = {
    ".git",
    ".venv",
    ".claude",
    ".codex",
    "__pycache__",
    "sessions",
    "receipts",
    "state",
}
READER_TARGET = "data/projector_gpu/brain_projector_clean_v1.pt"
READER_SOURCE = "data/projector_clean_20260924_gpu/seed0_infrastructure_recovery/seed_0/results/brain_projector.pt"

SOURCES = {
    "FlyWire": {
        "license": "CC-BY-NC-4.0",
        "official_license": "https://home.flywire.ai/guidelines",
        "download": "https://codex.flywire.ai/",
        "annotations": "https://github.com/flyconnectome/flywire_annotations",
        "model_files": "https://github.com/philshiu/Drosophila_brain_model",
        "notes": "Use version 783 and the exact recorded annotation export. A newly downloaded annotation export may differ.",
    },
    "DoOR": {
        "license": "CC-BY-SA-4.0",
        "official_license": "https://github.com/ropensci/DoOR.data/blob/master/DESCRIPTION",
        "download": "https://github.com/ropensci/DoOR.data",
        "publication": "https://doi.org/10.1038/srep21841",
        "notes": "The three expected CSV exports are checksum-pinned. The exact historic export commit was not recorded.",
    },
    "MaleCNS": {
        "license": "CC-BY-4.0",
        "official_license": "https://male-cns.janelia.org/download/",
        "download": "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/",
        "notes": "scripts/build_vnc.py derives the bridge from the three named v1.0 files; derived NPZ/parquet hashes are recorded.",
    },
    "Shiu_code": {
        "license": "MIT",
        "download": "https://github.com/philshiu/Drosophila_brain_model",
        "notes": "MIT notice is retained for adapted equations and model code; it does not replace dataset-specific terms.",
    },
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_for(path):
    if path.startswith("data/door/"):
        return "DoOR"
    if "vnc_" in path or "malecns" in path:
        return "MaleCNS"
    return "FlyWire"


def inspect_file(path, relative):
    if path.is_symlink() or any(p in FORBIDDEN_PARTS for p in relative.parts):
        raise ValueError(f"Forbidden/symlink path in bundle selection: {relative}")
    if path.suffix in TEXT_SUFFIXES or path.name == "LICENSE":
        text = path.read_text()
        if PRIVATE_PATH.search(text) or KEY_MATERIAL.search(text):
            raise ValueError(f"Potential private path/key in selected file: {relative}")
        username = Path.home().name
        if len(username) >= 6 and username.lower() in text.lower():
            raise ValueError(f"Local account name in selected file: {relative}")


def export_json_paths(
    value, root, pointer="", *, aliases=None, include_reader_checkpoint=False, _root_prefix=None
):
    """Normalize known metadata paths only; never alter source or raw arrays.

    The exported ledger records the original file hash and changed JSON pointers.
    Unknown private paths still fail the ordinary privacy check after export.
    """
    if not isinstance(value, (dict, list, str)):
        return value, []
    changes = []
    root_prefix = _root_prefix if _root_prefix is not None else str(Path(root).resolve()) + "/"
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            child, changed = export_json_paths(
                item,
                root,
                pointer + "/" + key,
                aliases=aliases,
                include_reader_checkpoint=include_reader_checkpoint,
                _root_prefix=root_prefix,
            )
            out[key] = child
            changes.extend(changed)
        return out, changes
    if isinstance(value, list):
        out = []
        for i, item in enumerate(value):
            child, changed = export_json_paths(
                item,
                root,
                pointer + "/" + str(i),
                aliases=aliases,
                include_reader_checkpoint=include_reader_checkpoint,
                _root_prefix=root_prefix,
            )
            out.append(child)
            changes.extend(changed)
        return out, changes
    if isinstance(value, str):
        if aliases and value in aliases:
            return aliases[value], [pointer]
        if pointer.endswith("/claude_cli") and Path(value).name == "claude":
            if value != "claude":
                return "claude", [pointer]
        # Model weights are not redistributed; retain their checksum separately.
        if value.startswith(root_prefix) and value.endswith(".pt"):
            return (
                READER_TARGET
                if include_reader_checkpoint
                else "external-models/brain_projector.pt",
                [pointer],
            )
        if value.startswith(root_prefix):
            return value[len(root_prefix) :], [pointer]
        if value in ("/workspace/fly/data", "/workspace/fly/results"):
            return ("external-training-data" if value.endswith("data") else "training-output"), [
                pointer
            ]
    return value, []


def prepare_reader_exports(root):
    """Export three verified result sets; never copy training datasets or receipts."""
    root = Path(root).resolve()
    base = root / "data/projector_clean_20260924_gpu"
    directories = {
        0: base / "seed0_infrastructure_recovery/seed_0/results",
        1: base / "seed_1/results",
        2: base / "seed_2/results",
    }
    names = [
        "report.json",
        "predictions.json",
        "manifest.json",
        "source_manifest.json",
        "model_snapshot.json",
        "preparation_summary.json",
    ]
    exports = [(base / "verified_summary.json", Path("paper/reader_results/verified_summary.json"))]
    for seed, directory in directories.items():
        original_hashes = json.loads((directory / "SHA256.json").read_text())
        for name in names:
            source = directory / name
            if digest(source) != original_hashes[name]:
                raise ValueError(f"Reader result checksum mismatch: seed {seed}/{name}")
            exports.append((source, Path(f"paper/reader_results/seed_{seed}/{name}")))
        exports.append(
            (
                directory / "SHA256.json",
                Path(f"paper/reader_results/seed_{seed}/original_SHA256.json"),
            )
        )
    aliases = {}
    for source, target in exports:
        aliases[str(source)] = str(target)
        aliases[str(source.relative_to(root))] = str(target)
    ledger = {}
    for source, relative in exports:
        value = json.loads(source.read_text())
        exported, changed = export_json_paths(value, root, aliases=aliases)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if changed:
            target.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n")
        else:
            shutil.copyfile(source, target)
        inspect_file(target, relative)
        ledger[str(relative)] = {
            "source_sha256": digest(source),
            "export_sha256": digest(target),
            "changed_json_pointers": changed,
        }
    manifest = {
        "schema_version": 1,
        "seeds": [0, 1, 2],
        "files": ledger,
        "reports_and_predictions_included": [0, 1, 2],
        "checkpoint_copy_policy": "Only seed 0, and only with --include-reader-checkpoint",
        "original_SHA256_note": "Per-seed original_SHA256.json describes the original training directory, including external datasets and unshipped weights. Export SHA values are in this ledger.",
        "training_data_included": False,
        "billing_and_provider_receipts_included": False,
        "scope": "Historical cleaned reader evaluation on legacy state data. Not new DoOR or memory generalization.",
    }
    target = root / "paper/reader_results/export_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n")
    return {"status": "exported", "files": len(ledger), "seeds": [0, 1, 2], "checkpoints_copied": 0}


def reader_checkpoint(root):
    """Resolve only the already active, checksum-pinned seed-0 checkpoint."""
    root = Path(root).resolve()
    active = root / READER_TARGET
    source = active.resolve(strict=True)
    expected_source = (root / READER_SOURCE).resolve(strict=True)
    if not source.is_relative_to(root) or source != expected_source:
        raise ValueError(
            "Active reader does not resolve to the known seed-0 checkpoint inside this project"
        )
    expected_hash = json.loads((source.parent / "SHA256.json").read_text())["brain_projector.pt"]
    actual = digest(source)
    if actual != expected_hash:
        raise ValueError("Selected reader checkpoint checksum mismatch")
    # Inspect pickle string constants without unpickling or executing checkpoint code.
    n_strings = 0
    with zipfile.ZipFile(source) as archive:
        data_pickle = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        for op, value, _ in pickletools.genops(archive.read(data_pickle)):
            if op.name in {"BINUNICODE", "SHORT_BINUNICODE", "UNICODE"}:
                n_strings += 1
                if (
                    PRIVATE_PATH.search(value)
                    or KEY_MATERIAL.search(value)
                    or Path.home().name in value
                ):
                    raise ValueError("Private marker found in checkpoint metadata")
    return source, {
        "included": True,
        "seed": 0,
        "path": READER_TARGET,
        "sha256": actual,
        "bytes": source.stat().st_size,
        "serialization_string_constants_checked": n_strings,
        "other_seed_checkpoints_included": False,
        "base_qwen_weights_included": False,
    }


def selected_files(root, runs):
    files = set()
    for pattern in CODE_PATTERNS:
        files.update(p for p in root.glob(pattern) if p.is_file())
    required = [
        "flypet/experiments.py",
        "scripts/run_paper_cases.py",
        "paper/cases/protocol.json",
    ]
    for path in required:
        if not (root / path).is_file():
            raise FileNotFoundError(path)
    external = {}
    run_status = {}
    for run in runs:
        run_dir = root / "paper" / "results" / run
        if run_dir.parent != root / "paper" / "results" or not re.fullmatch(r"[A-Za-z0-9_-]+", run):
            raise ValueError("Run names must be simple directory names")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        if manifest.get("status") not in {"complete", "complete_with_recorded_failures"}:
            raise ValueError(f"Refusing incomplete run {run}: {manifest.get('status')}")
        run_status[run] = {
            "status": manifest["status"],
            "actual_records": manifest.get("actual_records", manifest.get("n_cases")),
            "protocol_sha256": manifest.get("protocol_sha256", manifest.get("panel_sha256")),
            "actual_llm_calls": manifest.get("actual_llm_calls"),
        }
        for relative, item in manifest.get("files", {}).items():
            if relative.startswith(("data/", "vendor/")):
                entry = dict(item, source=source_for(relative))
                if relative in external and external[relative] != entry:
                    raise ValueError(f"Conflicting dataset fingerprints: {relative}")
                external[relative] = entry
            elif relative.startswith(("flypet/", "scripts/", "paper/cases/")):
                local = root / relative
                if not local.exists() or digest(local) != item["sha256"]:
                    raise ValueError(f"Run source snapshot differs from current file: {relative}")
        # Explicit experiment artifacts only; arbitrary text logs and state dirs are excluded.
        files.update(
            p
            for p in run_dir.rglob("*")
            if p.is_file()
            and p.suffix in RUN_SUFFIXES
            and not any(part in FORBIDDEN_PARTS for part in p.relative_to(run_dir).parts)
        )
    return sorted(files), external, run_status


def verify_tree(root):
    root = Path(root)
    manifest = json.loads((root / "bundle-manifest.json").read_text())
    errors = []
    for relative, item in manifest["files"].items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            errors.append(relative)
            continue
        path = root / relative
        if (
            not path.is_file()
            or path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or digest(path) != item["sha256"]
        ):
            errors.append(relative)
    return {
        "ok": not errors,
        "checked_files": len(manifest["files"]),
        "errors": errors,
        "base_datasets_in_archive": False,
    }


def create_bundle(root, output, runs, extract_to=None, include_reader_checkpoint=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Choose a new bundle filename; existing archives are preserved")
    output.parent.mkdir(parents=True, exist_ok=True)
    files, external, run_status = selected_files(root, runs)
    reader_export_manifest = root / "paper/reader_results/export_manifest.json"
    if include_reader_checkpoint and not reader_export_manifest.is_file():
        raise FileNotFoundError(
            "Prepare all three reader result exports with --prepare-reader-results before including the checkpoint"
        )
    if reader_export_manifest.is_file():
        for relative, item in json.loads(reader_export_manifest.read_text())["files"].items():
            if not (root / relative).is_file() or digest(root / relative) != item["export_sha256"]:
                raise ValueError(f"Reader export fingerprint mismatch: {relative}")
    checkpoint_source, checkpoint_info = (
        reader_checkpoint(root)
        if include_reader_checkpoint
        else (
            None,
            {
                "included": False,
                "other_seed_checkpoints_included": False,
                "base_qwen_weights_included": False,
            },
        )
    )
    if include_reader_checkpoint:
        for run in runs:
            run_manifest = json.loads((root / "paper/results" / run / "manifest.json").read_text())
            used_hash = run_manifest.get("reader", {}).get("checkpoint_sha256")
            if used_hash and used_hash != checkpoint_info["sha256"]:
                raise ValueError(f"Included reader checkpoint differs from one recorded in {run}")
    with tempfile.TemporaryDirectory(prefix="flypet-review-") as tmp:
        staging = Path(tmp) / "flypet-artifact"
        staging.mkdir()
        copied, transformations = {}, {}
        for source in files:
            relative = source.relative_to(root)
            if not source.resolve().is_relative_to(root):
                raise ValueError(f"Selected file resolves outside project: {relative}")
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            changed = []
            if source.suffix in {".json", ".jsonl"} and str(relative).startswith(
                ("paper/results/", "paper/analysis/", "paper/reader_results/")
            ):
                # Only experiment metadata needs normalization. Preserve byte identity
                # of already portable metadata and every source/raw array file.
                if source.suffix == ".json":
                    exported, changed = export_json_paths(
                        json.loads(source.read_text()),
                        root,
                        include_reader_checkpoint=include_reader_checkpoint,
                    )
                    if changed:
                        target.write_text(json.dumps(exported, ensure_ascii=False, indent=2) + "\n")
                else:
                    lines = []
                    for i, line in enumerate(source.read_text().splitlines()):
                        if line:
                            exported, fields = export_json_paths(
                                json.loads(line),
                                root,
                                f"row{i}",
                                include_reader_checkpoint=include_reader_checkpoint,
                            )
                            changed.extend(fields)
                            lines.append(json.dumps(exported, ensure_ascii=False))
                    if changed:
                        target.write_text("\n".join(lines) + "\n")
            if not changed:
                inspect_file(source, relative)
                shutil.copyfile(source, target)
            else:
                transformations[str(relative)] = {
                    "source_sha256": digest(source),
                    "changed_json_pointers": changed,
                    "method": "project paths made relative; CLI reduced to executable name; external reader checkpoint gets placeholder",
                }
            inspect_file(target, relative)
            # PDF metadata must be checked by the manuscript build, not rewritten here.
            copied[str(relative)] = {"sha256": digest(target), "bytes": target.stat().st_size}
        if checkpoint_source is not None:
            target = staging / READER_TARGET
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint_source, target)
            if digest(target) != checkpoint_info["sha256"]:
                raise ValueError("Checkpoint copy checksum mismatch")
            copied[READER_TARGET] = {
                "sha256": checkpoint_info["sha256"],
                "bytes": checkpoint_info["bytes"],
            }
        data_manifest = {
            "base_datasets_included": False,
            "sources": SOURCES,
            "files": external,
            "reader_checkpoint": checkpoint_info,
            "qwen_base_model": {
                "included": False,
                "model": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "source": "https://huggingface.co/Qwen/Qwen3-0.6B",
            },
            "reader_training_data": {
                "included": False,
                "fingerprints": "paper/reader_results/seed_0/source_manifest.json",
            },
            "availability": "Exact current local assets can be staged and checksum-verified. Independent reconstruction from upstream has not yet been verified.",
        }
        (staging / "external-data.json").write_text(json.dumps(data_manifest, indent=2) + "\n")
        checkpoint_sentence = (
            "The selected seed-0 clean reader checkpoint is included."
            if include_reader_checkpoint
            else "Reader checkpoints are external in this archive."
        )
        (staging / "README.md").write_text(
            "# Anonymous reviewer artifact\n\nUse scripts/run_paper_cases.py --help to inspect or replay the selected cases. See external-data.json for required inputs.\n\nThis archive contains code and generated experiment outputs. "
            + checkpoint_sentence
            + " Base connectome/DoOR inputs and Qwen model/tokenizer files remain external; see `external-data.json`.\n"
        )
        for relative in ("external-data.json", "README.md"):
            p = staging / relative
            copied[relative] = {"sha256": digest(p), "bytes": p.stat().st_size}
        manifest = {
            "schema_version": 1,
            "archive_type": "anonymous_reviewer_artifact",
            "files": copied,
            "runs": run_status,
            "base_datasets_included": False,
            "reader_checkpoint": checkpoint_info,
            "export_transformations": transformations,
            "privacy": {
                "selection": "allowlist",
                "local_account_and_home_paths_checked": True,
                "private_state_sessions_receipts_venv_excluded": True,
                "tar_owner_names_removed": True,
            },
            "validation": "Archive checksum/extraction validation does not constitute a clean dependency install or a simulation replay.",
        }
        (staging / "bundle-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        checksum_lines = [f"{value['sha256']}  {key}" for key, value in sorted(copied.items())]
        checksum_lines.append(f"{digest(staging / 'bundle-manifest.json')}  bundle-manifest.json")
        (staging / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
        with output.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
                with tarfile.open(fileobj=gz, mode="w") as tf:
                    for path in sorted(staging.rglob("*")):
                        if not path.is_file():
                            continue
                        info = tarfile.TarInfo(str(path.relative_to(staging.parent)))
                        info.size, info.mode, info.mtime = path.stat().st_size, 0o644, 0
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        with path.open("rb") as f:
                            tf.addfile(info, f)
    report = {
        "archive": output.name,
        "bytes": output.stat().st_size,
        "sha256": digest(output),
        "files": len(copied) + 2,
        "runs": run_status,
        "external_data_files": len(external),
    }
    if extract_to:
        target = Path(extract_to).resolve()
        if target.exists() and any(target.iterdir()):
            raise FileExistsError("Extraction test directory must be empty")
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output) as tf:
            tf.extractall(target, filter="data")
        report["fresh_extraction"] = verify_tree(target / "flypet-artifact")
    output.with_suffix(output.suffix + ".sha256").write_text(f"{report['sha256']}  {output.name}\n")
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--runs", nargs="+", default=["smoke-v1"])
    ap.add_argument("--extract-to", type=Path)
    ap.add_argument("--verify-tree", type=Path)
    ap.add_argument(
        "--include-reader-checkpoint",
        action="store_true",
        help="Copy only the active verified clean seed-0 reader weights",
    )
    ap.add_argument(
        "--prepare-reader-results",
        action="store_true",
        help="Export verified clean reader reports/predictions/manifests without building archive",
    )
    args = ap.parse_args(argv)
    if args.prepare_reader_results:
        report = prepare_reader_exports(args.root)
    elif args.verify_tree:
        report = verify_tree(args.verify_tree)
    else:
        if args.output is None:
            ap.error("--output is required unless --verify-tree is used")
        report = create_bundle(
            args.root, args.output, args.runs, args.extract_to, args.include_reader_checkpoint
        )
    print(json.dumps(report, indent=2))
    return 0 if report.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
