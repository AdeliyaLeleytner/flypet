#!/usr/bin/env python3
"""Stage all research families, sanitize metadata, and build bounded release assets."""

from __future__ import annotations
import argparse
from collections import Counter
import gzip
import hashlib
import io
import ipaddress
import json
import os
from pathlib import Path
import pickletools
import re
import shutil
import struct
import sys
import tarfile
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_data import sha256_file
from release.check import PATTERNS, TEXT

FAMILIES = {
    "neural-history": ("latent",),
    "memory-interface": ("memory_interface",),
    "memory-forecast": ("memory_forecast",),
    "projector-history": ("projector", "state_dataset", "state_pairs", "odor_dataset"),
    "architecture-reader": ("architecture_reader",),
    "question-decoder": ("question_decoder",),
    "flytalk-checkpoints": ("flytalk_rl",),
    "courtship": ("courtship",),
    "physiology-history": ("physiology",),
    "other-experiments": (
        "teach_",
        "flytalk_",
        "rosetta_",
        "grpo_",
        "paper_execution",
        "demo_stimuli_results",
        "odor_modes",
        "rsa_",
        "state_modes",
    ),
}
IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
PRIVATE_FIELD = re.compile(
    r'(?i)(["\']?(?:api_key|access_token|auth_token|password|authorization)["\']?\s*[:=]\s*["\'])([^"\'\n]{16,})(["\'])'
)
CONTROL = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def excluded(name):
    path = Path(name)
    if path.name == ".DS_Store" or "__pycache__" in path.parts or path.suffix in {".pyc", ".lock"}:
        return "generated cache or process lock"
    if path.name.startswith(".env") or path.suffix in {".key", ".pem"}:
        return "private configuration"
    if path.name == "latent-private-state-current-hashes.json":
        return "private user-session state audit"
    if re.search(
        r"(?i)^(?:receipt|billing.*|offers?.*|instances?.*|create_response|create_stdout|create_stderr|ssh.*)\.json$",
        path.name,
    ):
        return "private infrastructure or billing metadata"
    return None


class Sanitizer:
    def __init__(self, source):
        self.source = source.resolve()
        self.changes = Counter()

    def text(self, value):
        old = value
        if re.search(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----", value):
            raise ValueError("Private key material requires excluding the containing artifact")
        value = value.replace(str(self.source) + "/", "").replace(
            str(self.source.parent) + "/", "~/"
        )
        for label, pattern in PATTERNS.items():
            replacement = {
                "credential": "[REDACTED_CREDENTIAL]",
                "personal path": "external/",
                "private host": "redacted-host.invalid",
                "personal email": "[REDACTED_EMAIL]",
            }[label]
            value = pattern.sub(replacement, value)

        def address(match):
            try:
                ip = ipaddress.ip_address(match.group())
            except ValueError:
                return match.group()
            return match.group() if ip.is_loopback or ip.is_unspecified else "redacted-host.invalid"

        value = IP.sub(address, value)
        value = PRIVATE_FIELD.sub(r"\1[REDACTED_CREDENTIAL]\3", value)
        value = CONTROL.sub("", value)
        if old != value:
            self.changes["metadata_strings"] += 1
        return value

    def pickle(self, data):
        """Rewrite only string opcodes, without importing or executing pickle globals."""
        operations = list(pickletools.genops(data))
        replacements = {}
        for i, (op, arg, start) in enumerate(operations):
            if isinstance(arg, str) and op.name in {
                "UNICODE",
                "BINUNICODE",
                "SHORT_BINUNICODE",
                "BINUNICODE8",
            }:
                clean = self.text(arg)
                if clean != arg:
                    encoded = clean.encode("utf-8")
                    end = operations[i + 1][2] if i + 1 < len(operations) else len(data)
                    replacements[start] = (end, b"X" + struct.pack("<I", len(encoded)) + encoded)
        if not replacements:
            return data
        # FRAME byte counts become invalid after a string changes; unframed protocol 4+ is valid.
        for i, (op, _, start) in enumerate(operations):
            if op.name == "FRAME":
                replacements[start] = (operations[i + 1][2], b"")
        chunks, cursor = [], 0
        for start, (end, value) in sorted(replacements.items()):
            chunks.extend((data[cursor:start], value))
            cursor = end
        chunks.append(data[cursor:])
        result = b"".join(chunks)
        list(pickletools.genops(result))
        return result

    def npy(self, data):
        import numpy as np

        stream = io.BytesIO(data)
        version = np.lib.format.read_magic(stream)
        reader = (
            np.lib.format.read_array_header_1_0
            if version == (1, 0)
            else np.lib.format.read_array_header_2_0
        )
        shape, order, dtype = reader(stream)
        if dtype.hasobject:
            position = stream.tell()
            return data[:position] + self.pickle(data[position:])
        if dtype.kind not in "US":
            return data
        array = np.load(io.BytesIO(data), allow_pickle=False)
        if dtype.kind == "S":
            cleaned = [self.text(x.decode()).encode() for x in array.ravel()]
        else:
            cleaned = [self.text(str(x)) for x in array.ravel()]
        if all(x == y for x, y in zip(array.ravel(), cleaned)):
            return data
        output = io.BytesIO()
        np.save(output, np.asarray(cleaned).reshape(shape), allow_pickle=False)
        return output.getvalue()

    def zip(self, source, destination):
        import numpy as np

        changes = {}
        with zipfile.ZipFile(source) as old:
            for entry in old.infolist():
                if Path(entry.filename).is_absolute() or ".." in Path(entry.filename).parts:
                    raise ValueError("Unsafe checkpoint archive member")
                if entry.filename.endswith(".pkl"):
                    original = old.read(entry)
                    content = self.pickle(original)
                elif entry.filename.endswith(".npy"):
                    with old.open(entry) as stream:
                        version = np.lib.format.read_magic(stream)
                        reader = (
                            np.lib.format.read_array_header_1_0
                            if version == (1, 0)
                            else np.lib.format.read_array_header_2_0
                        )
                        shape, order, dtype = reader(stream)
                    if not dtype.hasobject and dtype.kind not in "US":
                        continue
                    original = old.read(entry)
                    content = self.npy(original)
                else:
                    continue
                if content != original:
                    changes[entry.filename] = content
            if not changes:
                return False
            with zipfile.ZipFile(destination, "w") as new:
                for entry in old.infolist():
                    if entry.filename in changes:
                        new.writestr(entry, changes[entry.filename])
                    else:
                        with old.open(entry) as inp, new.open(entry, "w") as out:
                            shutil.copyfileobj(inp, out)
        return True

    def tar(self, source, destination):
        changed = False
        with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
            with (
                tarfile.open(source, "r:gz") as old,
                tarfile.open(destination, "w:gz", compresslevel=1, dereference=True) as new,
            ):
                seen = set()
                for entry in old:
                    name = Path(entry.name)
                    if (
                        name.is_absolute()
                        or ".." in name.parts
                        or not (entry.isfile() or entry.isdir())
                    ):
                        raise ValueError("Unsafe nested archive member")
                    if entry.name in seen:
                        raise ValueError("Duplicate nested archive member")
                    seen.add(entry.name)
                    if entry.isdir():
                        continue
                    if excluded(entry.name):
                        changed = True
                        self.changes["nested_operational_files_excluded"] += 1
                        continue
                    local = Path(tmp) / "input" / name
                    clean = Path(tmp) / "output" / name
                    local.parent.mkdir(parents=True, exist_ok=True)
                    with old.extractfile(entry) as inp, local.open("wb") as out:
                        shutil.copyfileobj(inp, out)
                    was_changed = self.file(local, clean)
                    changed |= was_changed
                    info = new.gettarinfo(str(clean), arcname=entry.name)
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with clean.open("rb") as stream:
                        new.addfile(info, stream)
                    local.unlink()
                    clean.unlink()
        return True

    def file(self, source, destination):
        destination.parent.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix
        if source.name.endswith(".tar.gz"):
            changed = self.tar(source, destination)
        elif suffix in {".npz", ".pt", ".pth"} and zipfile.is_zipfile(source):
            changed = self.zip(source, destination)
        elif suffix in TEXT or source.name in {"DONE", "LICENSE", "Dockerfile"}:
            changed = False
            with (
                source.open(encoding="utf-8") as inp,
                destination.open("w", encoding="utf-8") as out,
            ):
                for line in inp:
                    clean = self.text(line)
                    changed |= line != clean
                    out.write(clean)
        elif suffix == ".npy":
            content = source.read_bytes()
            clean = self.npy(content)
            changed = content != clean
            destination.write_bytes(clean)
        elif suffix in {".pkl", ".pickle"}:
            content = source.read_bytes()
            clean = self.pickle(content)
            changed = content != clean
            destination.write_bytes(clean)
        elif suffix == ".safetensors":
            with source.open("rb") as inp:
                length = int.from_bytes(inp.read(8), "little")
                if length > 100_000_000 or length > source.stat().st_size - 8:
                    raise ValueError("Invalid safetensors header")
                header = inp.read(length)
                clean = self.text(header.decode()).encode()
                changed = header != clean
                with destination.open("wb") as out:
                    clean += b" " * (-len(clean) % 8)
                    out.write(len(clean).to_bytes(8, "little"))
                    out.write(clean)
                    shutil.copyfileobj(inp, out)
        elif suffix == ".joblib":
            data = source.read_bytes()
            try:
                content = zlib.decompress(data)
            except zlib.error:
                content = data
            # Joblib may interleave binary arrays with pickle opcodes; never unpickle it.
            raw = content.decode("latin-1")
            if any(rx.search(raw) for rx in PATTERNS.values()):
                raise ValueError("Private metadata in joblib needs explicit handling")
            changed = False
        elif suffix in {".png", ".pdf", ".gif"}:
            # Figures carry no runtime configuration; inspect their readable metadata.
            raw = source.read_bytes().decode("latin-1")
            if any(rx.search(raw) for rx in PATTERNS.values()):
                raise ValueError("Private metadata in figure needs explicit handling")
            changed = False
        else:
            raise ValueError(f"Uninspected artifact format: {suffix}")
        if not changed:
            destination.unlink(missing_ok=True)
            os.link(source.resolve(), destination)
        return changed


def family(path):
    prefix = path.parts[1]
    if path.parts[0] == "output":
        if prefix.startswith("latent"):
            return "neural-history"
        if prefix.startswith("memory-forecast"):
            return "memory-forecast"
        if prefix.startswith("memory-interface"):
            return "memory-interface"
        return None
    for group, prefixes in FAMILIES.items():
        if prefix.startswith(prefixes):
            return group
    return None


def license_for(path):
    if path.suffix in {".py", ".sh"}:
        return "MIT; upstream code retains its notices"
    if path.suffix == ".safetensors" and "adapter" in path.name:
        return "Apache-2.0"
    return "Component-specific; see THIRD_PARTY_NOTICES.md"


def build_archives(args, base, included, coverage):
    archives = []
    for group in sorted({row["group"] for row in included}):
        rows = [row for row in included if row["group"] == group]
        parts, current, size = [], [], 0
        for row in rows:
            if current and size + row["size"] > args.max_part_bytes:
                parts.append(current)
                current, size = [], 0
            current.append(row)
            size += row["size"]
        if current:
            parts.append(current)
        for index, part in enumerate(parts, 1):
            name = f"flypet-{group}-{index:03d}-{args.tag}.tar.gz"
            destination = args.out / name
            with (
                destination.open("wb") as stream,
                gzip.GzipFile(
                    fileobj=stream, mode="wb", filename="", mtime=0, compresslevel=3
                ) as zipped,
            ):
                with tarfile.open(fileobj=zipped, mode="w", dereference=True) as archive:
                    for row in part:
                        path = args.stage / row["path"]
                        if sha256_file(path) != row["sha256"]:
                            raise ValueError("Staged artifact changed")
                        info = archive.gettarinfo(str(path), arcname=row["path"])
                        info.uid = info.gid = 0
                        info.uname = info.gname = ""
                        info.mtime = 0
                        info.mode = 0o644
                        with path.open("rb") as payload:
                            archive.addfile(info, payload)
                        row["archive"] = name
            if destination.stat().st_size >= 2 * 1024**3:
                raise ValueError("Archive exceeds the GitHub release asset limit")
            archives.append(
                {
                    "group": group,
                    "name": name,
                    "url": f"https://github.com/AdeliyaLeleytner/flypet/releases/download/{args.tag}/{name}",
                    "size": destination.stat().st_size,
                    "sha256": sha256_file(destination),
                    "file_count": len(part),
                }
            )
            print(
                f"packed {name}: {len(part)} files, {destination.stat().st_size / 1e6:.1f} MB",
                flush=True,
            )
    if coverage["selected_files"] != len(included) + len(coverage["already_published"]) + len(
        coverage["excluded"]
    ):
        raise AssertionError("Incomplete research coverage")
    base["files"].extend(included)
    base["archives"].extend(archives)
    base["schema_version"] = 2
    aliases = base.setdefault("group_aliases", {})
    aliases["research"] = sorted(
        set(aliases.get("research", ["results", "paper-results", "neural-link"]))
        | {r["group"] for r in included}
    )
    coverage["release"] = args.tag
    coverage["cautions"] = [
        "Historical failed, partial and superseded runs are included, not endorsed as successful results.",
        "latent_runtime_20260927_v1 was aborted because input refractoriness depended on the action; do not use it for new training.",
        "Former sealed panels are now public historical evaluation data, not unseen tests for future model selection.",
        "Embedded historical receipts retain source hashes; use this manifest for the released sanitized bytes.",
        "Pickle-based research checkpoints retain their original serialization; use weights_only=True when supported.",
    ]
    base["research_coverage"] = coverage
    from scripts.fetch_data import audit_manifest

    audit_manifest(base)
    target = args.out / "manifest.json"
    target.write_text(json.dumps(base, indent=2) + "\n")
    print(f"Complete candidate manifest: {target}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tag", default="v0.2.0")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/manifest.json")
    parser.add_argument("--max-part-bytes", type=int, default=900_000_000)
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    args.stage.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.manifest.read_text())
    existing = {row["path"]: row for row in base["files"]}
    sanitizer = Sanitizer(source)
    included, omitted, covered = [], [], []
    saved_path = args.out / "research-stage.json"
    saved = (
        {r["path"]: r for r in json.loads(saved_path.read_text())["files"]}
        if args.resume and saved_path.exists()
        else {}
    )
    paths = []
    for parent in ("data", "output"):
        for top in sorted((source / parent).iterdir()):
            if not family(top.relative_to(source)):
                continue
            paths.extend(
                sorted(p for p in (top.rglob("*") if top.is_dir() else [top]) if p.is_file())
            )
    for index, path in enumerate(paths):
        relative = path.relative_to(source)
        name = relative.as_posix()
        if reason := excluded(name):
            omitted.append({"path": name, "reason": reason})
            continue
        original_digest = sha256_file(path)
        if name in existing:
            covered.append(
                {"path": name, "source_sha256": original_digest, "group": existing[name]["group"]}
            )
            continue
        if path.is_symlink() and not path.resolve().is_relative_to(source):
            raise ValueError(f"External source symlink: {name}")
        output = args.stage / name
        cached = saved.get(name)
        if (
            cached
            and cached["source_sha256"] == original_digest
            and output.exists()
            and sha256_file(output) == cached["sha256"]
            and (not name.endswith(".tar.gz") or cached.get("nested_owners_normalized"))
        ):
            included.append(cached)
            continue
        output.unlink(missing_ok=True)
        try:
            changed = sanitizer.file(path, output)
        except Exception as exc:
            raise RuntimeError(f"Could not stage {name}: {type(exc).__name__}: {exc}") from exc
        row = {
            "path": name,
            "group": family(relative),
            "size": output.stat().st_size,
            "sha256": sha256_file(output),
            "source_sha256": original_digest,
            "source_size": path.stat().st_size,
            "licence": license_for(relative),
        }
        if name.endswith(".tar.gz"):
            row["nested_owners_normalized"] = True
        if path.is_symlink():
            row["materialized_from"] = path.readlink().as_posix()
        if changed:
            row["metadata_change"] = (
                "Private paths, addresses or credentials redacted; tensor storage/numeric arrays unchanged. Historical receipts retain original hashes."
            )
        included.append(row)
        if index % 200 == 0:
            print(f"staged {index}/{len(paths)} {name}", flush=True)
    coverage = {
        "selected_files": len(paths),
        "new_files": len(included),
        "already_published": covered,
        "excluded": omitted,
        "sanitization": dict(sanitizer.changes),
        "source_roots": {
            parent: sorted(
                p.name for p in (source / parent).iterdir() if family(p.relative_to(source))
            )
            for parent in ("data", "output")
        },
    }
    (args.out / "research-stage.json").write_text(
        json.dumps({"files": included, "coverage": coverage}, indent=2) + "\n"
    )
    if args.stage_only:
        print(
            json.dumps(
                {
                    "new_files": len(included),
                    "already_published": len(covered),
                    "excluded": len(omitted),
                }
            )
        )
        return
    build_archives(args, base, included, coverage)


if __name__ == "__main__":
    main()
