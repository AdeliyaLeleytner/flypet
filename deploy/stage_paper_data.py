#!/usr/bin/env python3
"""Stage only checksum-pinned external datasets into an extracted local artifact.

No network, symlinks, private memory, cache, logs, credentials, or model weights.
This local portability helper is not an independent upstream-download test.
"""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def stage(source_root, bundle_root):
    source_root, bundle_root = Path(source_root).resolve(), Path(bundle_root).resolve()
    manifest = json.loads((bundle_root / "external-data.json").read_text())
    entries = manifest["files"]
    checked = []
    for relative, entry in entries.items():
        path = Path(relative)
        if (
            path.is_absolute()
            or ".." in path.parts
            or not str(path).startswith(("data/", "vendor/"))
        ):
            raise ValueError(f"Unsafe manifest path: {relative}")
        source = source_root / path
        target = bundle_root / path
        if (
            source.is_symlink()
            or not source.is_file()
            or not source.resolve().is_relative_to(source_root)
            or digest(source) != entry["sha256"]
        ):
            raise ValueError(f"Source missing or fingerprint mismatch: {relative}")
        if not target.parent.resolve().is_relative_to(bundle_root):
            raise ValueError(f"Target parent resolves outside package: {relative}")
        if target.exists():
            if target.is_symlink() or digest(target) != entry["sha256"]:
                raise FileExistsError(f"Refusing to replace different existing asset: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        if digest(target) != entry["sha256"]:
            raise ValueError(f"Copy verification failed: {relative}")
        checked.append(relative)
    report = {
        "mode": "local_staging_only",
        "copied_or_verified": checked,
        "n_assets": len(checked),
        "network_download_test": False,
        "source_directory_retained_in_report": False,
    }
    (bundle_root / "local-staging-receipt.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-directory", type=Path, required=True)
    ap.add_argument("--bundle-root", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(stage(args.from_directory, args.bundle_root), indent=2))


if __name__ == "__main__":
    main()
