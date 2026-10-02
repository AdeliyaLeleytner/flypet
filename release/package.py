#!/usr/bin/env python3
"""Rebuild a release archive from verified local files; never upload anything."""

import argparse
import gzip
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.fetch_data import safe_destination, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("group")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    entries = [f for f in manifest["files"] if f["group"] == args.group]
    if not entries:
        parser.error("Unknown or empty artifact group")
    for entry in entries:
        path = safe_destination(entry["path"])
        if (
            path.is_symlink()
            or path.stat().st_size != entry["size"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError(f"Local artifact differs: {entry['path']}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        with gzip.GzipFile(fileobj=output, mode="wb", filename="", mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for entry in entries:
                    path = safe_destination(entry["path"])
                    info = archive.gettarinfo(str(path), arcname=entry["path"])
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    info.mtime = 0
                    info.mode = 0o644
                    with path.open("rb") as stream:
                        archive.addfile(info, stream)
    print(
        json.dumps(
            {
                "name": args.out.name,
                "group": args.group,
                "size": args.out.stat().st_size,
                "sha256": sha256_file(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
