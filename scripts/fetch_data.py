#!/usr/bin/env python3
"""Download checksum-pinned inputs or GitHub release artifacts.

    python scripts/fetch_data.py brain                 # simulation inputs
    python scripts/fetch_data.py neural-link           # experimental reader/writer bundle
    python scripts/fetch_data.py results paper-results # recorded research outputs
    python scripts/fetch_data.py vnc                   # optional MaleCNS inputs
    python scripts/fetch_data.py all --check           # local integrity check

Existing files that differ require --force. Archives are verified before installation.
Licenses and citations: THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time, urllib.request
import shutil
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "manifest.json"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, size: int, digest: str) -> None:
    """Stream url to dest.part, hashing as it goes; rename only if the checksum matches."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            h, done, t0 = hashlib.sha256(), 0, time.time()
            req = urllib.request.Request(url, headers={"User-Agent": "flypet-fetch-data"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
                while block := r.read(CHUNK):
                    f.write(block)
                    h.update(block)
                    done += len(block)
                    if sys.stderr.isatty():
                        print(
                            f"\r    {done / 1e6:7.1f} / {size / 1e6:.1f} MB",
                            end="",
                            file=sys.stderr,
                            flush=True,
                        )
            if sys.stderr.isatty():
                print(file=sys.stderr)
            if done != size:
                raise ValueError(f"size {done} differs from the manifest ({size})")
            if h.hexdigest() != digest:
                raise ValueError(f"sha256 {h.hexdigest()} differs from the manifest ({digest})")
            part.replace(dest)
            print(f"    {done / 1e6:.1f} MB in {time.time() - t0:.0f} s, sha256 ok")
            return
        except Exception as e:  # network errors and checksum mismatches are retried the same way
            print(f"    attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(3 * attempt)
    part.unlink(missing_ok=True)
    sys.exit(f"could not fetch {dest.relative_to(ROOT)} from {url}")


def safe_destination(name):
    relative = Path(name)
    destination = ROOT / relative
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not destination.resolve().is_relative_to(ROOT.resolve())
    ):
        raise ValueError(f"Unsafe artifact path: {name}")
    return destination


def extract_archive(archive, entries, force=False):
    """Check all members before replacing any destination; never extract links."""
    expected = {entry["path"]: entry for entry in entries}
    for name in expected:
        safe_destination(name)
    with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
        staging = Path(temporary)
        seen = set()
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream:
                if member.name not in expected or member.name in seen or not member.isfile():
                    raise ValueError(f"Unexpected archive member: {member.name}")
                entry = expected[member.name]
                if member.size != entry["size"]:
                    raise ValueError(f"Archive size mismatch: {member.name}")
                target = staging / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                if sha256_file(target) != entry["sha256"]:
                    raise ValueError(f"Archive hash mismatch: {member.name}")
                seen.add(member.name)
        if seen != set(expected):
            raise ValueError("Archive is missing declared files")
        # A failed integrity check above leaves all existing research files untouched.
        for name, entry in expected.items():
            destination = safe_destination(name)
            if destination.exists() and not force:
                if sha256_file(destination) != entry["sha256"]:
                    raise ValueError(f"Local file differs: {name}; use --force to replace")
        for name in expected:
            destination = safe_destination(name)
            if destination.exists() and not force:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            (staging / name).replace(destination)


def check_arrays(entry: dict) -> str:
    """Compare a built .npz with the array checksums in the manifest (npz files embed timestamps, so the
    file hash is not reproducible; the arrays are)."""
    import numpy as np

    path = ROOT / entry["path"]
    if not path.exists():
        return "missing"
    z = np.load(path, allow_pickle=False)
    for name, want in entry["arrays_sha256"].items():
        if (
            name not in z.files
            or hashlib.sha256(np.ascontiguousarray(z[name]).tobytes()).hexdigest() != want
        ):
            return f"differs in {name}"
    return "ok"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "groups",
        nargs="*",
        default=["brain"],
        help="brain (default), vnc, results, neural-link or all",
    )
    ap.add_argument(
        "--mirror",
        action="store_true",
        help="download from the Hugging Face mirror instead of upstream",
    )
    ap.add_argument("--check", action="store_true", help="verify files on disk, download nothing")
    ap.add_argument(
        "--force", action="store_true", help="replace local files whose checksum differs"
    )
    a = ap.parse_args()

    m = json.loads(MANIFEST.read_text())
    known = sorted({f["group"] for f in m["files"]})
    groups = known if "all" in a.groups else a.groups
    if unknown := set(groups) - set(known):
        ap.error(f"unknown group(s) {sorted(unknown)}; known: {known}")
    mirror = m.get("mirror", {}).get("base_url")
    if a.mirror and not mirror:
        ap.error(
            "No public mirror is configured; omit --mirror to use upstream sources and release archives"
        )

    archived = set()
    for artifact in m.get("archives", []):
        if artifact["group"] not in groups:
            continue
        entries = [f for f in m["files"] if f["group"] == artifact["group"]]
        archived.update(f["path"] for f in entries)
        if a.check:
            continue
        missing = False
        for entry in entries:
            path = safe_destination(entry["path"])
            if not path.exists():
                missing = True
            elif path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
                if not a.force:
                    sys.exit(f"Local file differs: {entry['path']}; use --force to replace")
                missing = True
        if missing:
            cache = ROOT / "data" / ".downloads" / artifact["name"]
            if not cache.exists() or sha256_file(cache) != artifact["sha256"]:
                download(artifact["url"], cache, artifact["size"], artifact["sha256"])
            extract_archive(cache, entries, force=a.force)

    problems = 0
    for f in (f for f in m["files"] if f["group"] in groups):
        dest = safe_destination(f["path"])
        state = "missing"
        if dest.exists():
            state = (
                "ok"
                if dest.stat().st_size == f["size"] and sha256_file(dest) == f["sha256"]
                else "differs"
            )
        if state == "ok":
            print(f"ok       {f['path']}")
            continue
        if a.check or (state == "differs" and not a.force):
            print(
                f"{state:8s} {f['path']}"
                + ("  (--force replaces it)" if state == "differs" and not a.check else "")
            )
            problems += 1
            continue
        if f["path"] in archived:
            raise ValueError(f"Archive did not install {f['path']}")
        url = mirror + f["path"] if a.mirror else f["upstream"]
        print(f"fetch    {f['path']}  ({f['size'] / 1e6:.1f} MB, {f['licence']})\n    <- {url}")
        download(url, dest, f["size"], f["sha256"])

    for d in (d for d in m.get("derived", []) if d["group"] in groups):
        state = check_arrays(d)
        if state == "missing" and a.check:
            print(f"not built yet  {d['path']} (the code builds it on first use)")
            continue
        if state != "ok" and not a.check and not problems:
            print(f"build    {d['path']}  <- {d['build']}")
            (ROOT / d["path"]).unlink(missing_ok=True)
            cmd = (
                [sys.executable, "-c", d["build"]]
                if d["build"].startswith("from ")
                else [sys.executable, *d["build"].split()]
            )
            subprocess.run(cmd, cwd=ROOT, check=True)
            state = check_arrays(d)
        print(
            f"{state:8s} {d['path']} (derived)"
            if state == "ok"
            else f"{state}: {d['path']} (derived)"
        )
        problems += state != "ok"

    if problems:
        sys.exit(f"{problems} file(s) missing or different from data/manifest.json")
    print("all requested files match data/manifest.json")


if __name__ == "__main__":
    main()
