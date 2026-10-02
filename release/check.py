#!/usr/bin/env python3
"""Scan tracked source or an export directory without printing matched secrets."""

from pathlib import Path
import argparse
import io
import pickletools
import shutil
import tarfile
import tempfile
import zipfile
import zlib
import json
import re
import subprocess
import sys


TEXT = {
    ".py",
    ".md",
    ".txt",
    ".json",
    ".jsonl",
    ".sh",
    ".html",
    ".js",
    ".cjs",
    ".css",
    ".csv",
    ".tsv",
    ".log",
    ".tex",
    ".bib",
    ".svg",
    ".sty",
    ".bst",
    ".service",
    ".Caddyfile",
    ".toml",
    ".yaml",
    ".yml",
    "",
}
PATTERNS = {
    "credential": re.compile(
        r"sk-(?:ant|proj)-[A-Za-z0-9_-]{12,}|\bhf_[A-Za-z0-9]{30,}|\bgh[opsu]_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"
    ),
    "personal path": re.compile(r"/(?:Users|home)/[A-Za-z][^/\s'\"]+/"),
    "private host": re.compile(r"[\w.-]+\.ts\.net"),
    "personal email": re.compile(
        r"[\w.+-]+@(?:gmail|yandex|icloud|outlook|hotmail|proton|mail)\.[a-z]+"
    ),
}
HINTS = {
    "credential": (
        "sk-ant-",
        "sk-proj-",
        "hf_",
        "gho_",
        "ghp_",
        "ghs_",
        "ghu_",
        "AKIA",
        "PRIVATE KEY",
    ),
    "personal path": ("/Users/", "/home/"),
    "private host": (".ts.net",),
    "personal email": ("@",),
}


def is_text(path):
    return path.suffix in TEXT or path.name in {"LICENSE", "Makefile", ".gitignore"}


def scan(value, name):
    findings = []
    for label, pattern in PATTERNS.items():
        if not any(hint in value for hint in HINTS[label]):
            continue
        for match in pattern.finditer(value):
            if label == "personal path" and match.group() == "/home/user/":
                continue  # Unprivileged container account.
            findings.append(f"{name}:{value.count(chr(10), 0, match.start()) + 1}: {label}")
    return findings


def scan_pickle(data, name):
    findings = []
    for operation, value, offset in pickletools.genops(data):
        if isinstance(value, str):
            findings.extend(scan(value, f"{name}[pickle:{offset}]"))
        elif isinstance(value, bytes):
            findings.extend(scan(value.decode("latin-1"), f"{name}[pickle:{offset}]"))
    return findings


def scan_npy(stream, name):
    import numpy as np

    version = np.lib.format.read_magic(stream)
    reader = (
        np.lib.format.read_array_header_1_0
        if version == (1, 0)
        else np.lib.format.read_array_header_2_0
    )
    shape, order, dtype = reader(stream)
    if dtype.hasobject:
        return scan_pickle(stream.read(), name)
    if dtype.kind not in "US":
        return []
    value = np.frombuffer(stream.read(), dtype=dtype)
    return scan("\n".join(map(str, value)), name)


def scan_binary(path, name):
    if path.name.endswith(".tar.gz"):
        findings = []
        with tempfile.TemporaryDirectory() as temporary, tarfile.open(path, "r:gz") as archive:
            for member in archive:
                relative = Path(member.name)
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or not (member.isfile() or member.isdir())
                ):
                    findings.append(f"{name}: unsafe nested member")
                    continue
                if not member.isfile():
                    continue
                target = Path(temporary) / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as inp, target.open("wb") as out:
                    shutil.copyfileobj(inp, out)
                nested_name = name + ":" + member.name
                if is_text(target):
                    with target.open() as stream:
                        for line_number, line in enumerate(stream, 1):
                            findings.extend(scan(line, f"{nested_name}:{line_number}"))
                else:
                    findings.extend(scan_binary(target, nested_name))
                target.unlink()
        return findings
    if path.suffix in {".npz", ".pt", ".pth"} and zipfile.is_zipfile(path):
        findings = []
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith(".pkl"):
                    findings.extend(scan_pickle(archive.read(member), name + ":" + member))
                elif member.endswith(".npy"):
                    with archive.open(member) as stream:
                        findings.extend(scan_npy(stream, name + ":" + member))
        return findings
    if path.suffix == ".npy":
        with path.open("rb") as stream:
            return scan_npy(stream, name)
    if path.suffix == ".safetensors":
        with path.open("rb") as stream:
            n = int.from_bytes(stream.read(8), "little")
            if n > path.stat().st_size - 8:
                return [f"{name}: invalid safetensors header"]
            return scan(stream.read(n).decode(), name)
    if path.suffix in {".pkl", ".pickle"}:
        return scan_pickle(path.read_bytes(), name)
    if path.suffix == ".joblib":
        data = path.read_bytes()
        try:
            data = zlib.decompress(data)
        except zlib.error:
            pass
        return scan(data.decode("latin-1"), name)
    if path.suffix in {".pdf", ".png", ".gif"}:
        return scan(path.read_bytes().decode("latin-1"), name)
    return []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", type=Path)
    args = parser.parse_args()
    root = args.directory.resolve() if args.directory else Path(__file__).resolve().parents[1]
    if args.directory:
        paths = sorted(p for p in root.rglob("*") if p.is_file())
    else:
        names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
        paths = [root / name for name in names if name]
    findings = []
    for number, path in enumerate(paths, 1):
        if number % 1000 == 0:
            print(f"scanned {number}/{len(paths)} files", file=sys.stderr, flush=True)
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(f"{name}: symlink")
        elif path.name.startswith(".env") or path.suffix in {".key", ".pem"}:
            findings.append(f"{name}: private configuration")
        elif is_text(path):
            with path.open() as stream:
                for number, line in enumerate(stream, 1):
                    findings.extend(scan(line, f"{name}:{number}"))
        else:
            findings.extend(scan_binary(path, name))
    print(json.dumps({"files": len(paths), "findings": findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
