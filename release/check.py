#!/usr/bin/env python3
"""Scan tracked source or an export directory without printing matched secrets."""

from pathlib import Path
import argparse
import json
import re
import subprocess


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


def is_text(path):
    return path.suffix in TEXT or path.name in {"LICENSE", "Makefile", ".gitignore"}


def scan(value, name):
    findings = []
    for label, pattern in PATTERNS.items():
        for match in pattern.finditer(value):
            if label == "personal path" and match.group() == "/home/user/":
                continue  # Unprivileged container account.
            findings.append(f"{name}:{value.count(chr(10), 0, match.start()) + 1}: {label}")
    return findings


def scan_binary(path, name):
    if path.suffix == ".npz":
        import numpy as np

        findings = []
        with np.load(path, allow_pickle=False) as arrays:
            for key in arrays.files:
                value = arrays[key]
                if value.dtype.kind in "US":
                    findings.extend(scan("\n".join(map(str, value.ravel())), f"{name}[{key}]"))
        return findings
    if path.suffix == ".safetensors":
        with path.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if length > path.stat().st_size - 8:
                return [f"{name}: invalid safetensors header"]
            return scan(stream.read(length).decode("utf-8"), name)
    if path.suffix in {".pkl", ".pickle"}:
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
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_symlink():
            findings.append(f"{name}: symlink")
        elif path.name.startswith(".env") or path.suffix in {".key", ".pem"}:
            findings.append(f"{name}: private configuration")
        elif is_text(path):
            findings.extend(scan(path.read_text(), name))
        else:
            findings.extend(scan_binary(path, name))
    print(json.dumps({"files": len(paths), "findings": findings}, indent=2))
    raise SystemExit(bool(findings))


if __name__ == "__main__":
    main()
