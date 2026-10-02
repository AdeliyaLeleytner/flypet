#!/usr/bin/env python3
"""Recover the historic 52-compound taxonomy from labels, without reading rates."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def recover(dataset, output):
    mapping, sources, n_labels = {}, [], 0
    for path in sorted(Path(dataset).glob("shard_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            for value in data["labels"]:
                row = json.loads(str(value))
                n_labels += 1
                compound, chem_class = row.get("compound"), row.get("chem_class")
                if compound:
                    if not chem_class or (compound in mapping and mapping[compound] != chem_class):
                        raise ValueError(f"Missing/conflicting original class: {compound}")
                    mapping[compound] = chem_class
        sources.append({"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if len(mapping) != 52:
        raise ValueError(f"Expected 52 historic compounds, found {len(mapping)}")
    classes = {
        kind: sorted(c for c, value in mapping.items() if value == kind)
        for kind in sorted(set(mapping.values()))
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(classes, ensure_ascii=False, indent=2) + "\n")
    receipt = {
        "n_compounds": len(mapping),
        "n_source_labels": n_labels,
        "sources": sources,
        "procedure": "Read labels only; preserve historic class assignment exactly; no reassignment or new chemistry claims.",
    }
    output.with_suffix(".provenance.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n"
    )
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(recover(args.dataset, args.output), indent=2))


if __name__ == "__main__":
    main()
