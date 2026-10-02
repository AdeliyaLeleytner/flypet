"""Reproduce the legacy row split without loading a model or changing source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def audit(root: Path) -> dict:
    rows, sources = [], []
    for folder in ("state_dataset", "state_dataset_b", "state_dataset_c", "state_pairs"):
        for path in sorted((root / folder).glob("shard_*.npz")):
            sources.append(
                {
                    "path": str(path.relative_to(root)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
            with np.load(path, allow_pickle=False) as z:
                for j in range(int(z["n"])):
                    a, b = map(int, z["offsets"][j : j + 2])
                    label = str(z["labels"][j])
                    digest = hashlib.sha256(
                        z["idx"][a:b].tobytes() + z["rate"][a:b].tobytes() + label.encode()
                    ).hexdigest()
                    keys = {s["key"] for s in json.loads(label)["stimuli"]}
                    rows.append((digest, {"sugar", "bitter"} <= keys))
    n = len(rows)
    if not n:
        raise ValueError(f"No legacy samples found in {root}")
    heldout = {i for i, (_, sb) in enumerate(rows) if sb}
    test = set(np.random.default_rng(0).permutation(n)[: int(0.15 * n)]) - heldout
    train = set(range(n)) - test - heldout
    train_hashes = {rows[i][0] for i in train}
    overlap = lambda ids: sum(rows[i][0] in train_hashes for i in ids)
    return {
        "audit_version": 1,
        "scope": "Legacy 6234-row split, not a new model evaluation",
        "identity": "SHA256 of stored sparse idx bytes + rate bytes + label UTF-8",
        "legacy_split": "NumPy default_rng(0), first floor(0.15*n) rows test, sugar+bitter held out",
        "rows": n,
        "unique_rows": len({h for h, _ in rows}),
        "train": len(train),
        "test": len(test),
        "interaction": len(heldout),
        "test_rows_with_train_duplicate": overlap(test),
        "generation_rows": min(100, len(test)),
        "generation_rows_with_train_duplicate": overlap(sorted(test)[:100]),
        "interaction_rows_with_train_duplicate": overlap(heldout),
        "interaction_unique_rows": len({rows[i][0] for i in heldout}),
        "sources": sources,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = audit(args.data)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2))
