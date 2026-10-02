#!/usr/bin/env python3
"""Strict field extraction audit of the small, preselected voice sample."""

import argparse, json, re
from pathlib import Path
from collections import Counter
import numpy as np

ODORS = [
    "ethyl acetate",
    "methyl acetate",
    "1-hexanol",
    "2-heptanone",
    "hexanal",
    "acetic acid",
    "linalool",
    "geosmin",
]


def field(kind, text):
    text = text.lower()
    if kind == "summary":
        if "no mbon output spikes" in text or "no approach or avoidance output" in text:
            return "silent"
        if "balanced" in text:
            return "balanced"
        match = re.search(r"(?:toward|towards) (approach|avoidance)", text)
        return match.group(1) if match else None
    if kind == "odor":
        found = tuple(name for name in ODORS if name in text)
        return found or None
    if kind == "dynamics":
        match = re.search(r"time bin (\d)", text)
        return int(match.group(1)) if match else ("silent" if "no spikes" in text else None)
    if kind == "population":
        match = re.search(r"the ([a-z0-9_ -]+?) population has the highest", text)
        return (
            match.group(1)
            if match
            else ("silent" if "no observed neuron population" in text else None)
        )
    if kind in ("change", "change_qualitative"):
        if "no resolved change" in text or "no change in the neural balance" in text:
            return "unchanged"
        if "shifted toward approach" in text:
            return "toward approach"
        if "shifted toward avoidance" in text:
            return "toward avoidance"
        match = re.search(r"shifting (toward approach|toward avoidance|not at all)", text)
        return (
            ("unchanged" if match.group(1) == "not at all" else match.group(1)) if match else None
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    data = json.loads(a.predictions.read_text())
    result = {
        "scope": "strict template-field audit; 12 preselected examples per question type, not a broad free-dialogue benchmark",
        "splits": {},
    }
    for split in ("validation", "development_test"):
        groups = {}
        rows = data[split + "/voice"]
        for kind in ("summary", "dynamics", "odor", "population", "change", "change_qualitative"):
            chosen = [r for r in rows if r["kind"] == kind]
            targets = [field(kind, r["target"]) for r in chosen]
            predictions = [field(kind, r["text"]) for r in chosen]
            groups[kind] = {
                "n": len(chosen),
                "recognized": sum(p is not None for p in predictions),
                "correct": sum(p is not None and p == t for p, t in zip(predictions, targets)),
                "target_distribution": dict(Counter(map(str, targets))),
                "examples": [
                    {
                        "id": r["id"],
                        "text": r["text"],
                        "target": r["target"],
                        "recognized_prediction": pred,
                        "recognized_target": target,
                    }
                    for r, pred, target in zip(chosen, predictions, targets)
                ],
            }
        result["splits"][split] = groups
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                split: {
                    kind: {
                        k: v for k, v in row.items() if k not in ("examples", "target_distribution")
                    }
                    for kind, row in groups.items()
                }
                for split, groups in result["splits"].items()
            }
        )
    )


if __name__ == "__main__":
    main()
