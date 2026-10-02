"""Balanced, paired event histories; no whole-history encoder or decoder cache."""

import hashlib, json, random
from pathlib import Path

OBJECTS = ("key", "coin", "ring")
LOCATIONS = ("red", "blue", "green", "yellow")


def generate(seed=20260924):
    rng = random.Random(seed)
    splits = {s: [] for s in ("train", "validation", "test")}
    for target in OBJECTS:
        for ai, a in enumerate(LOCATIONS):
            for b in LOCATIONS[ai + 1 :]:
                groups = []
                for d in LOCATIONS:
                    block = [
                        (other, c, d) for other in OBJECTS if other != target for c in LOCATIONS
                    ]
                    rng.shuffle(block)
                    groups.extend(
                        (("train" if i < 6 else "validation" if i == 6 else "test"), item)
                        for i, item in enumerate(block)
                    )
                for split, (other, c, d) in groups:
                    identity = (target, other, a, b, c, d)
                    group = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:16]
                    for assignment, (first, last) in enumerate(((a, b), (b, a))):
                        events = [(target, first), (other, c), (target, last), (other, d)]
                        history = [f"The {obj} moved to the {loc} box." for obj, loc in events]
                        for query in (target, other):
                            answer = last if query == target else d
                            splits[split].append(
                                {
                                    "id": f"{group}-a{assignment}-{query}",
                                    "group": group,
                                    "assignment": assignment,
                                    "history": history,
                                    "events": events,
                                    "query_object": query,
                                    "query": f"Where is the {query} now?",
                                    "answer": answer,
                                    "query_is_last_mentioned": query == other,
                                    "split": split,
                                }
                            )
    for rows in splits.values():
        rng.shuffle(rows)
    return splits


def question_text(row):
    # This is the only natural-language information passed to the decoder.
    return (
        "Track an object in colored boxes. Answer with exactly one word: "
        "red, blue, green, or yellow. " + row["query"]
    )


def write_dataset(directory):
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    data = generate()
    files = {}
    for split, rows in data.items():
        path = directory / (split + ".jsonl")
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "counts": {s: len(r) for s, r in data.items()},
        "files": files,
        "labels": list(LOCATIONS),
        "groups_cross_split": False,
        "split_policy": "Per target object, unordered location pair, and final distractor location:6/1/1 groups; both event orders and both questions stay together.",
        "control": "Counterfactuals preserve the event multiset but swap the target object moves; distractor question answer stays unchanged.",
        "test_selection": "Fixed before model runs; checkpoint selection uses validation only.",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
