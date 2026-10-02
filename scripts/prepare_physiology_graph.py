"""Prepare immutable corrected graph and declared engineered text ports."""

import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def prepare(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    source = ROOT / "data/connectivity_783.npz"
    with np.load(source) as z:
        order = z["order"]
        pre = z["i_pre"]
        post = z["i_post"]
        w = z["w"].astype(np.float32)
    ann = (
        pd.read_csv(
            ROOT / "data/flywire_annotations_783.tsv",
            sep="\t",
            usecols=["root_id", "cell_class", "cell_type", "top_nt"],
        )
        .set_index("root_id")
        .reindex(order)
        .fillna("")
    )
    cc = ann.cell_class.to_numpy()
    nt = ann.top_nt.to_numpy()
    ct = ann.cell_type.to_numpy()
    scale = np.ones(len(w), np.float32)
    records = {}
    modifications = [
        ("ALLN_ACh_removed", (cc[pre] == "ALLN") & (nt[pre] == "acetylcholine") & (w > 0), 0.0),
        (
            "ALLN_monoamine_inhibitory",
            (cc[pre] == "ALLN")
            & np.isin(nt[pre], ["dopamine", "serotonin", "octopamine"])
            & (w > 0),
            -1.0,
        ),
        ("DAN_no_fast_synapse", cc[pre] == "DAN", 0.0),
        ("PN_KC_gain", (cc[pre] == "ALPN") & (cc[post] == "Kenyon_Cell"), 2.0),
        ("KC_MBON_gain", (cc[pre] == "Kenyon_Cell") & (cc[post] == "MBON"), 3.0),
    ]
    for name, mask, factor in modifications:
        scale[mask] *= factor
        records[name] = {"edges": int(mask.sum()), "factor": factor}
    corrected = w * scale
    groups = [
        np.flatnonzero((cc == "olfactory") & (ct == t)) for t in sorted(set(ct[cc == "olfactory"]))
    ]
    inputs = []
    for offset in range(max(map(len, groups))):
        for group in groups:
            if offset < len(group):
                inputs.append(int(group[offset]))
        if len(inputs) >= 128:
            break
    inputs = np.asarray(inputs[:128], np.int64)
    # Fixed distance-stratified ports selected without labels or trained responses.
    seen = set(inputs.tolist())
    frontier = inputs
    layers = []
    for _ in range(3):
        reached = post[np.isin(pre, frontier) & (corrected != 0)]
        nodes, counts = np.unique(reached, return_counts=True)
        valid = np.asarray([i not in seen for i in nodes])
        nodes = nodes[valid]
        counts = counts[valid]
        selected = nodes[np.argsort(-counts, kind="stable")[:128]]
        layers.append(selected)
        seen.update(nodes.tolist())
        frontier = nodes
    outputs = np.concatenate(layers).astype(np.int64)
    if len(inputs) != 128 or len(outputs) < 128 or np.intersect1d(inputs, outputs).size:
        raise ValueError("Insufficient/disjoint ports")
    masks = {
        name: np.flatnonzero(cc == name)
        for name in ["olfactory", "ALPN", "Kenyon_Cell", "MBON", "DAN"]
    }
    masks["descending"] = np.flatnonzero(np.char.startswith(ct.astype(str), "DN"))
    np.savez_compressed(
        out / "graph.npz",
        order=order,
        pre=pre,
        post=post,
        contacts=corrected,
        input_indices=inputs,
        output_indices=outputs,
        **{"population_" + k: v for k, v in masks.items()},
    )
    manifest = {
        "nodes": len(order),
        "edges": len(pre),
        "source_graph_sha256": sha(source),
        "annotation_sha256": sha(ROOT / "data/flywire_annotations_783.tsv"),
        "corrections_reference_sha256": sha(ROOT / "flypet/corrections.py"),
        "corrections": records,
        "graph_sha256": sha(out / "graph.npz"),
        "input_ports": len(inputs),
        "output_ports": len(outputs),
        "port_policy": "128 ORNs round-robin by type; up to128 disjoint reachable neurons at each of1,2,3hops. No label/response selection. Text-to-current mapping is engineered, not natural olfaction.",
        "output_population_counts": {
            str(k): int(v) for k, v in zip(*np.unique(cc[outputs], return_counts=True))
        },
        "physical_time_step_ms": 0.1,
        "pet_memory_loaded": False,
        "full_graph_retained": True,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    prepare(p.parse_args().out)
