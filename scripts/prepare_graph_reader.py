#!/usr/bin/env python3
"""Build the static graph prior for exactly the training-selected rate features."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flypet import connectome as C
from flypet.memory_reader_data import load_dataset, prepare_dataset, digest


def build_graph(neuron_indices, output, dataset_hashes=None, split_fingerprint=None):
    from flypet.corrections import masks, DEFAULT

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    order = C.neuron_order()[0]
    node_indices = np.asarray(neuron_indices, dtype=np.int64)
    position = np.full(len(order), -1, dtype=np.int64)
    position[node_indices] = np.arange(len(node_indices))
    pre, post, weight = C.connections()
    mask = (position[pre] >= 0) & (position[post] >= 0)
    p, q, w = pre[mask], post[mask], weight[mask]
    fake = SimpleNamespace(i_pre=p, _i_post=q, w0=w, flyid2i=C.id_maps()[0])
    factors = np.ones(len(w), dtype=np.float32)
    corrections = masks(fake)
    for name in DEFAULT:
        for selected, factor in corrections[name]:
            factors[selected] *= factor
    corrected = w.astype(np.float32) * factors
    keep = corrected != 0
    ann = C.annotations().reindex(order[node_indices])
    columns = ("cell_class", "cell_type", "top_nt", "side")
    codes, vocabs = [], {}
    for name in columns:
        values = ann[name].astype("string").fillna("unknown").replace("", "unknown").tolist()
        vocab = sorted(set(values))
        lookup = {v: i for i, v in enumerate(vocab)}
        codes.append([lookup[v] for v in values])
        vocabs[name] = vocab
    np.savez_compressed(
        out / "graph.npz",
        neuron_indices=node_indices,
        metadata=np.asarray(codes, dtype=np.int64).T,
        metadata_sizes=np.asarray([len(vocabs[c]) for c in columns]),
        src=position[p[keep]],
        dst=position[q[keep]],
        weight=corrected[keep],
    )
    manifest = {
        "graph_sha256": digest(out / "graph.npz"),
        "dataset_sources": dataset_hashes,
        "split_fingerprint": split_fingerprint,
        "n_nodes": len(node_indices),
        "n_edges": int(keep.sum()),
        "n_raw_induced_edges": int(mask.sum()),
        "positive_edges": int((corrected[keep] > 0).sum()),
        "negative_edges": int((corrected[keep] < 0).sum()),
        "metadata_columns": list(columns),
        "metadata_vocabularies": vocabs,
        "static_naive_corrections": list(DEFAULT),
        "learned_memory_access": False,
        "scope": "induced subgraph on train-selected downstream rate features",
        "relation_order": [
            "incoming_positive",
            "outgoing_positive",
            "incoming_negative",
            "outgoing_negative",
        ],
        "normalization": "absolute corrected synapse weight, row sum one per relation; zero rows stay zero",
        "sources": {
            str(p.relative_to(ROOT)): digest(p)
            for p in (
                C.PATH_CON_NPZ,
                C.PATH_ANN,
                C.PATH_COMP,
                ROOT / "flypet/connectome.py",
                ROOT / "flypet/corrections.py",
                Path(__file__).resolve(),
            )
        },
    }
    (out / "graph_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from flypet.corrections import DEFAULT

    generation = json.loads((Path(a.data) / "manifest.json").read_text())
    if (
        generation.get("status") != "complete"
        or tuple(generation.get("corrections", [])) != DEFAULT
    ):
        raise ValueError("Graph requires completed data with the same naive corrections")
    for path in (
        C.PATH_CON_NPZ,
        C.PATH_ANN,
        C.PATH_COMP,
        ROOT / "flypet/connectome.py",
        ROOT / "flypet/corrections.py",
    ):
        name = str(path.relative_to(ROOT))
        if generation.get("files", {}).get(name, {}).get("sha256") != digest(path):
            raise ValueError(f"Graph source differs from simulation generation: {name}")
    rows, rates, neurons, hashes = load_dataset(a.data)
    prepared = prepare_dataset(rows, rates, neurons)
    m = build_graph(
        prepared["neuron_indices"], a.out, hashes, prepared["manifest"]["split_fingerprint"]
    )
    print(
        json.dumps(
            {
                k: m[k]
                for k in ("n_nodes", "n_edges", "positive_edges", "negative_edges", "graph_sha256")
            }
        )
    )


if __name__ == "__main__":
    main()
