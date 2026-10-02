"""Experimental state readers; no changes to the pet's runtime/checkpoint.

All readers observe the same train-selected mean firing rates. The graph is a
STATIC induced subgraph with naive model corrections, never learned memory.
Attention and graph readers share node identities, metadata and latent pooling.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def relation_arrays(src, dst, weights, n):
    """Incoming/outgoing positive/negative, each receiver row normalized.

    Direction convention: A[receiver, sender] @ H aggregates sender features.
    Sign is a separate relation, not silently discarded or used in a degree sum.
    """
    src, dst, weights = np.asarray(src), np.asarray(dst), np.asarray(weights)
    if not (src.shape == dst.shape == weights.shape) or src.ndim != 1:
        raise ValueError("Edge arrays must be aligned vectors")
    if (
        not np.isfinite(weights).all()
        or np.any(src < 0)
        or np.any(dst < 0)
        or np.any(src >= n)
        or np.any(dst >= n)
    ):
        raise ValueError("Invalid edge weight or endpoint")
    result = []
    for sign in (1, -1):
        mask = weights * sign > 0
        s, t, w = src[mask], dst[mask], np.abs(weights[mask]).astype(np.float32)
        for receiver, sender in ((t, s), (s, t)):
            mass = np.bincount(receiver, weights=w, minlength=n)
            normalized = w / np.maximum(mass[receiver], 1e-12)
            result.append(
                (np.stack((receiver, sender)).astype(np.int64), normalized.astype(np.float32))
            )
    return result


class DenseProjector(nn.Module):
    def __init__(self, n, tokens, output_dim, hidden=1024, **unused):
        super().__init__()
        self.tokens, self.output_dim = tokens, output_dim
        self.net = nn.Sequential(
            nn.Linear(n, hidden), nn.GELU(), nn.Linear(hidden, tokens * output_dim)
        )
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        return self.norm(self.net(x).reshape(len(x), self.tokens, self.output_dim))


class GraphLayer(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.self_map = nn.Linear(width, width)
        self.relations = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(4))
        self.norm = nn.LayerNorm(width)

    def forward(self, h, adjacency):
        batch, n, width = h.shape
        flat = h.permute(1, 0, 2).reshape(n, batch * width).contiguous()
        message = self.self_map(h)
        for matrix, transform in zip(adjacency, self.relations):
            aggregated = torch.sparse.mm(matrix, flat).reshape(n, batch, width).permute(1, 0, 2)
            message = message + transform(aggregated)
        return self.norm(h + torch.nn.functional.gelu(message))


class AttentionProjector(nn.Module):
    def __init__(
        self,
        n,
        tokens,
        output_dim,
        metadata,
        metadata_sizes,
        width=128,
        graph=None,
        graph_layers=2,
        **unused,
    ):
        super().__init__()
        if width % 4:
            raise ValueError("Width must be divisible by four attention heads")
        metadata = np.asarray(metadata)
        if metadata.shape != (n, len(metadata_sizes)):
            raise ValueError("Metadata must match the selected neuron order")
        self.register_buffer("metadata", torch.as_tensor(metadata, dtype=torch.long))
        self.identity = nn.Embedding(n, width)
        self.categories = nn.ModuleList(nn.Embedding(int(size), width) for size in metadata_sizes)
        self.rate = nn.Linear(1, width)
        self.node_encoder = nn.Sequential(nn.LayerNorm(width), nn.GELU(), nn.Linear(width, width))
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True, dropout=0)
        self.latent_norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        # Instantiate graph-only parameters last so common weights have identical
        # initialization for attention and graph under the same seed.
        self.graph_layers = nn.ModuleList(
            GraphLayer(width) for _ in range(graph_layers if graph else 0)
        )
        if graph:
            arrays = relation_arrays(graph["src"], graph["dst"], graph["weight"], n)
            for i, (indices, values) in enumerate(arrays):
                a = torch.sparse_coo_tensor(
                    torch.from_numpy(indices),
                    torch.from_numpy(values),
                    (n, n),
                    check_invariants=True,
                ).coalesce()
                self.register_buffer(f"adjacency_{i}", a)

    def forward(self, x):
        h = self.rate(x.unsqueeze(-1)) + self.identity.weight.unsqueeze(0)
        for j, embedding in enumerate(self.categories):
            h = h + embedding(self.metadata[:, j]).unsqueeze(0)
        h = self.node_encoder(h)
        if self.graph_layers:
            adjacency = [getattr(self, f"adjacency_{i}") for i in range(4)]
            for layer in self.graph_layers:
                h = layer(h, adjacency)
        queries = self.queries.unsqueeze(0).expand(len(x), -1, -1)
        pooled, _ = self.attention(queries, h, h, need_weights=False)
        return self.output_norm(self.output(self.latent_norm(queries + pooled)))


def make_projector(mode, n, tokens, output_dim, graph_data, hidden=1024, width=128):
    if mode == "mlp":
        return DenseProjector(n, tokens, output_dim, hidden)
    if mode not in ("attention", "graph"):
        raise ValueError(mode)
    return AttentionProjector(
        n,
        tokens,
        output_dim,
        graph_data["metadata"],
        graph_data["metadata_sizes"],
        width=width,
        graph=graph_data if mode == "graph" else None,
    )
