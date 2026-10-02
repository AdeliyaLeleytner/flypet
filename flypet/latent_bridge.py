"""Trainable bridges between neural observations and an open language model."""

from __future__ import annotations

import torch
from torch import nn


class NeuralReader(nn.Module):
    def __init__(
        self, n_neurons, channels, metadata, embedding_dim, embedding_std, tokens=16, width=128
    ):
        super().__init__()
        meta = torch.as_tensor(metadata, dtype=torch.long)
        if meta.shape[0] != n_neurons or width % 4:
            raise ValueError("Invalid reader geometry")
        self.register_buffer("metadata", meta)
        self.identity = nn.Embedding(n_neurons, width)
        self.categories = nn.ModuleList(
            nn.Embedding(int(meta[:, i].max()) + 1, width) for i in range(meta.shape[1])
        )
        self.input = nn.Sequential(nn.Linear(channels, width), nn.SiLU(), nn.Linear(width, width))
        self.node_norm = nn.LayerNorm(width)
        self.queries = nn.Parameter(torch.randn(tokens, width) * 0.02)
        self.attention = nn.MultiheadAttention(width, 4, batch_first=True)
        self.latent_norm = nn.LayerNorm(width)
        self.output = nn.Sequential(nn.Linear(width, embedding_dim), nn.LayerNorm(embedding_dim))
        self.auxiliary = nn.Linear(tokens * width, 3)
        self.embedding_std = float(embedding_std)

    def forward(self, features):
        h = self.input(features) + self.identity.weight[None]
        for i, layer in enumerate(self.categories):
            h = h + layer(self.metadata[:, i])[None]
        h = self.node_norm(h)
        queries = self.queries[None].expand(len(features), -1, -1)
        pooled, _ = self.attention(queries, h, h, need_weights=False)
        latent = self.latent_norm(pooled + queries)
        return self.output(latent) * self.embedding_std, self.auxiliary(latent.flatten(1))
