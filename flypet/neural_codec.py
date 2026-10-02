"""Learned rich MBON state bottleneck, pretrained before language alignment."""

import torch
from torch import nn


class NeuralCodec(nn.Module):
    def __init__(self, neurons=96, channels=9, tokens=4, width=128):
        super().__init__()
        self.neurons = neurons
        self.channels = channels
        self.tokens = tokens
        self.width = width
        self.register_buffer("mean", torch.zeros(neurons, channels))
        self.register_buffer("std", torch.ones(neurons, channels))
        self.encoder = nn.Sequential(
            nn.Linear(neurons * channels, 512), nn.SiLU(), nn.Linear(512, tokens * width)
        )
        self.token_norm = nn.LayerNorm(width)
        self.reconstruct = nn.Linear(tokens * width, neurons * channels)
        self.readout = nn.Sequential(nn.Linear(tokens * width, 128), nn.SiLU(), nn.Linear(128, 3))

    def encode(self, x):
        normalized = ((x - self.mean) / self.std).clamp(-20, 20)
        return self.token_norm(
            self.encoder(normalized.flatten(1)).reshape(len(x), self.tokens, self.width)
        )

    def forward(self, x):
        tokens = self.encode(x)
        normalized = ((x - self.mean) / self.std).clamp(-20, 20)
        flat = tokens.flatten(1)
        return tokens, self.readout(flat), self.reconstruct(flat).reshape_as(x), normalized
