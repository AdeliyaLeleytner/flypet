"""Trainable physiological core conditioned language model; history never enters LM context."""

import math
import numpy as np
import torch
from torch import nn
from .physiology_torch import PhysiologicalCore

MODEL_ID = "Qwen/Qwen3-4B"
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class PhysiologicalLanguageBridge(nn.Module):
    def __init__(
        self,
        graph,
        embedding_dim,
        embedding_std,
        tokens=8,
        hidden=256,
        ticks_per_event=128,
        checkpoint_steps=64,
        max_drive_mv=160.0,
        train_core=True,
        physics_dtype="float32",
        surrogate_scale=0.1,
    ):
        super().__init__()
        self.core = PhysiologicalCore(
            len(graph["order"]),
            graph["pre"],
            graph["post"],
            graph["contacts"],
            surrogate_scale=surrogate_scale,
        )
        if physics_dtype == "float64":
            self.core.double()
            with torch.no_grad():
                self.core.raw_gain.fill_(math.log(0.75))
        elif physics_dtype != "float32":
            raise ValueError("Unknown physics precision")
        self.core.raw_gain.requires_grad_(train_core)
        self.register_buffer(
            "input_indices", torch.as_tensor(graph["input_indices"], dtype=torch.long)
        )
        self.register_buffer(
            "output_indices", torch.as_tensor(graph["output_indices"], dtype=torch.long)
        )
        self.encoder = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(self.input_indices)),
        )
        nn.init.constant_(self.encoder[-1].bias, -1.0)
        self.decoder = nn.Sequential(
            nn.Linear(2 * len(self.output_indices), hidden),
            nn.SiLU(),
            nn.Linear(hidden, tokens * embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.tokens = tokens
        self.embedding_dim = embedding_dim
        self.embedding_std = float(embedding_std)
        self.ticks_per_event = ticks_per_event
        self.checkpoint_steps = checkpoint_steps
        self.max_drive_mv = max_drive_mv

    def forward(self, event_embeddings, intervention="normal"):
        """event_embeddings[B,events,D], each event encoded independently with frozen token embeddings."""
        if intervention not in ("normal", "zero_state", "reverse_history"):
            raise ValueError("Unknown intervention")
        if intervention == "zero_state":
            features = event_embeddings.new_zeros(
                len(event_embeddings), 2 * len(self.output_indices)
            )
            projected = self.decoder(features.float()).reshape(
                len(features), self.tokens, self.embedding_dim
            )
            prefix = self.output_norm(projected) * self.embedding_std
            zero = features.new_zeros(())
            return prefix, {
                "spikes": zero,
                "active_neurons": zero,
                "state_rms": zero,
                "prefix_rms": prefix.detach().square().mean().sqrt(),
            }
        if intervention == "reverse_history":
            event_embeddings = event_embeddings.flip(1)
        currents = self.max_drive_mv * torch.sigmoid(self.encoder(event_embeddings.float()))
        drive = currents.repeat_interleave(self.ticks_per_event, dim=1).transpose(0, 1).contiguous()
        state = self.core(
            drive.to(self.core.raw_gain.dtype),
            self.input_indices,
            self.output_indices,
            checkpoint_steps=self.checkpoint_steps,
            readout_ticks=min(64, len(drive)),
        )
        # Only final physiological state reaches the decoder. No cumulative
        # history embedding, past-text cache or event counts bypasses the brain.
        features = torch.cat((state["voltage_mv"] / 7.0, state["synaptic_mv"] / 20.0), dim=-1)
        projected = self.decoder(features.float()).reshape(
            len(features), self.tokens, self.embedding_dim
        )
        prefix = self.output_norm(projected) * self.embedding_std
        return prefix, {
            "spikes": state["spike_count"].detach(),
            "active_neurons": (state["neuron_spike_counts"] > 0).sum().detach(),
            "state_rms": features.detach().square().mean().sqrt(),
            "prefix_rms": prefix.detach().square().mean().sqrt(),
        }

    def trainable_state(self):
        # Exclude large, fixed anatomical buffers; graph hash is separately pinned.
        return {
            k: v.detach().cpu().clone()
            for k, v in self.state_dict().items()
            if not k.startswith("core.w") and k not in ("input_indices", "output_indices")
        }

    def load_trainable_state(self, state):
        result = self.load_state_dict(state, strict=False)
        if (
            set(result.missing_keys) != {"core.w", "core.wt", "input_indices", "output_indices"}
            or result.unexpected_keys
        ):
            raise ValueError("Bridge checkpoint schema differs")
