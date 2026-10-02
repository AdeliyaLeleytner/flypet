"""Matched decoders for a controlled readout of cached physiological states.

The caller supplies the same final-state features to both arms. This module has
no encoder, recurrent state, access to histories, or language-model parameters.
Question IDs identify the queried object, never its location/answer.
"""

from collections.abc import Mapping
import math

import torch
from torch import nn
from torch.nn import functional as F


QUERY_OBJECTS = ("key", "coin", "ring")


class CachedStateQuestionDecoder(nn.Module):
    """An MLP producing soft-prefix tokens from fixed brain-state features.

    ``conditioned=False`` supplies zero question channels to the *same-sized*
    MLP. All question-column weights start at zero in both arms, so copying the
    state dict gives exactly equal initial prefixes. Arm choice is recorded in
    :meth:`make_checkpoint`, not in the weight state dict; copying weights from
    one arm to another therefore does not silently change the recipient arm.

    Optional normalization must be fitted on training features only and supplied
    identically to both arms. Standard deviations are floored at 0.01 and the
    standardized features clipped to [-10, 10] by default. Omitting both
    statistics leaves features unchanged, including no clipping.
    """

    CHECKPOINT_VERSION = 1

    def __init__(
        self,
        feature_dim,
        embedding_dim,
        embedding_std,
        *,
        tokens=8,
        hidden=256,
        conditioned=False,
        feature_mean=None,
        feature_std=None,
        normalization_min_std=0.01,
        normalization_clip=10.0,
    ):
        super().__init__()
        for name, value in (
            ("feature_dim", feature_dim),
            ("embedding_dim", embedding_dim),
            ("tokens", tokens),
            ("hidden", hidden),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(conditioned, bool):
            raise ValueError("conditioned must be a boolean")
        if not math.isfinite(float(embedding_std)) or float(embedding_std) <= 0:
            raise ValueError("embedding_std must be finite and positive")
        if (feature_mean is None) != (feature_std is None):
            raise ValueError("Supply both normalization statistics or neither")
        if not math.isfinite(float(normalization_min_std)) or normalization_min_std <= 0:
            raise ValueError("normalization_min_std must be finite and positive")
        if not math.isfinite(float(normalization_clip)) or normalization_clip <= 0:
            raise ValueError("normalization_clip must be finite and positive")
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.embedding_std = float(embedding_std)
        self.tokens = tokens
        self.hidden = hidden
        self.conditioned = conditioned
        self.normalized = feature_mean is not None
        self.normalization_min_std = float(normalization_min_std)
        self.normalization_clip = float(normalization_clip)
        mean = (
            torch.zeros(feature_dim)
            if feature_mean is None
            else torch.as_tensor(feature_mean, dtype=torch.float32).detach().clone()
        )
        std = (
            torch.ones(feature_dim)
            if feature_std is None
            else torch.as_tensor(feature_std, dtype=torch.float32).detach().clone()
        )
        if mean.shape != (feature_dim,) or std.shape != (feature_dim,):
            raise ValueError("Normalization statistics must have shape [feature_dim]")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("Normalization statistics must be finite")
        if not (std >= 0).all():
            raise ValueError("Normalization standard deviations must be nonnegative")
        if self.normalized:
            std.clamp_(min=self.normalization_min_std)
        self.register_buffer("feature_mean", mean)
        self.register_buffer("feature_std", std)
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim + len(QUERY_OBJECTS), hidden),
            nn.SiLU(),
            nn.Linear(hidden, tokens * embedding_dim),
        )
        self.output_norm = nn.LayerNorm(embedding_dim)
        with torch.no_grad():
            self.decoder[0].weight[:, feature_dim:].zero_()

    def forward(self, features, query_ids):
        """Return prefixes from ``features[B,F]`` and integer ``query_ids[B]``.

        IDs follow ``QUERY_OBJECTS``. Plain-arm IDs are validated but have no
        effect on its output. Features are converted to the decoder's dtype;
        callers must put model and features on the same device explicitly.
        """
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("features must be a tensor of shape [batch, feature_dim]")
        if features.shape[0] == 0 or features.shape[1] != self.feature_dim:
            raise ValueError("features has an empty batch or wrong feature dimension")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("features must contain finite floating-point values")
        if features.device != self.decoder[0].weight.device:
            raise ValueError("features and decoder must be on the same device")
        if not isinstance(query_ids, torch.Tensor):
            query_ids = torch.as_tensor(query_ids, device=features.device)
        if query_ids.ndim != 1 or query_ids.shape[0] != features.shape[0]:
            raise ValueError("query_ids must have shape [batch]")
        if query_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("query_ids must be integers")
        if (query_ids < 0).any() or (query_ids >= len(QUERY_OBJECTS)).any():
            raise ValueError("query_ids contains an unknown object")
        x = features.to(dtype=self.decoder[0].weight.dtype)
        if self.normalized:
            x = ((x - self.feature_mean) / self.feature_std).clamp(
                min=-self.normalization_clip, max=self.normalization_clip
            )
        if self.conditioned:
            query = F.one_hot(
                query_ids.to(device=x.device, dtype=torch.long), num_classes=len(QUERY_OBJECTS)
            ).to(dtype=x.dtype)
        else:
            query = x.new_zeros((len(x), len(QUERY_OBJECTS)))
        projected = self.decoder(torch.cat((x, query), dim=-1))
        return (
            self.output_norm(projected.reshape(len(x), self.tokens, self.embedding_dim))
            * self.embedding_std
        )

    def initialize_from_bridge_state(self, state):
        """Copy the old physiological bridge decoder, adding zero query columns.

        Other bridge entries (encoder, gains, etc.) are intentionally ignored.
        Every required decoder tensor is checked before any parameter is changed.
        This method does not alter the supplied normalization statistics.
        """
        if not isinstance(state, Mapping):
            raise ValueError("bridge state must be a tensor mapping")
        expected = {
            "decoder.0.weight": (self.hidden, self.feature_dim),
            "decoder.0.bias": (self.hidden,),
            "decoder.2.weight": (self.tokens * self.embedding_dim, self.hidden),
            "decoder.2.bias": (self.tokens * self.embedding_dim,),
            "output_norm.weight": (self.embedding_dim,),
            "output_norm.bias": (self.embedding_dim,),
        }
        for key, shape in expected.items():
            value = state.get(key)
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
                raise ValueError(f"Missing or incompatible bridge tensor: {key}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"Non-finite or non-floating bridge tensor: {key}")
        target = self.state_dict()
        with torch.no_grad():
            self.decoder[0].weight[:, : self.feature_dim].copy_(state["decoder.0.weight"])
            self.decoder[0].weight[:, self.feature_dim :].zero_()
            for key in expected:
                if key != "decoder.0.weight":
                    target[key].copy_(state[key])
        return self

    def make_checkpoint(self):
        """Return an independent CPU snapshot, including arm and architecture."""
        return {
            "version": self.CHECKPOINT_VERSION,
            "query_objects": list(QUERY_OBJECTS),
            "config": {
                "feature_dim": self.feature_dim,
                "embedding_dim": self.embedding_dim,
                "embedding_std": self.embedding_std,
                "tokens": self.tokens,
                "hidden": self.hidden,
                "conditioned": self.conditioned,
                "normalization_min_std": self.normalization_min_std,
                "normalization_clip": self.normalization_clip,
            },
            "normalized": self.normalized,
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.state_dict().items()},
        }

    @classmethod
    def from_checkpoint(cls, checkpoint):
        """Restore a checkpoint on CPU; callers can then move it to a device."""
        if (
            not isinstance(checkpoint, Mapping)
            or checkpoint.get("version") != cls.CHECKPOINT_VERSION
        ):
            raise ValueError("Unknown decoder checkpoint schema")
        if checkpoint.get("query_objects") != list(QUERY_OBJECTS):
            raise ValueError("Checkpoint query-object mapping differs")
        state = checkpoint["state_dict"]
        model = cls(
            **checkpoint["config"],
            feature_mean=state["feature_mean"],
            feature_std=state["feature_std"],
        )
        model.load_state_dict(state, strict=True)
        model.normalized = bool(checkpoint["normalized"])
        return model
