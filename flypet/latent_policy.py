"""Continuous policy and group-relative objective for frozen latent observations.

The policy acts on contextual LLM hidden states. Brian2 is outside autograd;
policy gradients use the recorded probability of the sampled continuous action.
"""

from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


class ContinuousPolicy(nn.Module):
    def __init__(self, feature_dim, hidden=128, action_high=(1.5, 32.0, 32.0)):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, len(action_high)),
        )
        self.raw_scale = nn.Parameter(torch.zeros(len(action_high)))
        self.register_buffer("action_high", torch.tensor(action_high, dtype=torch.float32))

    def parameters_for(self, features):
        mean = self.net(features)
        # Smooth bounded scale, with a nonzero exploration floor. Avoid a hard
        # parameter clamp that can strand a parameter outside its gradient band.
        log_std = (-0.4 + 1.4 * torch.sigmoid(self.raw_scale)).expand_as(mean)
        return mean, log_std

    def log_prob(self, features, unconstrained_action):
        mean, log_std = self.parameters_for(features)
        z = unconstrained_action
        normal = (
            -0.5 * ((z - mean) * torch.exp(-log_std)).square()
            - log_std
            - 0.5 * math.log(2 * math.pi)
        )
        # Stable log |d(high*sigmoid(z))/dz|, including saturated actions.
        jacobian = self.action_high.log() + F.logsigmoid(z) + F.logsigmoid(-z)
        return (normal - jacobian).sum(-1)

    def sample(self, features, generator=None, deterministic=False):
        mean, log_std = self.parameters_for(features)
        z = (
            mean
            if deterministic
            else mean
            + torch.exp(log_std)
            * torch.randn(mean.shape, device=mean.device, dtype=mean.dtype, generator=generator)
        )
        return self.action_high * torch.sigmoid(z), z, self.log_prob(features, z)

    def imitation_loss(self, features, actions):
        target = torch.logit((actions / self.action_high).clamp(0.0001, 0.9999)).clamp(-4.0, 4.0)
        mean, log_std = self.parameters_for(features)
        return (0.5 * ((target - mean) * torch.exp(-log_std)).square() + log_std).mean()

    def kl_to(self, reference, features):
        mean, log_std = self.parameters_for(features)
        with torch.no_grad():
            other_mean, other_log_std = reference.parameters_for(features)
        # The shared invertible sigmoid transform preserves KL exactly.
        return (
            other_log_std
            - log_std
            + (torch.exp(2 * log_std) + (mean - other_mean).square())
            / (2 * torch.exp(2 * other_log_std))
            - 0.5
        ).sum(-1)


def group_advantages(rewards, epsilon=1e-6, std_floor=0.05):
    """One row is attempts of one task from one initial state/noise seed."""
    centered = rewards - rewards.mean(-1, keepdim=True)
    std = rewards.std(-1, unbiased=False, keepdim=True)
    return torch.where(
        std > epsilon, centered / std.clamp_min(max(epsilon, std_floor)), torch.zeros_like(centered)
    )


def clipped_objective(
    policy, reference, features, z, old_log_prob, advantage, clip=0.2, kl_weight=0.01
):
    logp = policy.log_prob(features, z)
    ratio = torch.exp(logp - old_log_prob.detach())
    surrogate = torch.minimum(
        ratio * advantage.detach(), ratio.clamp(1 - clip, 1 + clip) * advantage.detach()
    )
    kl = policy.kl_to(reference, features)
    return (-surrogate + kl_weight * kl).mean(), {
        "kl": kl.detach().mean(),
        "ratio": ratio.detach().mean(),
    }
