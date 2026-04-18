"""
Shared network building blocks used by all baselines.
All baselines use the same MLP architecture and hyperparameter conventions
so that comparisons are fair (no baseline gets an architectural advantage).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


def mlp(
    in_dim: int,
    hidden_dims: List[int],
    out_dim: int,
    activation: nn.Module = nn.ReLU,
    output_activation: nn.Module = None,
) -> nn.Sequential:
    layers = []
    dims = [in_dim] + hidden_dims
    for i in range(len(dims) - 1):
        layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
    layers.append(nn.Linear(dims[-1], out_dim))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class QNetwork(nn.Module):
    """Twin Q-networks (clipped double Q) used by CQL, TD3+BC, IQL."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        dims = [hidden_dim] * n_layers
        self.q1 = mlp(obs_dim + act_dim, dims, 1)
        self.q2 = mlp(obs_dim + act_dim, dims, 1)

    def both(self, obs: torch.Tensor, act: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.both(obs, act)
        return torch.min(q1, q2)


class VNetwork(nn.Module):
    """Value network V(s) used by IQL."""

    def __init__(self, obs_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        dims = [hidden_dim] * n_layers
        self.net = mlp(obs_dim, dims, 1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


class DeterministicActor(nn.Module):
    """Deterministic policy π(s) → a used by TD3+BC."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        dims = [hidden_dim] * n_layers
        self.net = mlp(obs_dim, dims, act_dim, output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class GaussianActor(nn.Module):
    """
    Stochastic Gaussian policy used by CQL (SAC-style) and IQL.
    Outputs mean and log_std; samples via reparameterization.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
    ):
        super().__init__()
        dims = [hidden_dim] * n_layers
        self.net      = mlp(obs_dim, dims, hidden_dim)
        self.mu_head  = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
        with_log_prob: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h       = F.relu(self.net(obs))
        mu      = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std     = log_std.exp()

        if deterministic:
            action = torch.tanh(mu)
            log_prob = torch.zeros(obs.shape[0], device=obs.device)
        else:
            dist     = torch.distributions.Normal(mu, std)
            x_t      = dist.rsample()
            action   = torch.tanh(x_t)
            if with_log_prob:
                log_prob = dist.log_prob(x_t).sum(-1)
                log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            else:
                log_prob = torch.zeros(obs.shape[0], device=obs.device)

        return action, log_prob


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for pt, ps in zip(target.parameters(), source.parameters()):
        pt.data.copy_(tau * ps.data + (1 - tau) * pt.data)


def compute_normalized_score(reward: float, env_name: str, ref_scores: dict) -> float:
    if env_name not in ref_scores:
        return reward
    refs = ref_scores[env_name]
    return (reward - refs["random"]) / (refs["expert"] - refs["random"]) * 100
