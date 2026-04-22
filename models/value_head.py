"""
Value Head — predicts Q̂(sₜ, âₖ) for each candidate action.

Design Y (default):
    Candidates are appended to the context table as new rows with masked rewards.
    The backbone re-encodes them: each candidate row cross-attends against prior
    (s,a,r) rows, effectively computing "how good is âₖ given what I've seen?"
    Value head reads the backbone's output representation for each candidate row.

Design X (shallow ablation, --shallow_value flag):
    Candidates are concatenated with h (from pass 1) and fed directly to an MLP.
    No backbone re-encoding. Faster but loses context-awareness.

CQL conservatism (offline phase):
    Penalize Q-values of OOD actions:
    L_CQL = E_{s,a~D}[Q(s,a)] - E_{s,â~N(0,1)}[Q(s,â)]
    This prevents the value head from assigning high Q to unseen actions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def _mlp(in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
    layers = [nn.Linear(in_dim, hidden_dim), nn.SiLU()]
    for _ in range(n_layers - 1):
        layers += [nn.Linear(hidden_dim, hidden_dim), nn.SiLU()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class ValueHead(nn.Module):
    """
    Maps backbone representation of (s, âₖ) row → scalar Q̂(s, âₖ).

    In Design Y, `h` comes from the backbone's second pass where the
    candidate row was in the table alongside context rows. The cross-attention
    has already done the "compare candidate to context" reasoning.

    In Design X (shallow), `h` is from pass 1 concatenated with âₖ.
    """

    def __init__(
        self,
        hidden_dim: int,
        hidden_size: int = 256,
        n_layers: int = 2,
    ):
        super().__init__()
        self.net = _mlp(hidden_dim, hidden_size, 1, n_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.net[-1].weight, gain=1.0)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h    : (B, K, hidden_dim)  — backbone reps for K candidate rows
        Returns: (B, K)           — Q-values for each candidate
        """
        B, K, D = h.shape
        q = self.net(h.reshape(B * K, D))   # (B*K, 1)
        return q.view(B, K)              # (B, K)


class ShallowValueHead(nn.Module):
    """
    Design X ablation: shallow MLP, no backbone re-encoding.
    Input = [h_pass1, âₖ] concatenated.
    """

    def __init__(self, hidden_dim: int, act_dim: int, hidden_size: int = 256, n_layers: int = 2):
        super().__init__()
        self.net = _mlp(hidden_dim + act_dim, hidden_size, 1, n_layers)

    def forward(self, h1: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        """
        h1         : (B, hidden_dim)  — from pass 1
        candidates : (B, K, act_dim)
        Returns    : (B, K)
        """
        B, K, A = candidates.shape
        h_exp = h1.unsqueeze(1).expand(B, K, -1)            # (B, K, hidden)
        inp   = torch.cat([h_exp, candidates], dim=-1)       # (B, K, hidden+act)
        q     = self.net(inp.reshape(B * K, -1))                # (B*K, 1)
        return q.view(B, K)                                  # (B, K)


class TDLoss(nn.Module):
    """
    Temporal Difference (TD) loss for the value head.

    Standard Q-learning target:
        y = r + γ · max_k Q̂_target(s', â'ₖ)

    With CQL conservatism (offline phase):
        L = MSE(Q̂(s,a), y)
          + α · [logsumexp_k Q̂(s, random_âₖ) - Q̂(s, a_dataset)]

    The CQL term penalizes high Q-values for out-of-distribution actions.
    """

    def __init__(self, gamma: float = 0.99, cql_alpha: float = 1.0, cql_n_random: int = 10):
        super().__init__()
        self.gamma      = gamma
        self.cql_alpha  = cql_alpha
        self.cql_n_random = cql_n_random

    def td_loss(
        self,
        q_pred:   torch.Tensor,   # (B,)  — Q̂(sₜ, aₜ) for the dataset action
        q_target: torch.Tensor,   # (B,)  — r + γ·max Q̂_target(s', â')
    ) -> torch.Tensor:
        return F.mse_loss(q_pred, q_target.detach())

    def cql_penalty(
        self,
        q_random:  torch.Tensor,  # (B, K)  — Q-values of random actions
        q_dataset: torch.Tensor,  # (B,)    — Q-value of dataset action
    ) -> torch.Tensor:
        """
        CQL penalty:
            logsumexp over random Q-values minus dataset Q-value.
        Encourages Q to be low for random/OOD actions and higher for dataset actions.
        """
        logsumexp_q = torch.logsumexp(q_random, dim=1)       # (B,)
        return (logsumexp_q - q_dataset).mean()

    def compute(
        self,
        q_pred:    torch.Tensor,   # (B,)
        q_target:  torch.Tensor,   # (B,)
        q_random:  Optional[torch.Tensor] = None,  # (B, K_rand)
    ) -> Tuple[torch.Tensor, dict]:
        """
        Full TD + CQL loss.

        Returns:
            total_loss : scalar
            info       : dict for logging
        """
        td = self.td_loss(q_pred, q_target)
        info = {
            "value/td_loss":    td.item(),
            "value/q_pred":     q_pred.mean().item(),
            "value/q_target":   q_target.mean().item(),
        }

        if self.cql_alpha > 0 and q_random is not None:
            cql = self.cql_penalty(q_random, q_pred)
            total = td + self.cql_alpha * cql
            info["value/cql_penalty"] = cql.item()
        else:
            total = td

        info["value/total_loss"] = total.item()
        return total, info


def build_value_head(config, hidden_dim: int, act_dim: int) -> nn.Module:
    if config.shallow_value:
        return ShallowValueHead(
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            hidden_size=config.value_hidden_dim,
            n_layers=config.value_n_layers,
        )
    else:
        return ValueHead(
            hidden_dim=hidden_dim,
            hidden_size=config.value_hidden_dim,
            n_layers=config.value_n_layers,
        )
