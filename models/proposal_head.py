"""
Proposal Head — expands single backbone representation h into K candidate actions.

P2 (default): Gaussian head — outputs μ, σ per action dim, samples K candidates.
  - Diversity is automatic (sampling variance)
  - σ is the uncertainty signal for UCB exploration
  - Trained with: NLL loss (maximize likelihood of true action under learned Gaussian)

P1 (ablation): MLP fan-out — outputs K×action_dim vector, sliced into K candidates.
  - Needs explicit diversity loss
  - Fully deterministic (no stochasticity)
  - Trained with: MSE to true action + diversity regularization
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


# ── P2: Gaussian Proposal Head ────────────────────────────────────────────────

class GaussianProposalHead(nn.Module):
    """
    Maps backbone hidden state h → Gaussian(μ, σ) over action space.
    Samples K candidates from this distribution.

    Architecture:
        h (hidden_dim) → shared trunk → μ head (act_dim)
                                      → log_σ head (act_dim)
        candidates: âₖ ~ μ + σ · εₖ,  εₖ ~ N(0,I),  k=1..K

    Loss (behavioral cloning, offline phase):
        NLL = -log N(a_true | μ, σ²)    ← maximize likelihood of observed action
    """

    def __init__(
        self,
        hidden_dim: int,
        act_dim: int,
        n_candidates: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        log_sigma_min: float = -5.0,
        log_sigma_max: float = 2.0,
        action_scale: float = 1.0,   # for tanh-bounded action spaces
    ):
        super().__init__()
        self.act_dim       = act_dim
        self.n_candidates  = n_candidates
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max
        self.action_scale  = action_scale

        # Shared trunk
        self.trunk = _mlp(hidden_dim, hidden_size, hidden_size, n_layers)

        # Mean head
        self.mu_head = nn.Linear(hidden_size, act_dim)

        # Log-sigma head
        self.log_sigma_head = nn.Linear(hidden_size, act_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.zeros_(self.mu_head.bias)
        nn.init.orthogonal_(self.log_sigma_head.weight, gain=0.01)
        nn.init.constant_(self.log_sigma_head.bias, -1.0)  # start with small sigma

    def forward(
        self,
        h: torch.Tensor,              # (B, hidden_dim)  — query row representation
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            candidates : (B, K, act_dim)   — K sampled action candidates
            mu         : (B, act_dim)      — distribution mean
            sigma      : (B, act_dim)      — distribution std (uncertainty signal)
        """
        trunk_out = self.trunk(h)                                  # (B, hidden)
        mu        = self.mu_head(trunk_out)                        # (B, act_dim)
        log_sigma = self.log_sigma_head(trunk_out)                 # (B, act_dim)
        log_sigma = torch.clamp(log_sigma, self.log_sigma_min, self.log_sigma_max)
        sigma     = log_sigma.exp()                                # (B, act_dim)

        B = h.shape[0]
        K = self.n_candidates

        if deterministic:
            # Return K copies of the mean (for evaluation)
            candidates = mu.unsqueeze(1).expand(B, K, -1)
        else:
            # Sample K candidates via reparameterization trick
            eps        = torch.randn(B, K, self.act_dim, device=h.device)  # (B, K, act)
            candidates = mu.unsqueeze(1) + sigma.unsqueeze(1) * eps        # (B, K, act)

        # Squash to action bounds via tanh (standard for continuous control)
        candidates = torch.tanh(candidates) * self.action_scale

        return candidates, mu, sigma

    def log_prob(self, actions: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Log probability of actions under the Gaussian (before tanh squashing).
        Used for behavioral cloning NLL loss.

        actions : (B, act_dim)  — true actions from dataset
        Returns : (B,)          — per-sample log probs
        """
        # Unsquash true action through atanh (inverse of tanh)
        actions_clamped = actions.clamp(-0.999, 0.999) / self.action_scale
        actions_raw     = torch.atanh(actions_clamped)

        dist = torch.distributions.Normal(mu, sigma + 1e-8)
        log_p = dist.log_prob(actions_raw).sum(dim=-1)   # (B,)

        # Jacobian correction for tanh squashing
        log_p -= torch.log(1 - actions_clamped.pow(2) + 1e-6).sum(dim=-1)
        return log_p

    def bc_loss(
        self, h: torch.Tensor, true_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Behavioral cloning loss: NLL of true action under learned Gaussian.

        h            : (B, hidden_dim)
        true_actions : (B, act_dim)
        Returns:
            loss  : scalar
            info  : dict with mu_mean, sigma_mean for logging
        """
        _, mu, sigma = self.forward(h)
        nll  = -self.log_prob(true_actions, mu, sigma)   # (B,)
        loss = nll.mean()
        info = {
            "proposal/nll":       loss.item(),
            "proposal/mu_mean":   mu.abs().mean().item(),
            "proposal/sigma_mean": sigma.mean().item(),
        }
        return loss, info


# ── P1: MLP Fan-out Proposal Head (ablation) ──────────────────────────────────

class MLPProposalHead(nn.Module):
    """
    Deterministic MLP that outputs K action vectors simultaneously.
    h → Linear → reshape → [â₁, ..., âₖ]

    Needs diversity loss: penalize pairwise similarity between candidates.
    Behavioral cloning loss: MSE to nearest candidate (min over K).
    """

    def __init__(
        self,
        hidden_dim: int,
        act_dim: int,
        n_candidates: int,
        hidden_size: int = 256,
        n_layers: int = 2,
        diversity_loss_weight: float = 0.1,
        action_scale: float = 1.0,
    ):
        super().__init__()
        self.act_dim              = act_dim
        self.n_candidates         = n_candidates
        self.diversity_loss_weight= diversity_loss_weight
        self.action_scale         = action_scale

        self.net = _mlp(hidden_dim, hidden_size, n_candidates * act_dim, n_layers)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, h: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            candidates : (B, K, act_dim)
            mu         : (B, act_dim)       — mean of candidates (for interface compat)
            sigma      : (B, act_dim)       — std of candidates  (not principled)
        """
        B = h.shape[0]
        out        = self.net(h)                                    # (B, K*act_dim)
        candidates = out.view(B, self.n_candidates, self.act_dim)   # (B, K, act_dim)
        candidates = torch.tanh(candidates) * self.action_scale

        mu    = candidates.mean(dim=1)                              # (B, act_dim)
        sigma = candidates.std(dim=1) + 1e-8                       # (B, act_dim)
        return candidates, mu, sigma

    def bc_loss(
        self, h: torch.Tensor, true_actions: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        BC loss = min_k MSE(âₖ, a_true) + diversity regularization.
        """
        candidates, mu, sigma = self.forward(h)                    # (B, K, act)

        # Min-k MSE: penalize the closest candidate to true action
        diff = (candidates - true_actions.unsqueeze(1)).pow(2).sum(-1)  # (B, K)
        bc_loss = diff.min(dim=1).values.mean()                    # scalar

        # Diversity loss: repulsion between candidates
        # Minimize mean pairwise cosine similarity
        cands_norm = F.normalize(candidates, dim=-1)               # (B, K, act)
        sim_matrix = torch.bmm(cands_norm, cands_norm.transpose(1, 2))  # (B, K, K)
        mask = ~torch.eye(self.n_candidates, dtype=torch.bool, device=h.device)
        div_loss = sim_matrix[:, mask].mean()

        loss = bc_loss + self.diversity_loss_weight * div_loss
        info = {
            "proposal/bc_loss":  bc_loss.item(),
            "proposal/div_loss": div_loss.item(),
        }
        return loss, info


# ── Factory ───────────────────────────────────────────────────────────────────

def build_proposal_head(config, hidden_dim: int, act_dim: int) -> nn.Module:
    if config.proposal_type == "gaussian":
        return GaussianProposalHead(
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            n_candidates=config.n_candidates,
            hidden_size=config.proposal_hidden_dim,
            n_layers=config.proposal_n_layers,
        )
    elif config.proposal_type == "mlp":
        return MLPProposalHead(
            hidden_dim=hidden_dim,
            act_dim=act_dim,
            n_candidates=config.n_candidates,
            hidden_size=config.proposal_hidden_dim,
            n_layers=config.proposal_n_layers,
            diversity_loss_weight=config.diversity_loss_weight,
        )
    else:
        raise ValueError(f"Unknown proposal_type: {config.proposal_type}")
