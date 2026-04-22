"""
TabRL Agent — full model combining backbone + proposal head + value head.

Implements the full two-pass inference loop:

    Pass 1 (Proposal):
        table = [context_rows: (s,a,r)] + [query_row: (s,?,?)]
        h1    = backbone(table)[:, query_position, :]   → (B, hidden)
        candidates, mu, sigma = proposal_head(h1)       → (B, K, act_dim)

    Pass 2 (Evaluation, Design Y):
        table2 = [context_rows: (s,a,r)] + [K candidate rows: (s,âₖ,?)]
        H2     = backbone(table2)[:, K_positions, :]    → (B, K, hidden)
        Q̂      = value_head(H2)                         → (B, K)

    Action selection:
        if use_ucb:  aₜ = argmax_k Q̂ₖ + β · σₖ
        else:        aₜ = argmax_k Q̂ₖ

    TD target (for target network):
        y = r + γ · (1-done) · max_k Q̂_target(s', â'ₖ)

    Also holds target networks (soft-updated copies of value head + backbone).
"""

from __future__ import annotations

import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

from .backbone import TabPFNBackbone, TransformerBackbone, build_backbone
from .proposal_head import build_proposal_head
from .value_head import build_value_head, TDLoss


class TabRLAgent(nn.Module):

    def __init__(self, config, obs_dim: int, act_dim: int):
        super().__init__()
        self.config   = config
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.device   = torch.device(config.device if torch.cuda.is_available() else "cpu")

        # Feature dim for context table: obs + act columns
        self.feature_dim = obs_dim + act_dim

        # ── Build components ─────────────────────────────────────────────────
        self.backbone      = build_backbone(config, self.feature_dim)
        hidden_dim         = self.backbone.hidden_dim

        self.proposal_head = build_proposal_head(config, hidden_dim, act_dim)
        self.value_head    = build_value_head(config, hidden_dim, act_dim)

        # Target network (slow-moving copy of value head for stable TD targets)
        self.value_head_target = copy.deepcopy(self.value_head)
        for p in self.value_head_target.parameters():
            p.requires_grad = False

        # Target backbone (only used when backbone is NOT frozen)
        if not config.freeze_backbone:
            self.backbone_target = copy.deepcopy(self.backbone)
            for p in self.backbone_target.parameters():
                p.requires_grad = False
        else:
            self.backbone_target = None

        # Loss
        self.td_loss_fn = TDLoss(
            gamma=config.gamma,
            cql_alpha=config.cql_alpha,
            cql_n_random=config.cql_n_random,
        )

        # Freeze backbone if requested
        if config.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print("[Agent] Backbone frozen — only training heads")
        else:
            print("[Agent] Backbone unfrozen — fine-tuning end-to-end")

        self.to(self.device)

    # ── Pass 1: Proposal ─────────────────────────────────────────────────────

    def propose(
        self,
        context_X: torch.Tensor,    # (B, L, obs+act)
        context_y: torch.Tensor,    # (B, L)
        query_obs: torch.Tensor,    # (B, obs_dim)
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pass 1: propose K candidate actions.

        Returns:
            candidates : (B, K, act_dim)
            mu         : (B, act_dim)
            sigma      : (B, act_dim)
            h1         : (B, hidden_dim)  — backbone rep for proposal step
        """
        # Query row: [obs, zeros_for_action]  — action column masked
        act_zeros = torch.zeros(
            query_obs.shape[0], 1, self.act_dim,
            device=query_obs.device
        )
        # Build query_X for pass 1: shape (B, 1, obs+act)
        query_X = torch.cat([
            query_obs.unsqueeze(1),                                # (B, 1, obs)
            act_zeros,                                             # (B, 1, act)
        ], dim=-1)                                                 # (B, 1, obs+act)

        # Run backbone
        H1 = self.backbone(context_X, context_y, query_X)         # (B, 1, hidden)
        h1 = H1[:, 0, :]                                           # (B, hidden)

        # Proposal head: expand h1 → K candidates
        candidates, mu, sigma = self.proposal_head(h1, deterministic=deterministic)

        return candidates, mu, sigma, h1

    # ── Pass 2: Evaluation ───────────────────────────────────────────────────

    def evaluate(
        self,
        context_X:  torch.Tensor,    # (B, L, obs+act)
        context_y:  torch.Tensor,    # (B, L)
        query_obs:  torch.Tensor,    # (B, obs_dim)
        candidates: torch.Tensor,    # (B, K, act_dim)
        use_target: bool = False,    # True when computing TD targets
    ) -> torch.Tensor:
        """
        Pass 2 (Design Y): re-encode candidates through backbone, score with value head.

        Each candidate becomes a row in the table: [sₜ, âₖ, ?]
        All K candidates enter the table simultaneously — one forward pass.
        The backbone cross-attends candidates against context rows.

        Returns:
            Q : (B, K)   — Q-value for each candidate
        """
        B, K, _ = candidates.shape

        # Build candidate query rows: (B, K, obs+act)
        obs_exp   = query_obs.unsqueeze(1).expand(B, K, -1)        # (B, K, obs)
        query_X   = torch.cat([obs_exp, candidates], dim=-1)       # (B, K, obs+act)

        # Select backbone and value head
        backbone   = self.backbone_target if (use_target and self.backbone_target) else self.backbone
        value_head = self.value_head_target if use_target else self.value_head

        if self.config.shallow_value:
            # Design X: no re-encoding, value head gets [h1, âₖ] directly
            # h1 must come from pass 1; caller handles this separately
            raise RuntimeError(
                "ShallowValueHead requires h1 from pass 1. "
                "Use evaluate_shallow() instead."
            )

        # Design Y: backbone re-encoding
        H2 = backbone(context_X, context_y, query_X)               # (B, K, hidden)
        Q  = value_head(H2)                                         # (B, K)

        return Q

    def evaluate_shallow(
        self,
        h1:         torch.Tensor,    # (B, hidden) — from pass 1
        candidates: torch.Tensor,    # (B, K, act_dim)
        use_target: bool = False,
    ) -> torch.Tensor:
        """
        Design X (ablation): shallow value scoring without backbone re-encoding.
        """
        value_head = self.value_head_target if use_target else self.value_head
        return value_head(h1, candidates)                           # (B, K)

    # ── Action Selection ─────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        context_X: torch.Tensor,    # (1, L, obs+act)
        context_y: torch.Tensor,    # (1, L)
        obs:       torch.Tensor,    # (1, obs_dim)
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, dict]:
        """
        Full inference loop: propose → evaluate → select.
        Returns action as numpy array for env.step().
        """
        self.eval()

        # Pass 1
        candidates, mu, sigma, h1 = self.propose(
            context_X, context_y, obs, deterministic=deterministic
        )

        # Pass 2
        if self.config.shallow_value:
            Q = self.evaluate_shallow(h1, candidates)
        else:
            Q = self.evaluate(context_X, context_y, obs, candidates)

        # Action selection
        if self.config.use_ucb and not deterministic:
            # UCB: argmax Q̂ + β·σ̂ (sigma averaged over act dims)
            sigma_scalar = sigma.mean(dim=-1, keepdim=True).expand_as(Q)  # rough per-candidate σ
            # Better: use per-candidate sigma from distribution variance
            # σ per candidate ≈ sigma (same distribution, different samples)
            ucb_scores   = Q + self.config.beta_ucb * sigma.mean(-1, keepdim=True)
            best_k       = ucb_scores.argmax(dim=1)                    # (B,)
        else:
            best_k = Q.argmax(dim=1)                                   # (B,)

        # Gather best action
        best_action = candidates[torch.arange(obs.shape[0]), best_k]  # (B, act)

        info = {
            "Q_mean":     Q.mean().item(),
            "Q_max":      Q.max().item(),
            "sigma_mean": sigma.mean().item(),
        }

        self.train()
        return best_action[0].cpu().numpy(), info

    # ── TD Training Step ─────────────────────────────────────────────────────

    def td_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Full TD update step. Handles both offline (CQL) and online phases.

        batch keys: context_X, context_y, query_obs, query_act, query_rew,
                    next_obs, terminal
        """
        context_X  = batch["context_X"].to(self.device)    # (B, L, obs+act)
        context_y  = batch["context_y"].to(self.device)    # (B, L)
        query_obs  = batch["query_obs"].to(self.device)    # (B, obs)
        query_act  = batch["query_act"].to(self.device)    # (B, act)
        rewards    = batch["query_rew"].to(self.device)    # (B,)
        next_obs   = batch["next_obs"].to(self.device)     # (B, obs)
        terminals  = batch["terminal"].to(self.device)     # (B,)

        B = context_X.shape[0]

        # ── Compute TD target (no gradient) ──────────────────────────────────
        with torch.no_grad():
            # Propose K candidates for next state
            next_cands, _, _, _ = self.propose(
                context_X, context_y, next_obs, deterministic=False
            )

            # Evaluate with target network
            if self.config.shallow_value:
                # Need h1 from target backbone for shallow path
                act_zeros = torch.zeros(B, 1, self.act_dim, device=self.device)
                next_qX   = torch.cat([next_obs.unsqueeze(1), act_zeros], dim=-1)
                H1_next   = (self.backbone_target or self.backbone)(context_X, context_y, next_qX)
                h1_next   = H1_next[:, 0, :]
                Q_next    = self.evaluate_shallow(h1_next, next_cands, use_target=True)
            else:
                Q_next = self.evaluate(
                    context_X, context_y, next_obs, next_cands, use_target=True
                )

            V_next   = Q_next.max(dim=1).values                    # (B,)
            # Clip V_next to prevent target explosion
            V_next    = torch.clamp(V_next, -200.0, 200.0)
            td_target = rewards + self.config.gamma * (1 - terminals) * V_next
            # Clip TD target itself
            td_target = torch.clamp(td_target, -200.0, 200.0)

        # ── Q̂(sₜ, aₜ) for dataset action ────────────────────────────────────
        # Wrap dataset action as a single "candidate"
        query_act_expanded = query_act.unsqueeze(1)                 # (B, 1, act)

        if self.config.shallow_value:
            _, _, _, h1 = self.propose(context_X, context_y, query_obs)
            Q_dataset   = self.evaluate_shallow(h1, query_act_expanded)[:, 0]
        else:
            Q_dataset = self.evaluate(
                context_X, context_y, query_obs, query_act_expanded
            )[:, 0]                                                  # (B,)

        # ── CQL random penalty ───────────────────────────────────────────────
        Q_random = None
        if self.config.cql_alpha > 0:
            random_acts = torch.rand(
                B, self.config.cql_n_random, self.act_dim, device=self.device
            ) * 2 - 1  # uniform in [-1, 1]

            if self.config.shallow_value:
                Q_random = self.evaluate_shallow(h1, random_acts)
            else:
                Q_random = self.evaluate(
                    context_X, context_y, query_obs, random_acts
                )

        # ── Loss ─────────────────────────────────────────────────────────────
        td_loss, info = self.td_loss_fn.compute(Q_dataset, td_target, Q_random)
        return td_loss, info

    def bc_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, dict]:
        """
        Behavioral cloning step for the proposal head.
        Maximize likelihood of dataset actions under learned Gaussian.
        """
        context_X = batch["context_X"].to(self.device)
        context_y = batch["context_y"].to(self.device)
        query_obs = batch["query_obs"].to(self.device)
        query_act = batch["query_act"].to(self.device)

        # Pass 1 to get h1
        _, _, _, h1 = self.propose(context_X, context_y, query_obs)

        # BC loss on proposal head
        loss, info = self.proposal_head.bc_loss(h1, query_act)
        return loss, info

    # ── Soft Target Network Update ────────────────────────────────────────────

    @torch.no_grad()
    def soft_update_target(self, tau: Optional[float] = None):
        """Polyak averaging: θ_target = τ·θ + (1-τ)·θ_target"""
        tau = tau or self.config.tau
        for p, pt in zip(self.value_head.parameters(), self.value_head_target.parameters()):
            pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

        if self.backbone_target is not None:
            for p, pt in zip(self.backbone.parameters(), self.backbone_target.parameters()):
                pt.data.copy_(tau * p.data + (1 - tau) * pt.data)

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save({
            "proposal_head":        self.proposal_head.state_dict(),
            "value_head":           self.value_head.state_dict(),
            "value_head_target":    self.value_head_target.state_dict(),
            "backbone_state":       self.backbone.state_dict() if not self.config.freeze_backbone else None,
        }, path)
        print(f"[Agent] Saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.proposal_head.load_state_dict(ckpt["proposal_head"])
        self.value_head.load_state_dict(ckpt["value_head"])
        self.value_head_target.load_state_dict(ckpt["value_head_target"])
        if ckpt["backbone_state"] is not None and not self.config.freeze_backbone:
            self.backbone.load_state_dict(ckpt["backbone_state"])
        print(f"[Agent] Loaded from {path}")
