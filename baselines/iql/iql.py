"""
Baseline 4: IQL — Offline Reinforcement Learning with Implicit Q-Learning
==========================================================================
Kostrikov, Nair & Levine, ICLR 2022.

Key idea: avoid querying Q at OOD actions entirely by reformulating the
Bellman backup using expectile regression on a separate V-function:
    L_V   = E_{s,a~D}[L_τ^2(Q(s,a) - V(s))]       expectile loss on V
    L_Q   = E[r + γ·V(s') - Q(s,a)]²               standard TD on Q
    L_π   = E[exp(β·(Q(s,a) - V(s))) · log π(a|s)] AWR-style actor

Since we only evaluate Q at dataset (s,a) pairs, IQL completely avoids
OOD action extrapolation — a clean theoretical advantage.

IQL is considered the strongest standard offline RL baseline as of 2022-2024
and consistently outperforms CQL on AntMaze tasks.

Reference: https://arxiv.org/abs/2110.06169
"""

from __future__ import annotations

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from baselines.shared.networks import QNetwork, VNetwork, GaussianActor, soft_update
from baselines.shared.trainer import FlatReplayBuffer, evaluate_policy_baseline, BaseTrainer


def expectile_loss(diff: torch.Tensor, expectile: float = 0.7) -> torch.Tensor:
    """
    Asymmetric L2 (expectile regression):
        L_τ²(u) = |τ - 1(u<0)| · u²
    τ > 0.5 focuses on upper expectile (optimistic value estimate).
    """
    weight = torch.where(diff > 0,
                         torch.full_like(diff, expectile),
                         torch.full_like(diff, 1 - expectile))
    return (weight * diff.pow(2)).mean()


class IQLAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor         = GaussianActor(obs_dim, act_dim, hidden_dim)
        self.critic        = QNetwork(obs_dim, act_dim, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        self.vf            = VNetwork(obs_dim, hidden_dim)

    def forward(self, obs: torch.Tensor):
        return self.actor(obs, deterministic=True)[0]


class IQLTrainer(BaseTrainer):
    """
    IQL trainer.

    Key hyperparameters (from original paper):
      expectile τ = 0.7 (locomotion), 0.9 (antmaze)
      temperature β = 3.0 (locomotion), 10.0 (antmaze)
      tau = 0.005, gamma = 0.99
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config,
        save_dir: str,
        device: str,
        expectile:   float = 0.7,
        temperature: float = 3.0,
        gamma:       float = 0.99,
        tau:         float = 0.005,
    ):
        super().__init__("iql", config, save_dir, device)
        self.expectile   = expectile
        self.temperature = temperature
        self.gamma       = gamma
        self.tau         = tau

        self.agent     = IQLAgent(obs_dim, act_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(),  lr=3e-4)
        self.critic_opt= torch.optim.Adam(self.agent.critic.parameters(), lr=3e-4)
        self.vf_opt    = torch.optim.Adam(self.agent.vf.parameters(),     lr=3e-4)

    def train(
        self,
        data: Dict[str, np.ndarray],
        env=None,
        normalizers: dict = None,
        n_steps: int = 1_000_000,
        batch_size: int = 256,
        eval_every: int = 5000,
        log_every:  int = 1000,
    ) -> float:
        buffer = FlatReplayBuffer(
            data["observations"].shape[1],
            data["actions"].shape[1],
        )
        buffer.add_d4rl(data)

        obs_mean = data["observations"].mean(0)
        obs_std  = data["observations"].std(0) + 1e-8

        print(f"\n[IQL] Training for {n_steps:,} steps | τ={self.expectile} β={self.temperature}")
        t0 = time.time()

        for step in range(1, n_steps + 1):
            batch = buffer.sample(batch_size, self.device)
            obs, actions, rewards = batch["obs"], batch["actions"], batch["rewards"]
            next_obs, terminals   = batch["next_obs"], batch["terminals"]

            # ── V update (expectile regression) ───────────────────────────────
            with torch.no_grad():
                q1, q2  = self.agent.critic_target.both(obs, actions)
                q_min   = torch.min(q1, q2)   # (B,)

            v        = self.agent.vf(obs)      # (B,)
            vf_loss  = expectile_loss(q_min - v, self.expectile)

            self.vf_opt.zero_grad()
            vf_loss.backward()
            self.vf_opt.step()

            # ── Q update (standard TD using V for backup) ─────────────────────
            with torch.no_grad():
                v_next   = self.agent.vf(next_obs)
                q_target = rewards + self.gamma * (1 - terminals) * v_next

            q1, q2      = self.agent.critic.both(obs, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            soft_update(self.agent.critic_target, self.agent.critic, self.tau)

            # ── Actor update (AWR / advantage-weighted regression) ────────────
            with torch.no_grad():
                q1, q2  = self.agent.critic_target.both(obs, actions)
                q_min   = torch.min(q1, q2)
                v       = self.agent.vf(obs)
                adv     = q_min - v                                # (B,)
                # Exponentiated advantage (clamped for stability)
                weights = torch.exp(self.temperature * adv).clamp(max=100.0)

            # Log probability of dataset actions under current policy
            _, log_prob = self.agent.actor(obs, with_log_prob=True)
            # Approximate: re-compute log prob at dataset actions
            # IQL uses log π(a_data | s) weighted by exponentiated advantage
            pi, _        = self.agent.actor(obs)
            log_pi_data  = -F.mse_loss(pi, actions, reduction='none').sum(-1) * 0.5  # Gaussian approx

            actor_loss   = -(weights * log_pi_data).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            if step % log_every == 0:
                elapsed = time.time() - t0
                self._log({
                    "vf_loss":     vf_loss.item(),
                    "critic_loss": critic_loss.item(),
                    "actor_loss":  actor_loss.item(),
                    "adv_mean":    adv.mean().item(),
                }, step)
                print(
                    f"  [IQL {step:>7d}] "
                    f"vf={vf_loss.item():.4f}  q={critic_loss.item():.4f}  "
                    f"actor={actor_loss.item():.4f}  adv={adv.mean().item():.3f}  "
                    f"({log_every/elapsed:.0f} it/s)"
                )
                t0 = time.time()

            if env is not None and step % eval_every == 0:
                _, score = evaluate_policy_baseline(
                    self.agent, env, obs_mean, obs_std,
                    n_episodes=10, device=self.device,
                    env_name=getattr(self.config, 'env_name', ''),
                    ref_scores=getattr(self.config, 'D4RL_REF_SCORES', {}),
                )
                print(f"  [IQL eval {step}] normalized_score={score:.2f}")
                self._log({"eval_score": score}, step)
                self.maybe_save(score, {
                    "actor":  self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                    "vf":     self.agent.vf.state_dict(),
                })

        self.close()
        print(f"[IQL] Done. Best score: {self.best_score:.2f}")
        return self.best_score
