"""
Baseline 2: TD3+BC — A Minimalist Approach to Offline Reinforcement Learning
==============================================================================
Fujimoto & Gu, NeurIPS 2021.

Key idea: standard TD3 actor-critic + a BC regularization term on the actor.
    L_actor = -λ · Q(s,π(s)) + ||π(s) - a_dataset||²
where λ normalizes the Q term by the mean absolute Q value.

This is the single most important baseline for any offline RL paper.
Almost every NeurIPS offline RL paper from 2021-2024 reports TD3+BC numbers.

Reference: https://arxiv.org/abs/2106.06860
Official code: https://github.com/sfujim/TD3_BC
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

from baselines.shared.networks import QNetwork, DeterministicActor, soft_update
from baselines.shared.trainer import FlatReplayBuffer, evaluate_policy_baseline, BaseTrainer


class TD3BCAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor        = DeterministicActor(obs_dim, act_dim, hidden_dim)
        self.critic       = QNetwork(obs_dim, act_dim, hidden_dim)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target= copy.deepcopy(self.critic)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class TD3BCTrainer(BaseTrainer):
    """
    TD3+BC trainer.

    Hyperparameters match the original paper for fair comparison:
      alpha=2.5, tau=0.005, gamma=0.99, policy_noise=0.2, noise_clip=0.5
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config,
        save_dir: str,
        device: str,
        alpha: float  = 2.5,    # BC weight (normalizing λ)
        gamma: float  = 0.99,
        tau:   float  = 0.005,
        policy_noise: float = 0.2,
        noise_clip:   float = 0.5,
        policy_freq:  int   = 2,
    ):
        super().__init__("td3bc", config, save_dir, device)
        self.alpha        = alpha
        self.gamma        = gamma
        self.tau          = tau
        self.policy_noise = policy_noise
        self.noise_clip   = noise_clip
        self.policy_freq  = policy_freq

        self.agent   = TD3BCAgent(obs_dim, act_dim).to(device)
        self.actor_opt  = torch.optim.Adam(self.agent.actor.parameters(),  lr=3e-4)
        self.critic_opt = torch.optim.Adam(self.agent.critic.parameters(), lr=3e-4)

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
        # Load data into replay buffer
        buffer = FlatReplayBuffer(
            data["observations"].shape[1],
            data["actions"].shape[1],
        )
        buffer.add_d4rl(data)

        obs_mean = data["observations"].mean(0)
        obs_std  = data["observations"].std(0) + 1e-8

        print(f"\n[TD3+BC] Training for {n_steps:,} steps | buffer={len(buffer):,}")
        t0   = time.time()

        for step in range(1, n_steps + 1):
            batch = buffer.sample(batch_size, self.device)
            obs, actions, rewards = batch["obs"], batch["actions"], batch["rewards"]
            next_obs, terminals   = batch["next_obs"], batch["terminals"]

            # ── Critic update ─────────────────────────────────────────────────
            with torch.no_grad():
                noise      = (torch.randn_like(actions) * self.policy_noise).clamp(
                    -self.noise_clip, self.noise_clip
                )
                next_acts  = (self.agent.actor_target(next_obs) + noise).clamp(-1, 1)
                q1_t, q2_t = self.agent.critic_target.both(next_obs, next_acts)
                q_target   = rewards + self.gamma * (1 - terminals) * torch.min(q1_t, q2_t)

            q1, q2       = self.agent.critic.both(obs, actions)
            critic_loss  = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # ── Actor update (every policy_freq steps) ────────────────────────
            actor_loss_val = 0.0
            if step % self.policy_freq == 0:
                pi          = self.agent.actor(obs)
                q_pi        = self.agent.critic.q1(
                    torch.cat([obs, pi], -1)
                ).squeeze(-1)

                # Normalization: λ = alpha / (mean |Q|)
                lam         = self.alpha / (q_pi.abs().mean().detach() + 1e-8)
                actor_loss  = -lam * q_pi.mean() + F.mse_loss(pi, actions)
                actor_loss_val = actor_loss.item()

                self.actor_opt.zero_grad()
                actor_loss.backward()
                self.actor_opt.step()

                soft_update(self.agent.actor_target,  self.agent.actor,  self.tau)
                soft_update(self.agent.critic_target, self.agent.critic, self.tau)

            if step % log_every == 0:
                elapsed = time.time() - t0
                self._log({
                    "critic_loss": critic_loss.item(),
                    "actor_loss":  actor_loss_val,
                }, step)
                print(
                    f"  [TD3+BC {step:>7d}] "
                    f"critic={critic_loss.item():.4f}  actor={actor_loss_val:.4f}  "
                    f"({log_every/elapsed:.0f} it/s)"
                )
                t0 = time.time()

            if env is not None and step % eval_every == 0:
                _, score = evaluate_policy_baseline(
                    self.agent.actor, env, obs_mean, obs_std,
                    n_episodes=10, device=self.device,
                    env_name=getattr(self.config, 'env_name', ''),
                    ref_scores=getattr(self.config, 'D4RL_REF_SCORES', {}),
                )
                print(f"  [TD3+BC eval {step}] normalized_score={score:.2f}")
                self._log({"eval_score": score}, step)
                self.maybe_save(score, {
                    "actor": self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                })

        self.close()
        print(f"[TD3+BC] Done. Best score: {self.best_score:.2f}")
        return self.best_score
