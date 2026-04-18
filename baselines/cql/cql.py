"""
Baseline 3: CQL — Conservative Q-Learning
==========================================
Kumar et al., NeurIPS 2020.

Key idea: add a conservatism penalty to the Q-function that penalizes
Q-values at actions NOT in the dataset:
    L_CQL = E_{s,a~D}[Q(s,a)] - E_{s,â~μ}[Q(s,â)] + standard TD error

Uses SAC-style stochastic actor. The CQL(H) variant (used here) adds a
learned temperature α that trades off conservatism vs. reward maximization.

This is the second most important baseline after TD3+BC for offline RL.

Reference: https://arxiv.org/abs/2006.04779
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

from baselines.shared.networks import QNetwork, GaussianActor, soft_update
from baselines.shared.trainer import FlatReplayBuffer, evaluate_policy_baseline, BaseTrainer


class CQLAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.actor  = GaussianActor(obs_dim, act_dim, hidden_dim)
        self.critic = QNetwork(obs_dim, act_dim, hidden_dim)
        self.critic_target = copy.deepcopy(self.critic)
        # Learnable entropy temperature
        self.log_alpha = nn.Parameter(torch.zeros(1))

    @property
    def alpha(self):
        return self.log_alpha.exp()


class CQLTrainer(BaseTrainer):
    """
    CQL(H) trainer — SAC actor + conservative Q penalty.

    Key hyperparameters (from original paper):
      cql_weight=1.0, num_random=10, target_entropy=-act_dim
      tau=0.005, gamma=0.99
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config,
        save_dir: str,
        device: str,
        cql_weight:     float = 1.0,
        num_random:     int   = 10,
        gamma:          float = 0.99,
        tau:            float = 0.005,
        target_entropy: float = None,
    ):
        super().__init__("cql", config, save_dir, device)
        self.cql_weight     = cql_weight
        self.num_random     = num_random
        self.gamma          = gamma
        self.tau            = tau
        self.target_entropy = target_entropy if target_entropy else -act_dim
        self.act_dim        = act_dim

        self.agent     = CQLAgent(obs_dim, act_dim).to(device)
        self.actor_opt = torch.optim.Adam(self.agent.actor.parameters(),  lr=1e-4)
        self.critic_opt= torch.optim.Adam(self.agent.critic.parameters(), lr=3e-4)
        self.alpha_opt = torch.optim.Adam([self.agent.log_alpha],         lr=1e-4)

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

        print(f"\n[CQL] Training for {n_steps:,} steps | buffer={len(buffer):,}")
        t0 = time.time()

        for step in range(1, n_steps + 1):
            batch = buffer.sample(batch_size, self.device)
            obs, actions, rewards = batch["obs"], batch["actions"], batch["rewards"]
            next_obs, terminals   = batch["next_obs"], batch["terminals"]

            alpha = self.agent.alpha.detach()

            # ── Critic (Q) update ─────────────────────────────────────────────
            with torch.no_grad():
                next_acts, next_log_pi = self.agent.actor(next_obs)
                q1_t, q2_t = self.agent.critic_target.both(next_obs, next_acts)
                q_target   = rewards + self.gamma * (1 - terminals) * (
                    torch.min(q1_t, q2_t) - alpha * next_log_pi
                )

            q1, q2    = self.agent.critic.both(obs, actions)
            td_loss   = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            # CQL penalty: logsumexp over random + policy actions minus dataset Q
            # Sample random actions
            rand_acts = torch.FloatTensor(
                batch_size, self.num_random, self.act_dim
            ).uniform_(-1, 1).to(self.device)
            obs_exp = obs.unsqueeze(1).expand(-1, self.num_random, -1)  # (B, N, obs)

            # Q for random actions
            q1_rand = self.agent.critic.q1(
                torch.cat([obs_exp, rand_acts], -1).view(batch_size * self.num_random, -1)
            ).view(batch_size, self.num_random)
            q2_rand = self.agent.critic.q2(
                torch.cat([obs_exp, rand_acts], -1).view(batch_size * self.num_random, -1)
            ).view(batch_size, self.num_random)

            # Q for policy actions (current π)
            curr_acts, curr_log_pi = self.agent.actor(
                obs.unsqueeze(1).expand(-1, self.num_random, -1).reshape(
                    batch_size * self.num_random, -1
                )
            )
            q1_curr = self.agent.critic.q1(
                torch.cat([
                    obs_exp.reshape(batch_size * self.num_random, -1),
                    curr_acts
                ], -1)
            ).view(batch_size, self.num_random)
            q2_curr = self.agent.critic.q2(
                torch.cat([
                    obs_exp.reshape(batch_size * self.num_random, -1),
                    curr_acts
                ], -1)
            ).view(batch_size, self.num_random)

            # CQL: logsumexp(concat[Q_rand, Q_curr]) - Q(s,a_dataset)
            cql1 = torch.logsumexp(
                torch.cat([q1_rand, q1_curr], dim=1), dim=1
            ).mean() - q1.mean()
            cql2 = torch.logsumexp(
                torch.cat([q2_rand, q2_curr], dim=1), dim=1
            ).mean() - q2.mean()

            critic_loss = td_loss + self.cql_weight * (cql1 + cql2)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()
            soft_update(self.agent.critic_target, self.agent.critic, self.tau)

            # ── Actor update ──────────────────────────────────────────────────
            pi, log_pi = self.agent.actor(obs)
            q_pi       = self.agent.critic(obs, pi)
            actor_loss = (alpha * log_pi - q_pi).mean()

            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()

            # ── Alpha (entropy) update ────────────────────────────────────────
            alpha_loss = -(
                self.agent.log_alpha * (log_pi + self.target_entropy).detach()
            ).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            if step % log_every == 0:
                elapsed = time.time() - t0
                self._log({
                    "td_loss": td_loss.item(),
                    "cql_penalty": (cql1 + cql2).item(),
                    "actor_loss": actor_loss.item(),
                    "alpha": alpha.item(),
                }, step)
                print(
                    f"  [CQL {step:>7d}] "
                    f"td={td_loss.item():.4f}  cql={cql1.item()+cql2.item():.4f}  "
                    f"α={alpha.item():.3f}  ({log_every/elapsed:.0f} it/s)"
                )
                t0 = time.time()

            if env is not None and step % eval_every == 0:
                _, score = evaluate_policy_baseline(
                    self.agent.actor, env, obs_mean, obs_std,
                    n_episodes=10, device=self.device,
                    env_name=getattr(self.config, 'env_name', ''),
                    ref_scores=getattr(self.config, 'D4RL_REF_SCORES', {}),
                )
                print(f"  [CQL eval {step}] normalized_score={score:.2f}")
                self._log({"eval_score": score}, step)
                self.maybe_save(score, {
                    "actor":  self.agent.actor.state_dict(),
                    "critic": self.agent.critic.state_dict(),
                })

        self.close()
        print(f"[CQL] Done. Best score: {self.best_score:.2f}")
        return self.best_score
