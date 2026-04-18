"""
Baseline 1: Behavioral Cloning (BC)
====================================
The simplest offline RL baseline — supervised imitation of dataset actions.
No Q-learning, no conservatism. Pure regression: π(s) → a.

Why include: establishes the floor. TabRL must beat pure BC to justify
the Q-learning and ICL machinery. NeurIPS reviewers always check this.

Architecture: MLP actor, MSE loss (deterministic) or NLL (stochastic).
We use deterministic for fair comparison with TD3+BC.

Reference: Pomerleau (1989), re-used in every offline RL paper.
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from baselines.shared.networks import mlp, soft_update, compute_normalized_score
from baselines.shared.trainer import FlatReplayBuffer, evaluate_policy_baseline, BaseTrainer


class BCAgent(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        dims = [hidden_dim] * n_layers
        self.net = mlp(obs_dim, dims, act_dim,
                       output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class BCTrainer(BaseTrainer):
    """
    Trains BC agent via supervised regression on (s, a) pairs.
    No Q-function, no conservatism.
    """

    def __init__(self, obs_dim: int, act_dim: int, config, save_dir: str, device: str):
        super().__init__("bc", config, save_dir, device)
        self.actor     = BCAgent(obs_dim, act_dim).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)
        self.loss_fn   = nn.MSELoss()

    def train(
        self,
        data: Dict[str, np.ndarray],
        env=None,
        normalizers: dict = None,
        n_steps: int = 1_000_000,
        batch_size: int = 256,
        eval_every: int = 5000,
        log_every: int  = 1000,
    ) -> float:
        obs  = torch.from_numpy(data["observations"]).float()
        acts = torch.from_numpy(data["actions"]).float()

        dataset    = TensorDataset(obs, acts)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        data_iter  = iter(dataloader)

        obs_mean = data["observations"].mean(0)
        obs_std  = data["observations"].std(0) + 1e-8

        print(f"\n[BC] Training for {n_steps:,} steps ...")
        t0   = time.time()
        step = 0

        while step < n_steps:
            try:
                batch_obs, batch_acts = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch_obs, batch_acts = next(data_iter)

            batch_obs  = batch_obs.to(self.device)
            batch_acts = batch_acts.to(self.device)

            pred_acts = self.actor(batch_obs)
            loss      = self.loss_fn(pred_acts, batch_acts)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            step += 1

            if step % log_every == 0:
                elapsed = time.time() - t0
                self._log({"loss": loss.item()}, step)
                print(f"  [BC {step:>7d}] loss={loss.item():.4f}  ({log_every/elapsed:.0f} it/s)")
                t0 = time.time()

            if env is not None and step % eval_every == 0:
                _, score = evaluate_policy_baseline(
                    self.actor, env, obs_mean, obs_std,
                    n_episodes=10, device=self.device,
                    env_name=getattr(self.config, 'env_name', ''),
                    ref_scores=getattr(self.config, 'D4RL_REF_SCORES', {}),
                )
                print(f"  [BC eval {step}] normalized_score={score:.2f}")
                self._log({"eval_score": score}, step)
                self.maybe_save(score, self.actor.state_dict())

        self.close()
        print(f"[BC] Done. Best score: {self.best_score:.2f}")
        return self.best_score
