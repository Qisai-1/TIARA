"""
Shared training utilities for all baselines:
  - ReplayBuffer (simple flat buffer for baselines — no context needed)
  - evaluate_policy (standard rollout)
  - BaseTrainer  (common fit/eval loop)
"""

from __future__ import annotations

import os
import time
import numpy as np
import torch
from typing import Dict, Tuple, Optional
from .networks import compute_normalized_score


# ── Simple replay buffer (no context window needed for baselines) ─────────────

class FlatReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, max_size: int = 1_000_000):
        self.obs      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions  = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards  = np.zeros((max_size,),          dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.terminals= np.zeros((max_size,),          dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, max_size

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = act
        self.rewards[self.ptr]  = rew
        self.next_obs[self.ptr] = next_obs
        self.terminals[self.ptr]= float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_d4rl(self, data: Dict[str, np.ndarray]):
        n = len(data["observations"])
        for i in range(n):
            self.add(
                data["observations"][i],
                data["actions"][i],
                data["rewards"][i],
                data["next_observations"][i],
                data["terminals"][i],
            )

    def sample(self, batch_size: int, device: str = "cpu") -> Dict[str, torch.Tensor]:
        idx = np.random.choice(self.size, batch_size, replace=False)
        return {
            "obs":       torch.from_numpy(self.obs[idx]).to(device),
            "actions":   torch.from_numpy(self.actions[idx]).to(device),
            "rewards":   torch.from_numpy(self.rewards[idx]).to(device),
            "next_obs":  torch.from_numpy(self.next_obs[idx]).to(device),
            "terminals": torch.from_numpy(self.terminals[idx]).to(device),
        }

    def __len__(self): return self.size


# ── Policy evaluation ─────────────────────────────────────────────────────────

def evaluate_policy_baseline(
    actor,
    env,
    obs_mean: np.ndarray,
    obs_std:  np.ndarray,
    n_episodes: int = 10,
    device: str = "cpu",
    env_name: str = "",
    ref_scores: dict = None,
) -> Tuple[float, float]:
    """
    Roll out actor (deterministic) for n_episodes.
    Returns (mean_reward, normalized_score).
    """
    actor.eval()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_r = 0.0
        done    = False
        while not done:
            obs_n  = (obs - obs_mean) / obs_std
            obs_t  = torch.from_numpy(obs_n.astype(np.float32)).unsqueeze(0).to(device)
            with torch.no_grad():
                if hasattr(actor, 'forward'):
                    action = actor(obs_t)
                    if isinstance(action, tuple):
                        action = action[0]
                    action = action[0].cpu().numpy()
                else:
                    action = actor(obs_t)[0].cpu().numpy()
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            done = term or trunc
        rewards.append(total_r)

    actor.train()
    mean_r = float(np.mean(rewards))
    norm_s = compute_normalized_score(mean_r, env_name, ref_scores or {})
    return mean_r, norm_s


# ── Base trainer ──────────────────────────────────────────────────────────────

class BaseTrainer:
    """Common scaffolding: train loop, logging, checkpointing."""

    def __init__(self, name: str, config, save_dir: str, device: str):
        self.name     = name
        self.config   = config
        self.save_dir = save_dir
        self.device   = device
        os.makedirs(save_dir, exist_ok=True)
        self.log_path = os.path.join(save_dir, f"{name}_log.csv")
        self._log_file = open(self.log_path, "w")
        self._header_written = False
        self.best_score = -float("inf")
        self.best_path  = os.path.join(save_dir, f"{name}_best.pt")

    def _log(self, metrics: dict, step: int):
        metrics["step"] = step
        if not self._header_written:
            self._log_file.write(",".join(metrics.keys()) + "\n")
            self._header_written = True
        self._log_file.write(",".join(str(v) for v in metrics.values()) + "\n")
        self._log_file.flush()

    def close(self):
        self._log_file.close()

    def maybe_save(self, score: float, state_dict: dict):
        if score > self.best_score:
            self.best_score = score
            torch.save(state_dict, self.best_path)
            return True
        return False
