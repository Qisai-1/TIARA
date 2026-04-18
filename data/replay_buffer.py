"""
Replay buffer for online RL phase.

Supports three context sampling strategies:
  - "recent"   : sliding window of the last N transitions (fast)
  - "random"   : uniform random sample from buffer
  - "priority" : TD-error prioritized sampling (encourages informative context)
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class ReplayBuffer:
    """
    Circular replay buffer storing (s, a, r, s', done) transitions.
    Also stores TD-errors for priority-based context sampling.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        max_size: int = 1_000_000,
        context_len: int = 64,
        context_sampling: str = "recent",
        device: str = "cpu",
    ):
        self.obs_dim  = obs_dim
        self.act_dim  = act_dim
        self.max_size = max_size
        self.context_len = context_len
        self.context_sampling = context_sampling
        self.device = device

        self.observations      = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.actions           = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rewards           = np.zeros((max_size,),         dtype=np.float32)
        self.next_observations = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.terminals         = np.zeros((max_size,),         dtype=np.float32)
        self.td_errors         = np.ones( (max_size,),         dtype=np.float32)

        self.ptr  = 0    # write pointer
        self.size = 0    # current number of stored transitions

    def add(
        self,
        obs: np.ndarray,
        act: np.ndarray,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
        td_error: float = 1.0,
    ):
        self.observations[self.ptr]      = obs
        self.actions[self.ptr]           = act
        self.rewards[self.ptr]           = rew
        self.next_observations[self.ptr] = next_obs
        self.terminals[self.ptr]         = float(done)
        self.td_errors[self.ptr]         = abs(td_error) + 1e-6

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(self, data: Dict[str, np.ndarray]):
        """Bulk-add an entire offline dataset (for preloading D4RL data)."""
        n = len(data["observations"])
        for i in range(n):
            self.add(
                data["observations"][i],
                data["actions"][i],
                data["rewards"][i],
                data["next_observations"][i],
                data["terminals"][i],
            )

    def update_td_errors(self, indices: np.ndarray, td_errors: np.ndarray):
        self.td_errors[indices] = np.abs(td_errors) + 1e-6

    def sample_context(self, query_idx: int) -> Dict[str, np.ndarray]:
        """
        Sample a context window of self.context_len transitions
        to serve as the ICL table rows for a given query index.

        Returns dict with keys: context_X (L, obs+act), context_y (L,)
        """
        L = self.context_len
        valid = min(self.size, self.max_size)

        if self.context_sampling == "recent":
            # Last L transitions before query_idx
            idxs = np.arange(
                max(0, query_idx - L), query_idx
            ) % self.max_size
            # Pad with random if not enough history
            if len(idxs) < L:
                pad = np.random.choice(valid, L - len(idxs), replace=True)
                idxs = np.concatenate([pad, idxs])

        elif self.context_sampling == "random":
            idxs = np.random.choice(valid, L, replace=False if valid >= L else True)

        elif self.context_sampling == "priority":
            probs = self.td_errors[:valid] / self.td_errors[:valid].sum()
            idxs  = np.random.choice(valid, L, replace=False if valid >= L else True, p=probs)

        else:
            raise ValueError(f"Unknown context_sampling: {self.context_sampling}")

        ctx_obs = self.observations[idxs]       # (L, obs_dim)
        ctx_act = self.actions[idxs]            # (L, act_dim)
        ctx_rew = self.rewards[idxs]            # (L,)

        context_X = np.concatenate([ctx_obs, ctx_act], axis=-1)  # (L, obs+act)
        return {
            "context_X": context_X.astype(np.float32),
            "context_y": ctx_rew.astype(np.float32),
            "context_idxs": idxs,
        }

    def sample_batch(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of (context, query) pairs for TD training.
        Each query has its own freshly sampled context window.
        """
        assert self.size >= self.context_len + 1, \
            f"Buffer too small: {self.size} < {self.context_len + 1}"

        # Sample query indices (must have at least context_len prior entries)
        valid = min(self.size, self.max_size)
        query_idxs = np.random.choice(
            np.arange(self.context_len, valid), batch_size, replace=True
        )

        # Build batch tensors
        batch_context_X = []
        batch_context_y = []

        for qi in query_idxs:
            ctx = self.sample_context(qi)
            batch_context_X.append(ctx["context_X"])
            batch_context_y.append(ctx["context_y"])

        return {
            "context_X":  torch.from_numpy(np.stack(batch_context_X)),  # (B, L, obs+act)
            "context_y":  torch.from_numpy(np.stack(batch_context_y)),  # (B, L)
            "query_obs":  torch.from_numpy(self.observations[query_idxs]),       # (B, obs)
            "query_act":  torch.from_numpy(self.actions[query_idxs]),            # (B, act)
            "query_rew":  torch.from_numpy(self.rewards[query_idxs]),            # (B,)
            "next_obs":   torch.from_numpy(self.next_observations[query_idxs]),  # (B, obs)
            "terminal":   torch.from_numpy(self.terminals[query_idxs]),          # (B,)
        }

    def __len__(self):
        return self.size
