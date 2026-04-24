"""
Multi-Environment D4RL Dataset Loader
======================================
Loads multiple D4RL environments simultaneously and samples batches
with per-environment normalization.

Key design decisions:
  - Each environment has its OWN normalizer (obs/act dims differ across envs)
  - Each batch comes from ONE environment only (consistent column semantics)
  - Environments are sampled uniformly per batch by default
  - obs_dim and act_dim are padded to the MAX across all environments so the
    backbone sees a fixed-width table (padding with zeros)
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
import os


# ── Single environment raw loader ─────────────────────────────────────────────

def load_d4rl_dataset(minari_dataset_id: str) -> Dict[str, np.ndarray]:
    """Load one D4RL environment from Minari into flat numpy arrays."""
    try:
        import minari
    except ImportError:
        raise ImportError("pip install minari")

    print(f"[D4RL] Loading {minari_dataset_id} ...")

    try:
        dataset = minari.load_dataset(minari_dataset_id, download=True)
    except Exception as e:
        raise RuntimeError(
            f"Could not load '{minari_dataset_id}'.\n"
            f"Run: python -c \"import minari; "
            f"minari.download_dataset('{minari_dataset_id}')\"\n"
            f"Original error: {e}"
        )

    obs_list, act_list, rew_list, next_obs_list, term_list = [], [], [], [], []
    for episode in dataset.iterate_episodes():
        obs   = episode.observations
        acts  = episode.actions
        rews  = episode.rewards
        terms = episode.terminations
        obs_list.append(obs[:-1])
        next_obs_list.append(obs[1:])
        act_list.append(acts)
        rew_list.append(rews)
        term_list.append(terms)

    data = {
        "observations":      np.concatenate(obs_list).astype(np.float32),
        "actions":           np.concatenate(act_list).astype(np.float32),
        "rewards":           np.concatenate(rew_list).astype(np.float32),
        "next_observations": np.concatenate(next_obs_list).astype(np.float32),
        "terminals":         np.concatenate(term_list).astype(bool),
    }
    N = data["observations"].shape[0]
    print(f"[D4RL]   {N:>10,} transitions | "
          f"obs={data['observations'].shape[1]}  "
          f"act={data['actions'].shape[1]}")
    return data


# ── Per-environment normalizer ────────────────────────────────────────────────

class EnvNormalizer:
    """
    Fit-once, use-everywhere normalizer for one environment.
    Stores mean/std for observations, actions, and rewards separately.
    """

    def __init__(self, data: Dict[str, np.ndarray], eps: float = 1e-8):
        self.obs_mean = data["observations"].mean(0).astype(np.float32)
        self.obs_std  = (data["observations"].std(0) + eps).astype(np.float32)
        self.act_mean = data["actions"].mean(0).astype(np.float32)
        self.act_std  = (data["actions"].std(0) + eps).astype(np.float32)
        self.rew_mean = float(data["rewards"].mean())
        self.rew_std  = float(data["rewards"].std()) + eps
        self.obs_dim  = data["observations"].shape[1]
        self.act_dim  = data["actions"].shape[1]

    # ── numpy interface ───────────────────────────────────────────────────────
    def norm_obs(self, x: np.ndarray) -> np.ndarray:
        return (x - self.obs_mean) / self.obs_std

    def norm_act(self, x: np.ndarray) -> np.ndarray:
        return (x - self.act_mean) / self.act_std

    def norm_rew(self, r: np.ndarray) -> np.ndarray:
        return (r - self.rew_mean) / self.rew_std

    def denorm_act(self, x: np.ndarray) -> np.ndarray:
        return x * self.act_std + self.act_mean

    # ── torch interface (for inference) ───────────────────────────────────────
    def norm_obs_torch(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.from_numpy(self.obs_mean).to(x.device, x.dtype)
        s = torch.from_numpy(self.obs_std).to(x.device, x.dtype)
        return (x - m) / s

    def norm_act_torch(self, x: torch.Tensor) -> torch.Tensor:
        m = torch.from_numpy(self.act_mean).to(x.device, x.dtype)
        s = torch.from_numpy(self.act_std).to(x.device, x.dtype)
        return (x - m) / s

    def denorm_act_torch(self, x: torch.Tensor) -> torch.Tensor:
        s = torch.from_numpy(self.act_std).to(x.device, x.dtype)
        m = torch.from_numpy(self.act_mean).to(x.device, x.dtype)
        return x * s + m


# ── Single-environment ICL dataset ───────────────────────────────────────────

class ICLEnvDataset(Dataset):
    """
    ICL dataset for ONE environment.
    Samples (context_table, query_row) pairs.
    Pads obs/act to max_obs_dim / max_act_dim for cross-environment compatibility.
    """

    def __init__(
        self,
        data:        Dict[str, np.ndarray],
        normalizer:  EnvNormalizer,
        context_len: int,
        max_obs_dim: int,
        max_act_dim: int,
        gamma:       float = 0.99,
    ):
        # Normalize everything up front
        self.obs      = normalizer.norm_obs(data["observations"])
        self.acts     = normalizer.norm_act(data["actions"])
        self.rews     = data["rewards"].astype(np.float32)   # raw for RTG
        self.next_obs = normalizer.norm_obs(data["next_observations"])
        self.terminals= data["terminals"].astype(np.float32)

        self.context_len = context_len
        self.obs_dim     = normalizer.obs_dim
        self.act_dim     = normalizer.act_dim
        self.max_obs_dim = max_obs_dim
        self.max_act_dim = max_act_dim

        # ── Compute Return-To-Go (RTG) for every transition ───────────────────
        # RTG[t] = sum_{k=t}^{T} gamma^(k-t) * r[k]
        # This is what the backbone uses as context_y — it directly encodes
        # "how good is the future from this transition" which aligns with
        # Q-value estimation (TabPFN's ICL task: predict label from context)
        self.rtg = self._compute_rtg(data["rewards"], data["terminals"], gamma)

        # Normalise RTG to zero mean unit std for stable training
        rtg_mean = self.rtg.mean()
        rtg_std  = self.rtg.std() + 1e-8
        self.rtg = (self.rtg - rtg_mean) / rtg_std

        # Also keep normalised raw reward for TD target computation
        self.rews_norm = normalizer.norm_rew(data["rewards"])

        self.N = len(self.obs)
        # Valid query indices — need context_len prior transitions
        self.valid_idx = np.arange(context_len, self.N)

    @staticmethod
    def _compute_rtg(
        rewards:   np.ndarray,   # (N,)
        terminals: np.ndarray,   # (N,) bool
        gamma:     float = 0.99,
    ) -> np.ndarray:
        """
        Compute discounted return-to-go for each transition.
        Resets at episode boundaries (terminals).
        Returns normalised RTG array of shape (N,).
        """
        N   = len(rewards)
        rtg = np.zeros(N, dtype=np.float32)
        running = 0.0
        for t in reversed(range(N)):
            if terminals[t]:
                running = 0.0      # reset at episode end
            running   = rewards[t] + gamma * running
            rtg[t]    = running
        return rtg

    def __len__(self) -> int:
        return len(self.valid_idx)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        q       = self.valid_idx[idx]
        ctx_idx = np.arange(q - self.context_len, q)

        # Context window
        ctx_obs = self._pad2d(self.obs[ctx_idx],  self.max_obs_dim)  # (L, max_obs)
        ctx_act = self._pad2d(self.acts[ctx_idx], self.max_act_dim)  # (L, max_act)
        ctx_rtg = self.rtg[ctx_idx]                                   # (L,) RTG as label

        # Context table: concatenate obs + act columns
        # context_y = RTG (return-to-go) — aligns ICL task with Q-estimation
        context_X = np.concatenate(
            [ctx_obs, ctx_act], axis=-1
        ).astype(np.float32)                                          # (L, max_obs+max_act)

        # Query row
        q_obs  = self._pad1d(self.obs[q],       self.max_obs_dim)
        q_act  = self._pad1d(self.acts[q],      self.max_act_dim)
        q_next = self._pad1d(self.next_obs[q],  self.max_obs_dim)

        return {
            "context_X": torch.from_numpy(context_X),
            "context_y": torch.from_numpy(ctx_rtg.astype(np.float32)),  # RTG not raw r
            "query_obs": torch.from_numpy(q_obs.astype(np.float32)),
            "query_act": torch.from_numpy(q_act.astype(np.float32)),
            "query_rew": torch.tensor(float(self.rews_norm[q]), dtype=torch.float32),
            "next_obs":  torch.from_numpy(q_next.astype(np.float32)),
            "terminal":  torch.tensor(float(self.terminals[q]), dtype=torch.float32),
        }

    def _pad2d(self, x: np.ndarray, target: int) -> np.ndarray:
        if x.shape[1] == target:
            return x
        pad = np.zeros((x.shape[0], target - x.shape[1]), dtype=x.dtype)
        return np.concatenate([x, pad], axis=1)

    def _pad1d(self, x: np.ndarray, target: int) -> np.ndarray:
        if len(x) == target:
            return x
        return np.concatenate([x, np.zeros(target - len(x), dtype=x.dtype)])


# ── Multi-environment dataset ─────────────────────────────────────────────────

class MultiEnvDataset(Dataset):
    """
    Combines N single-environment ICL datasets into one.

    Each __getitem__ picks one environment uniformly at random,
    then picks a random (context, query) pair from it.
    All items have the same tensor shapes (padded to max dims).

    Why uniform sampling over proportional:
      - MuJoCo datasets all have ~1M transitions
      - Uniform = each environment contributes equally to each gradient step
      - This prevents halfcheetah (largest) from dominating
    """

    def __init__(
        self,
        env_datasets: List[ICLEnvDataset],
        env_names:    List[str],
    ):
        self.env_datasets = env_datasets
        self.env_names    = env_names
        self.n_envs       = len(env_datasets)
        self._total       = sum(len(d) for d in env_datasets)

        print(f"\n[MultiEnvDataset] {self.n_envs} environments loaded:")
        for name, ds in zip(env_names, env_datasets):
            print(f"  {name:<40}  {len(ds):>8,} samples")
        print(f"  {'Total':<40}  {self._total:>8,} samples\n")

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, _idx: int) -> Dict[str, torch.Tensor]:
        # Ignore _idx — always sample uniformly across environments
        env_id   = np.random.randint(self.n_envs)
        ds       = self.env_datasets[env_id]
        item_idx = np.random.randint(len(ds))
        item     = ds[item_idx]
        item["env_id"] = torch.tensor(env_id, dtype=torch.long)
        return item


# ── Environments to use ───────────────────────────────────────────────────────

# All 9 standard locomotion environments for pretraining
ALL_PRETRAIN_ENVS = [
    "mujoco/hopper/medium-v0",
    "mujoco/hopper/simple-v0",
    "mujoco/hopper/expert-v0",
    "mujoco/halfcheetah/medium-v0",
    "mujoco/halfcheetah/simple-v0",
    "mujoco/halfcheetah/expert-v0",
    "mujoco/walker2d/medium-v0",
    "mujoco/walker2d/simple-v0",
    "mujoco/walker2d/expert-v0",
]

# Short names for logging
ENV_SHORT_NAMES = {
    "mujoco/hopper/medium-v0":             "hopper-medium",
    "mujoco/hopper/simple-v0":      "hopper-medium-replay",
    "mujoco/hopper/expert-v0":      "hopper-medium-expert",
    "mujoco/halfcheetah/medium-v0":        "halfcheetah-medium",
    "mujoco/halfcheetah/simple-v0": "halfcheetah-medium-replay",
    "mujoco/halfcheetah/expert-v0": "halfcheetah-medium-expert",
    "mujoco/walker2d/medium-v0":           "walker2d-medium",
    "mujoco/walker2d/simple-v0":    "walker2d-medium-replay",
    "mujoco/walker2d/expert-v0":    "walker2d-medium-expert",
}


# ── Main factory function ─────────────────────────────────────────────────────

def build_multi_env_dataloader(
    minari_ids:  List[str],
    context_len: int,
    batch_size:  int = 256,
    num_workers: int = 4,
) -> Tuple[DataLoader, List[EnvNormalizer], List[str], int, int]:
    """
    Load multiple D4RL environments and return a combined DataLoader.

    Returns:
        dataloader   : DataLoader yielding multi-env batches
        normalizers  : list of EnvNormalizer, one per environment
        env_names    : short name for each environment
        max_obs_dim  : padded observation width used in context table
        max_act_dim  : padded action width used in context table
    """
    print(f"\n{'='*60}")
    print(f"  Loading {len(minari_ids)} environments for multi-env pretraining")
    print(f"{'='*60}")

    # Step 1: load raw data + fit normalizers
    all_data    = []
    normalizers = []
    env_names   = []

    for mid in minari_ids:
        data = load_d4rl_dataset(mid)
        norm = EnvNormalizer(data)
        all_data.append(data)
        normalizers.append(norm)
        env_names.append(ENV_SHORT_NAMES.get(mid, mid))

    # Step 2: find max dims for padding
    max_obs_dim = max(n.obs_dim for n in normalizers)
    max_act_dim = max(n.act_dim for n in normalizers)
    feature_dim = max_obs_dim + max_act_dim

    print(f"\n[MultiEnv] Dimensions across environments:")
    for name, norm in zip(env_names, normalizers):
        pad_o = max_obs_dim - norm.obs_dim
        pad_a = max_act_dim - norm.act_dim
        print(f"  {name:<35}  obs={norm.obs_dim}(+{pad_o})  act={norm.act_dim}(+{pad_a})")
    print(f"\n  Padded context table width: {feature_dim} columns")

    # Step 3: build per-environment ICL datasets
    env_datasets = []
    for data, norm, name in zip(all_data, normalizers, env_names):
        ds = ICLEnvDataset(
            data        = data,
            normalizer  = norm,
            context_len = context_len,
            max_obs_dim = max_obs_dim,
            max_act_dim = max_act_dim,
            gamma       = 0.99,
        )
        env_datasets.append(ds)
        print(f"  [Dataset] {name:<35}  {len(ds):>8,} valid samples")

    # Step 4: combine
    multi_dataset = MultiEnvDataset(env_datasets, env_names)

    # Step 5: DataLoader
    n_workers  = min(num_workers, os.cpu_count() or 1)
    dataloader = DataLoader(
        multi_dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = n_workers,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = n_workers > 0,
    )

    print(f"\n[MultiEnv] DataLoader ready — {len(dataloader):,} batches per epoch\n")
    return dataloader, normalizers, env_names, max_obs_dim, max_act_dim


# ── Legacy single-env helpers (kept for smoke tests) ─────────────────────────

class ICLTransitionDataset(Dataset):
    """Kept for backward compatibility with smoke tests."""

    def __init__(self, data, context_len=64, n_candidates=16,
                 obs_normalizer=None, act_normalizer=None, rew_normalizer=None):
        obs  = data["observations"]
        acts = data["actions"]
        rews = data["rewards"]
        if obs_normalizer:
            obs  = obs_normalizer.normalize(obs)
        if act_normalizer:
            acts = act_normalizer.normalize(acts)
        if rew_normalizer:
            rews = rew_normalizer.normalize(rews.reshape(-1,1)).reshape(-1)
        self.obs, self.acts, self.rews = obs, acts, rews
        self.next_obs  = data["next_observations"]
        self.terminals = data["terminals"]
        self.context_len = context_len
        self.valid_idx = np.arange(context_len, len(obs))
        self.rtg = ICLEnvDataset._compute_rtg(data["rewards"], data["terminals"])

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, idx):
        q   = self.valid_idx[idx]
        ctx = np.arange(q - self.context_len, q)
        X   = np.concatenate([self.obs[ctx], self.acts[ctx]], -1).astype(np.float32)
        return {
            "context_X": torch.from_numpy(X),
            "context_y": torch.from_numpy(self.rtg[ctx].astype(np.float32)),   # RTG not raw reward
            "query_obs": torch.from_numpy(self.obs[q].astype(np.float32)),
            "query_act": torch.from_numpy(self.acts[q].astype(np.float32)),
            "query_rew": torch.tensor(float(self.rews[q]), dtype=torch.float32),
            "next_obs":  torch.from_numpy(self.next_obs[q].astype(np.float32)),
            "terminal":  torch.tensor(float(self.terminals[q]), dtype=torch.float32),
        }


def make_pretrain_dataloader(data, config, obs_normalizer=None,
                              act_normalizer=None, rew_normalizer=None,
                              num_workers=4):
    """Legacy single-env loader — smoke tests only."""
    ds = ICLTransitionDataset(data, config.context_len, config.n_candidates,
                               obs_normalizer, act_normalizer, rew_normalizer)
    return DataLoader(ds, batch_size=config.batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True)