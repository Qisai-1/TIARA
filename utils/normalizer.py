"""
Online running mean/std normalizer.
Fit on offline data first, then frozen during training (standard practice).
"""

from __future__ import annotations
import numpy as np
import torch


class RunningNormalizer:
    """
    Normalize arrays using pre-computed mean and std.
    Call .fit(data) once on the full offline dataset, then .normalize().
    """

    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std:  Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, data: np.ndarray):
        """data shape: (N,) or (N, dim)"""
        self.mean = data.mean(axis=0)
        self.std  = data.std(axis=0) + self.eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x
        return (x - self.mean) / self.std

    def denormalize(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None:
            return x
        return x * self.std + self.mean

    def normalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        mean = torch.from_numpy(self.mean).to(x.device, dtype=x.dtype)
        std  = torch.from_numpy(self.std).to(x.device, dtype=x.dtype)
        return (x - mean) / std

    def denormalize_torch(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return x
        mean = torch.from_numpy(self.mean).to(x.device, dtype=x.dtype)
        std  = torch.from_numpy(self.std).to(x.device, dtype=x.dtype)
        return x * std + mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, d):
        self.mean = d["mean"]
        self.std  = d["std"]


def build_normalizers(data: dict) -> dict:
    """
    Fit normalizers for observations, actions, and rewards from offline data.
    Rewards are normalized as a scalar (single dimension).
    """
    obs_norm = RunningNormalizer()
    act_norm = RunningNormalizer()
    rew_norm = RunningNormalizer()

    obs_norm.fit(data["observations"])
    act_norm.fit(data["actions"])
    rew_norm.fit(data["rewards"].reshape(-1, 1))

    print(f"[Normalizer] obs  mean={obs_norm.mean.mean():.3f}  std={obs_norm.std.mean():.3f}")
    print(f"[Normalizer] act  mean={act_norm.mean.mean():.3f}  std={act_norm.std.mean():.3f}")
    print(f"[Normalizer] rew  mean={rew_norm.mean.item():.3f}  std={rew_norm.std.item():.3f}")

    return {"obs": obs_norm, "act": act_norm, "rew": rew_norm}
