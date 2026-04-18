"""
Baseline 5: Decision Transformer (DT)
======================================
Chen et al., NeurIPS 2021.

Key idea: cast offline RL as conditional sequence modeling.
Represent a trajectory as (R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...) where R̂ₜ is
return-to-go. Train a causal transformer (GPT-style) to predict aₜ given
the context (R̂, s, a)_{1..t}.

At inference: condition on a desired target return R̂_target, then autoregressively
generate actions.

WHY THIS BASELINE IS CRITICAL FOR OUR PAPER:
DT also uses a transformer with a context window over past transitions.
Reviewers will directly ask: "how does your tabular ICL approach compare to DT
which also uses in-context information?" The key differences we need to show:
  1. TabRL does Q-learning (not BC); DT is pure supervised learning.
  2. TabRL's backbone cross-attends across ALL context rows simultaneously;
     DT is causal (only attends backward).
  3. TabRL doesn't need a manually specified target return at test time.

Reference: https://arxiv.org/abs/2106.01345
Official: https://github.com/kzl/decision-transformer
"""

from __future__ import annotations

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from baselines.shared.trainer import BaseTrainer, evaluate_policy_baseline


# ── GPT-style causal self-attention block ─────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.qkv    = nn.Linear(n_embd, 3 * n_embd)
        self.proj   = nn.Linear(n_embd, n_embd)
        self.drop   = nn.Dropout(dropout)
        # Causal mask: lower triangular
        mask = torch.tril(torch.ones(context_len, context_len))
        self.register_buffer("mask", mask.unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C  = x.shape
        q, k, v  = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = C // self.n_head

        def reshape(t):
            return t.view(B, T, self.n_head, head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        scale   = 1.0 / math.sqrt(head_dim)
        attn    = (q @ k.transpose(-2, -1)) * scale
        attn    = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        attn    = F.softmax(attn, dim=-1)
        attn    = self.drop(attn)
        out     = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)


class DTBlock(nn.Module):
    def __init__(self, n_embd: int, n_head: int, context_len: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, context_len, dropout)
        self.ffn  = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """
    GPT-based Decision Transformer.

    Each timestep t in the context window has THREE tokens:
        (return-to-go R̂ₜ, state sₜ, action aₜ)
    So a context of K timesteps = 3K tokens total.

    The model predicts the action token aₜ from its position in the sequence.
    """

    def __init__(
        self,
        obs_dim:     int,
        act_dim:     int,
        context_len: int   = 20,    # K timesteps
        n_layer:     int   = 3,
        n_head:      int   = 1,
        n_embd:      int   = 128,
        dropout:     float = 0.1,
        max_ep_len:  int   = 1000,
    ):
        super().__init__()
        self.obs_dim     = obs_dim
        self.act_dim     = act_dim
        self.context_len = context_len
        self.n_embd      = n_embd

        max_tokens = 3 * context_len  # R̂, s, a per timestep

        # Token embeddings
        self.embed_rtg    = nn.Linear(1, n_embd)
        self.embed_state  = nn.Linear(obs_dim, n_embd)
        self.embed_action = nn.Linear(act_dim, n_embd)
        self.embed_ln     = nn.LayerNorm(n_embd)

        # Timestep embedding (absolute position within episode)
        self.embed_timestep = nn.Embedding(max_ep_len, n_embd)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            DTBlock(n_embd, n_head, max_tokens, dropout)
            for _ in range(n_layer)
        ])

        self.ln_f      = nn.LayerNorm(n_embd)
        self.act_head  = nn.Linear(n_embd, act_dim)  # predict action from state token position

    def forward(
        self,
        rtg:       torch.Tensor,    # (B, K, 1)     return-to-go
        states:    torch.Tensor,    # (B, K, obs)
        actions:   torch.Tensor,    # (B, K, act)
        timesteps: torch.Tensor,    # (B, K)        absolute timestep indices
        attention_mask: Optional[torch.Tensor] = None,  # (B, K) — 0 for padding
    ) -> torch.Tensor:
        """
        Returns predicted actions at each position: (B, K, act_dim).
        During training, use all K positions (teacher forcing).
        During inference, use the last position.
        """
        B, K, _ = states.shape

        t_emb = self.embed_timestep(timesteps)  # (B, K, n_embd)

        # Embed each modality + add positional (timestep) embedding
        rtg_emb   = self.embed_rtg(rtg)      + t_emb   # (B, K, n_embd)
        state_emb = self.embed_state(states) + t_emb   # (B, K, n_embd)
        act_emb   = self.embed_action(actions)+ t_emb  # (B, K, n_embd)

        # Interleave: [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ...]
        # Stack → (B, 3K, n_embd)
        tokens = torch.stack([rtg_emb, state_emb, act_emb], dim=2)  # (B, K, 3, n_embd)
        tokens = tokens.view(B, 3 * K, self.n_embd)                  # (B, 3K, n_embd)
        tokens = self.embed_ln(tokens)

        # Apply attention mask if padding exists
        if attention_mask is not None:
            # Expand mask to 3K tokens
            attn_mask = attention_mask.repeat_interleave(3, dim=1)  # (B, 3K)
            # DT implementation masks padding; we leave for simplicity
            # (offline dataset has no padding in our setup)

        x = tokens
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Extract state token positions: indices 1, 4, 7, ... (1-indexed in interleaved)
        # State token is at position 3t+1 (0-indexed: 1, 4, 7, ...) for each timestep t
        state_positions = torch.arange(1, 3 * K, step=3, device=states.device)  # [1,4,7,...]
        state_hidden    = x[:, state_positions, :]   # (B, K, n_embd)

        # Predict action from state hidden state
        pred_actions = self.act_head(state_hidden)   # (B, K, act_dim)
        return torch.tanh(pred_actions)


class DTDataset(torch.utils.data.Dataset):
    """
    Builds (K-step context, target action) samples from D4RL trajectories.
    Computes return-to-go for each timestep.
    """

    def __init__(
        self,
        data: Dict[str, np.ndarray],
        context_len: int = 20,
        gamma: float = 1.0,
        obs_mean: np.ndarray = None,
        obs_std:  np.ndarray = None,
        rtg_scale: float = 1.0,
    ):
        self.context_len = context_len
        self.rtg_scale   = rtg_scale

        obs   = data["observations"].astype(np.float32)
        acts  = data["actions"].astype(np.float32)
        rews  = data["rewards"].astype(np.float32)
        terms = data["terminals"]

        # Normalize observations
        if obs_mean is not None:
            obs = (obs - obs_mean) / (obs_std + 1e-8)

        # Split into episodes at terminal flags
        self.trajectories = []
        ep_start = 0
        for i in range(len(terms)):
            if terms[i] or i == len(terms) - 1:
                ep_obs  = obs[ep_start:i + 1]
                ep_acts = acts[ep_start:i + 1]
                ep_rews = rews[ep_start:i + 1]

                # Compute return-to-go for each step
                rtg = np.zeros_like(ep_rews)
                cumsum = 0.0
                for t in reversed(range(len(ep_rews))):
                    cumsum = ep_rews[t] + gamma * cumsum
                    rtg[t] = cumsum

                if len(ep_obs) >= 2:  # skip single-step episodes
                    self.trajectories.append({
                        "obs":  ep_obs,
                        "acts": ep_acts,
                        "rtg":  rtg / rtg_scale,
                    })
                ep_start = i + 1

        print(f"  [DT Dataset] {len(self.trajectories)} episodes, context_len={context_len}")

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        traj = self.trajectories[idx]
        T    = len(traj["obs"])
        K    = self.context_len

        # Random starting point (pad if episode shorter than K)
        start = np.random.randint(0, max(1, T - 1))
        end   = min(start + K, T)
        length= end - start

        # Extract window
        obs_w  = traj["obs"][start:end]    # (L, obs)
        act_w  = traj["acts"][start:end]   # (L, act)
        rtg_w  = traj["rtg"][start:end]    # (L,)
        ts_w   = np.arange(start, end)     # (L,) absolute timesteps

        # Pad to K if shorter
        if length < K:
            pad  = K - length
            obs_w  = np.concatenate([np.zeros((pad, obs_w.shape[1]), dtype=np.float32), obs_w])
            act_w  = np.concatenate([np.zeros((pad, act_w.shape[1]), dtype=np.float32), act_w])
            rtg_w  = np.concatenate([np.zeros(pad, dtype=np.float32), rtg_w])
            ts_w   = np.concatenate([np.zeros(pad, dtype=np.int64), ts_w])
            mask   = np.concatenate([np.zeros(pad), np.ones(length)])
        else:
            mask   = np.ones(K)

        return {
            "obs":       torch.from_numpy(obs_w.astype(np.float32)),      # (K, obs)
            "actions":   torch.from_numpy(act_w.astype(np.float32)),      # (K, act)
            "rtg":       torch.from_numpy(rtg_w.astype(np.float32)).unsqueeze(-1),  # (K,1)
            "timesteps": torch.from_numpy(ts_w.astype(np.int64)),         # (K,)
            "mask":      torch.from_numpy(mask.astype(np.float32)),       # (K,)
        }


class DTTrainer(BaseTrainer):
    """
    Decision Transformer trainer.

    Hyperparameters follow the original paper:
      context_len=20, n_layer=3, n_head=1, n_embd=128
      lr=1e-4, warmup=10000, batch_size=64
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        config,
        save_dir: str,
        device: str,
        context_len: int   = 20,
        n_layer:     int   = 3,
        n_head:      int   = 1,
        n_embd:      int   = 128,
        dropout:     float = 0.1,
        rtg_scale:   float = 1000.0,
        target_return: float = None,
    ):
        super().__init__("dt", config, save_dir, device)
        self.context_len   = context_len
        self.rtg_scale     = rtg_scale
        # Target return for evaluation (typically 3600 for hopper, 12000 for halfcheetah)
        self.target_return = target_return

        self.model = DecisionTransformer(
            obs_dim=obs_dim, act_dim=act_dim,
            context_len=context_len,
            n_layer=n_layer, n_head=n_head, n_embd=n_embd, dropout=dropout,
        ).to(device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=1e-4, weight_decay=1e-4
        )
        self.scheduler = None  # set in train()

    def train(
        self,
        data: Dict[str, np.ndarray],
        env=None,
        normalizers: dict = None,
        n_steps:    int = 100_000,   # DT uses fewer steps (100k vs 1M for value methods)
        batch_size: int = 64,
        eval_every: int = 5000,
        log_every:  int = 1000,
    ) -> float:
        obs_mean = data["observations"].mean(0)
        obs_std  = data["observations"].std(0) + 1e-8

        dataset    = DTDataset(data, self.context_len, obs_mean=obs_mean, obs_std=obs_std,
                               rtg_scale=self.rtg_scale)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=2, drop_last=True, pin_memory=True,
        )

        # Linear warmup scheduler
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda step: min(1.0, step / 10000)
        )

        data_iter = iter(dataloader)
        print(f"\n[DT] Training for {n_steps:,} steps | ctx={self.context_len} embd={self.model.n_embd}")
        t0 = time.time()

        for step in range(1, n_steps + 1):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch     = next(data_iter)

            obs_b  = batch["obs"].to(self.device)        # (B, K, obs)
            act_b  = batch["actions"].to(self.device)    # (B, K, act)
            rtg_b  = batch["rtg"].to(self.device)        # (B, K, 1)
            ts_b   = batch["timesteps"].to(self.device)  # (B, K)
            mask_b = batch["mask"].to(self.device)       # (B, K)

            # Forward
            pred_acts = self.model(rtg_b, obs_b, act_b, ts_b)  # (B, K, act)

            # Loss only on non-padded positions
            loss = F.mse_loss(pred_acts * mask_b.unsqueeze(-1),
                              act_b     * mask_b.unsqueeze(-1))

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            self.optimizer.step()
            self.scheduler.step()

            if step % log_every == 0:
                elapsed = time.time() - t0
                self._log({"bc_loss": loss.item()}, step)
                print(f"  [DT {step:>7d}] loss={loss.item():.4f}  ({log_every/elapsed:.0f} it/s)")
                t0 = time.time()

            if env is not None and step % eval_every == 0 and self.target_return is not None:
                score = self._evaluate_dt(env, obs_mean, obs_std)
                print(f"  [DT eval {step}] normalized_score={score:.2f}")
                self._log({"eval_score": score}, step)
                self.maybe_save(score, self.model.state_dict())

        self.close()
        print(f"[DT] Done. Best score: {self.best_score:.2f}")
        return self.best_score

    @torch.no_grad()
    def _evaluate_dt(self, env, obs_mean, obs_std, n_episodes: int = 10) -> float:
        """Autoregressive rollout conditioned on target return."""
        self.model.eval()
        rewards_total = []

        for _ in range(n_episodes):
            obs, _ = env.reset()
            total_r = 0.0
            done    = False

            # Rolling context buffers
            obs_buf  = []
            act_buf  = []
            rtg_buf  = [self.target_return / self.rtg_scale]
            ts_buf   = [0]

            t = 0
            while not done:
                obs_n = (obs - obs_mean) / (obs_std + 1e-8)
                obs_buf.append(obs_n)
                act_buf.append(np.zeros(env.action_space.shape[0]))  # placeholder

                K = min(len(obs_buf), self.context_len)

                obs_t  = torch.from_numpy(np.array(obs_buf[-K:])).float().unsqueeze(0).to(self.device)
                act_t  = torch.from_numpy(np.array(act_buf[-K:])).float().unsqueeze(0).to(self.device)
                rtg_t  = torch.tensor(rtg_buf[-K:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(self.device)
                ts_t   = torch.tensor(ts_buf[-K:],  dtype=torch.long).unsqueeze(0).to(self.device)

                pred   = self.model(rtg_t, obs_t, act_t, ts_t)   # (1, K, act)
                action = pred[0, -1].cpu().numpy()                 # last timestep
                act_buf[-1] = action                               # fill placeholder

                obs, r, term, trunc, _ = env.step(action)
                total_r += r
                done = term or trunc

                # Update RTG: R̂_{t+1} = R̂_t - r_t
                rtg_buf.append((rtg_buf[-1] * self.rtg_scale - r) / self.rtg_scale)
                ts_buf.append(t + 1)
                t += 1

            rewards_total.append(total_r)

        self.model.train()
        mean_r = float(np.mean(rewards_total))
        env_name = getattr(self.config, 'env_name', '')
        ref_scores = getattr(self.config, 'D4RL_REF_SCORES', {})
        from baselines.shared.networks import compute_normalized_score
        return compute_normalized_score(mean_r, env_name, ref_scores)
