"""
Online TD Training — Phase 2.

Fine-tunes the agent by interacting with the live environment.
Collects new transitions, adds to replay buffer, runs TD updates.
"""

from __future__ import annotations

import os
import time
import torch
import numpy as np
from typing import Optional

from ..models.tabrl_agent import TabRLAgent
from ..data.replay_buffer import ReplayBuffer
from ..utils.logger import Logger
from ..evaluation.evaluator import evaluate_policy


def online_train(
    agent:      TabRLAgent,
    env,                        # gymnasium env
    replay_buffer: ReplayBuffer,
    config,
    logger:     Logger,
    save_dir:   str,
    normalizers: dict,
) -> None:
    """
    Online RL training loop.

    1. Collect transition using current policy
    2. Add to replay buffer
    3. After warmup_steps: sample batch, run TD update
    4. Periodically evaluate and checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    optimizer = _build_online_optimizer(agent, config)

    obs, _ = env.reset()
    episode_reward = 0.0
    episode_step   = 0
    episode_count  = 0
    best_score     = -float("inf")
    best_path      = os.path.join(save_dir, "online_best.pt")

    print(f"\n{'='*60}")
    print(f"  Online Fine-tuning: {config.online_steps:,} steps")
    print(f"  Warmup: {config.warmup_steps} steps before TD updates")
    print(f"{'='*60}\n")

    t0 = time.time()

    for step in range(1, config.online_steps + 1):

        # ── Collect transition ────────────────────────────────────────────────
        # Normalize obs
        obs_norm = normalizers["obs"].normalize(obs.astype(np.float32))

        # Build context from buffer
        if len(replay_buffer) >= config.context_len:
            ctx = replay_buffer.sample_context(len(replay_buffer) - 1)
            ctx_X = torch.from_numpy(ctx["context_X"]).unsqueeze(0).to(agent.device)  # (1, L, f)
            ctx_y = torch.from_numpy(ctx["context_y"]).unsqueeze(0).to(agent.device)  # (1, L)
        else:
            # Not enough data yet: use zeros as context
            L = config.context_len
            ctx_X = torch.zeros(1, L, agent.feature_dim, device=agent.device)
            ctx_y = torch.zeros(1, L, device=agent.device)

        obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(agent.device)

        # Select action
        if step < config.warmup_steps:
            action = env.action_space.sample()
        else:
            action, act_info = agent.select_action(ctx_X, ctx_y, obs_t, deterministic=False)

        # Step environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Normalize for storage
        next_obs_norm = normalizers["obs"].normalize(next_obs.astype(np.float32))
        act_norm      = normalizers["act"].normalize(action.astype(np.float32))
        rew_norm      = float(normalizers["rew"].normalize(np.array([[reward]]))[0, 0])

        replay_buffer.add(obs_norm, act_norm, rew_norm, next_obs_norm, float(terminated))
        episode_reward += reward
        episode_step   += 1

        obs = next_obs
        if done:
            obs, _ = env.reset()
            logger.log({
                "online/episode_reward": episode_reward,
                "online/episode_steps":  episode_step,
                "online/episode":        episode_count,
            }, step=step)
            episode_reward = 0.0
            episode_step   = 0
            episode_count += 1

        # ── TD Update ─────────────────────────────────────────────────────────
        if step >= config.warmup_steps and len(replay_buffer) >= config.context_len + config.batch_size:
            batch = replay_buffer.sample_batch(config.batch_size)
            batch = {k: v.to(agent.device) for k, v in batch.items()}

            optimizer.zero_grad(set_to_none=True)
            td_loss, td_info = agent.td_step(batch)
            td_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), config.grad_clip)
            optimizer.step()
            agent.soft_update_target()

            if step % config.log_every == 0:
                elapsed = time.time() - t0
                logger.log({**td_info, "online/step": step}, step=step)
                print(
                    f"[online {step:>6d}] "
                    f"td={td_info['value/td_loss']:.4f}  "
                    f"Q={td_info['value/q_pred']:.3f}  "
                    f"buf={len(replay_buffer):,}  "
                    f"({config.log_every / elapsed:.1f} it/s)"
                )
                t0 = time.time()

        # ── Evaluation ────────────────────────────────────────────────────────
        if step % config.eval_every == 0:
            score = evaluate_policy(
                agent, env, normalizers, config,
                n_episodes=config.eval_episodes
            )
            logger.log({"online/normalized_score": score}, step=step)
            print(f"\n[EVAL step={step}] Normalized score: {score:.2f}\n")

            if score > best_score:
                best_score = score
                agent.save(best_path)
                print(f"  → New best! Saved to {best_path}")

    print(f"\n[Online] Done. Best normalized score: {best_score:.2f}")


def _build_online_optimizer(agent: TabRLAgent, config):
    """Online phase: typically only train heads, backbone frozen or very slow."""
    head_params = (
        list(agent.proposal_head.parameters()) +
        list(agent.value_head.parameters())
    )
    if config.freeze_backbone:
        return torch.optim.AdamW(head_params, lr=config.head_lr, weight_decay=1e-4)
    else:
        backbone_params = [p for p in agent.backbone.parameters() if p.requires_grad]
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": config.backbone_lr * 0.1},  # even slower online
            {"params": head_params,     "lr": config.head_lr},
        ], weight_decay=1e-4)
