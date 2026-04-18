"""
Policy evaluation: rollout agent in environment and compute normalized score.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Optional


def evaluate_policy(
    agent,
    env,
    normalizers: dict,
    config,
    n_episodes: int = 10,
) -> float:
    """
    Roll out the agent deterministically for n_episodes.
    Returns D4RL normalized score (0 = random, 100 = expert).
    """
    agent.eval()
    total_rewards = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        done = False
        step = 0

        # Fresh context buffer per episode (start empty, fill as we go)
        ctx_obs_list = []
        ctx_act_list = []
        ctx_rew_list = []

        while not done:
            obs_norm = normalizers["obs"].normalize(obs.astype(np.float32))

            # Build context from this episode's history (or zeros if empty)
            L = config.context_len
            if len(ctx_obs_list) >= 1:
                # Take last L transitions from this episode
                hist_obs = np.array(ctx_obs_list[-L:])        # (min(step,L), obs)
                hist_act = np.array(ctx_act_list[-L:])
                hist_rew = np.array(ctx_rew_list[-L:])

                # Pad to L if shorter
                if len(hist_obs) < L:
                    pad = L - len(hist_obs)
                    hist_obs = np.concatenate([np.zeros((pad, agent.obs_dim), dtype=np.float32), hist_obs])
                    hist_act = np.concatenate([np.zeros((pad, agent.act_dim), dtype=np.float32), hist_act])
                    hist_rew = np.concatenate([np.zeros(pad, dtype=np.float32), hist_rew])

                ctx_X_np = np.concatenate([hist_obs, hist_act], axis=-1)   # (L, obs+act)
                ctx_X = torch.from_numpy(ctx_X_np).unsqueeze(0).to(agent.device)
                ctx_y = torch.from_numpy(hist_rew).unsqueeze(0).to(agent.device)
            else:
                ctx_X = torch.zeros(1, L, agent.feature_dim, device=agent.device)
                ctx_y = torch.zeros(1, L, device=agent.device)

            obs_t = torch.from_numpy(obs_norm).unsqueeze(0).to(agent.device)

            with torch.no_grad():
                action, _ = agent.select_action(ctx_X, ctx_y, obs_t, deterministic=True)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record normalized transition for context
            act_norm = normalizers["act"].normalize(action.astype(np.float32))
            rew_norm = float(normalizers["rew"].normalize(np.array([[reward]]))[0, 0])
            ctx_obs_list.append(obs_norm)
            ctx_act_list.append(act_norm)
            ctx_rew_list.append(rew_norm)

            episode_reward += reward
            obs = next_obs
            step += 1

        total_rewards.append(episode_reward)

    agent.train()

    mean_reward = np.mean(total_rewards)
    std_reward  = np.std(total_rewards)

    # D4RL normalized score
    if config.env_name in config.D4RL_REF_SCORES:
        refs = config.D4RL_REF_SCORES[config.env_name]
        normalized = (mean_reward - refs["random"]) / (refs["expert"] - refs["random"]) * 100
    else:
        normalized = mean_reward

    print(f"  [Eval] {n_episodes} episodes | "
          f"reward={mean_reward:.1f}±{std_reward:.1f} | "
          f"normalized={normalized:.2f}")

    return float(normalized)
