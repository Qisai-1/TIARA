"""
TabRL — Main Training Entry Point
===================================
Multi-environment pretraining is the DEFAULT and primary mode.
Single-environment is only available via --single_env for debugging.

Usage:
    # ── RECOMMENDED: multi-environment pretraining (paper results) ──
    python tabrl/train.py --phase pretrain

    # ── DEBUG ONLY: single environment ──
    python tabrl/train.py --phase pretrain --single_env hopper-medium-v2

    # ── Evaluate a pretrained checkpoint on one environment ──
    python tabrl/train.py --phase eval \
        --checkpoint runs/multi_env_pretrain/.../pretrain_best.pt \
        --eval_env hopper-medium-v2
"""

import os
import sys
import argparse
import random
import numpy as np
import torch

# Insert the PARENT of the tabrl directory so that
# import TIARA.xxx works regardless of where you run from
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from TIARA.configs.base_config import TabRLConfig
from TIARA.data.d4rl_loader import (
    build_multi_env_dataloader,
    ALL_PRETRAIN_ENVS,
    load_d4rl_dataset,
    make_pretrain_dataloader,
)
from TIARA.models.tabrl_agent import TabRLAgent
from TIARA.training.pretrain import pretrain
from TIARA.utils.normalizer import build_normalizers
from TIARA.utils.logger import Logger


def parse_args():
    config = TabRLConfig()
    parser = argparse.ArgumentParser(description="TabRL training")

    # ── Mode ──────────────────────────────────────────────────────────────────
    parser.add_argument("--phase", type=str, default="pretrain",
                        choices=["pretrain", "eval"],
                        help="pretrain = multi-env pretraining | eval = evaluate checkpoint")

    # ── Multi-env settings (default mode) ─────────────────────────────────────
    parser.add_argument("--envs", type=str, default="all",
                        help="'all' = all 9 locomotion envs, or comma-separated minari IDs")

    # ── Single-env debug mode ─────────────────────────────────────────────────
    parser.add_argument("--single_env", type=str, default=None,
                        help="Debug only: train on one env (e.g. hopper-medium-v2)")

    # ── Evaluation ────────────────────────────────────────────────────────────
    parser.add_argument("--eval_env", type=str, default="hopper-medium-v2",
                        help="Environment to evaluate on")
    parser.add_argument("--checkpoint", type=str, default=None)

    # ── ICL ───────────────────────────────────────────────────────────────────
    parser.add_argument("--context_len",  type=int, default=config.context_len)
    parser.add_argument("--n_candidates", type=int, default=config.n_candidates)

    # ── Proposal ──────────────────────────────────────────────────────────────
    parser.add_argument("--proposal_type", type=str, default=config.proposal_type,
                        choices=["gaussian", "mlp"])

    # ── Value ─────────────────────────────────────────────────────────────────
    parser.add_argument("--shallow_value", action="store_true")
    parser.add_argument("--cql_alpha",     type=float, default=config.cql_alpha)
    parser.add_argument("--cql_n_random",  type=int,   default=config.cql_n_random,
                        help="Random actions for CQL penalty (default 3)")

    # ── Backbone ──────────────────────────────────────────────────────────────
    parser.add_argument("--freeze_backbone",  action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone",
                        action="store_false",
                        help="Unfreeze backbone for full fine-tuning")
    parser.add_argument("--backbone_lr",    type=float, default=config.backbone_lr)

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--pretrain_steps", type=int,   default=config.pretrain_steps)
    parser.add_argument("--batch_size",     type=int,   default=config.batch_size)
    parser.add_argument("--head_lr",        type=float, default=config.head_lr)

    # ── Logging ───────────────────────────────────────────────────────────────
    parser.add_argument("--save_dir",  type=str, default="runs")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--seed",      type=int, default=42)

    args = parser.parse_args()

    # Apply to config
    config.phase           = args.phase
    config.context_len     = args.context_len
    config.n_candidates    = args.n_candidates
    config.proposal_type   = args.proposal_type
    config.shallow_value   = args.shallow_value
    config.cql_alpha       = args.cql_alpha
    config.cql_n_random    = args.cql_n_random
    config.freeze_backbone = args.freeze_backbone
    config.backbone_lr     = args.backbone_lr
    config.pretrain_steps  = args.pretrain_steps
    config.batch_size      = args.batch_size
    config.head_lr         = args.head_lr
    config.save_dir        = args.save_dir
    config.use_wandb       = args.use_wandb
    config.seed            = args.seed

    # Device
    if torch.cuda.is_available():
        config.device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        config.device = "mps"
    else:
        config.device = "cpu"

    return config, args


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_run_name(config, args, multi_env: bool) -> str:
    mode = "multi_env" if multi_env else getattr(args, "single_env", "unknown")
    return (
        f"{mode}_"
        f"ctx{config.context_len}_"
        f"K{config.n_candidates}_"
        f"{config.proposal_type}_"
        f"seed{config.seed}"
    )


def main():
    config, args = parse_args()
    set_seed(config.seed)

    print(f"\n{'='*60}")
    print(f"  TabRL  |  device={config.device}  seed={config.seed}")
    print(f"  phase={config.phase}  freeze_backbone={config.freeze_backbone}")
    print(f"{'='*60}\n")

    # ── PHASE: PRETRAIN ───────────────────────────────────────────────────────
    if config.phase == "pretrain":

        single_env = args.single_env  # None means multi-env (default)
        multi_env  = (single_env is None)

        run_name = make_run_name(config, args, multi_env)
        save_dir = os.path.join(config.save_dir, run_name)
        os.makedirs(save_dir, exist_ok=True)
        run_name_wandb = run_name
        logger   = Logger(save_dir, config, use_wandb=config.use_wandb, run_name=run_name_wandb)

        if multi_env:
            # ── RECOMMENDED PATH: multi-environment pretraining ───────────────
            print("[Main] Mode: MULTI-ENVIRONMENT pretraining (paper results)")
            print(f"[Main] Environments: all 9 D4RL locomotion datasets\n")

            # Choose which envs
            if args.envs == "all":
                minari_ids = ALL_PRETRAIN_ENVS
            else:
                minari_ids = [e.strip() for e in args.envs.split(",")]

            # Download hint
            print("[Main] Tip: pre-download all datasets to avoid timeout:")
            print("  python tabrl/scripts/download_data.py\n")

            # Build multi-env dataloader
            dataloader, normalizers, env_names, max_obs_dim, max_act_dim = \
                build_multi_env_dataloader(
                    minari_ids  = minari_ids,
                    context_len = config.context_len,
                    batch_size  = config.batch_size,
                    num_workers = 4,
                )

            # Feature dim = padded obs + padded act
            feature_dim = max_obs_dim + max_act_dim
            print(f"[Main] Padded feature_dim for backbone: {feature_dim}")

            # The agent needs to know the padded dimensions
            # Use max dims so the backbone handles all environments
            agent = TabRLAgent(
                config,
                obs_dim = max_obs_dim,
                act_dim = max_act_dim,
            )

        else:
            # ── DEBUG PATH: single environment ────────────────────────────────
            print(f"[Main] Mode: SINGLE-ENV debug  ({single_env})")
            print("[Main] Note: this is for debugging only, not for paper results\n")

            minari_id = config.D4RL_TO_MINARI.get(single_env)
            if minari_id is None:
                raise ValueError(f"Unknown env: {single_env}")

            data = load_d4rl_dataset(minari_id)
            normalizers_dict = build_normalizers(data)

            # Normalize
            from TIARA.utils.normalizer import RunningNormalizer
            obs_norm = normalizers_dict["obs"]
            act_norm = normalizers_dict["act"]
            rew_norm = normalizers_dict["rew"]

            data_norm = {
                "observations":      obs_norm.normalize(data["observations"]),
                "actions":           act_norm.normalize(data["actions"]),
                "rewards":           rew_norm.normalize(
                    data["rewards"].reshape(-1,1)).reshape(-1),
                "next_observations": obs_norm.normalize(data["next_observations"]),
                "terminals":         data["terminals"],
            }

            obs_dim = data["observations"].shape[1]
            act_dim = data["actions"].shape[1]

            dataloader = make_pretrain_dataloader(
                data_norm, config, num_workers=4
            )

            agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)

        # ── Common: run pretraining ───────────────────────────────────────────
        total_params = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print(f"[Main] Trainable parameters: {total_params:,}")

        best_ckpt = pretrain(agent, dataloader, config, logger, save_dir)

        print(f"\n[Main] Pretraining complete.")
        print(f"[Main] Checkpoint saved at: {best_ckpt}")
        print(f"\n[Main] Next step — evaluate on each environment:")
        print(f"  python tabrl/train.py --phase eval \\")
        print(f"      --checkpoint {best_ckpt} \\")
        print(f"      --eval_env hopper-medium-v2")

        logger.close()

    # ── PHASE: EVAL ───────────────────────────────────────────────────────────
    elif config.phase == "eval":

        if not args.checkpoint:
            raise ValueError("--checkpoint required for eval phase")
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

        eval_env_name = args.eval_env
        minari_id     = config.D4RL_TO_MINARI.get(eval_env_name)
        if minari_id is None:
            raise ValueError(f"Unknown eval env: {eval_env_name}")

        print(f"[Main] Evaluating on: {eval_env_name}")

        # Load eval environment data to get dims and normalizer
        data = load_d4rl_dataset(minari_id)
        obs_dim = data["observations"].shape[1]
        act_dim = data["actions"].shape[1]

        from TIARA.data.d4rl_loader import EnvNormalizer
        norm = EnvNormalizer(data)

        # Build agent with eval env dims
        # Note: if pretrained with multi-env (padded dims), load with padded dims
        # For simplicity here we use eval env dims directly
        agent = TabRLAgent(config, obs_dim=obs_dim, act_dim=act_dim)
        agent.load(args.checkpoint)
        print(f"[Main] Loaded checkpoint: {args.checkpoint}")

        # Create gymnasium environment
        try:
            import gymnasium as gym
            env_id = eval_env_name.replace("-v2", "-v4")
            env = gym.make(env_id)
        except Exception as e:
            raise RuntimeError(
                f"Could not create gymnasium env '{eval_env_name}': {e}\n"
                f"Install: pip install gymnasium[mujoco]"
            )

        from TIARA.evaluation.evaluator import evaluate_policy
        from TIARA.utils.normalizer import RunningNormalizer
        normalizers = build_normalizers(data)

        score = evaluate_policy(
            agent, env, normalizers, config, n_episodes=10
        )
        print(f"\n[Main] Normalized score on {eval_env_name}: {score:.2f}")
        env.close()


if __name__ == "__main__":
    main()