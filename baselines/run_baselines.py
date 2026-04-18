"""
Master Baseline Runner
======================
Trains all baselines on a given D4RL environment and produces a
comparison table in the format expected for NeurIPS Table 1.

Usage:
    python tabrl/baselines/run_baselines.py --env hopper-medium-v2

This script:
  1. Loads D4RL data via Minari
  2. Normalizes observations
  3. Trains BC, TD3+BC, CQL, IQL, DT sequentially (or in parallel with --parallel)
  4. Evaluates each on 10 episodes
  5. Prints a LaTeX-formatted Table 1

Expected wall-clock time on a single GPU:
  BC:     ~10 min (1M steps)
  TD3+BC: ~30 min (1M steps)
  CQL:    ~60 min (1M steps)
  IQL:    ~30 min (1M steps)
  DT:     ~20 min (100K steps)
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from TIARA.configs.base_config import TabRLConfig
from TIARA.data.d4rl_loader import load_d4rl_dataset
from TIARA.utils.normalizer import build_normalizers

from baselines.bc.bc           import BCTrainer
from baselines.td3bc.td3bc     import TD3BCTrainer
from baselines.cql.cql         import CQLTrainer
from baselines.iql.iql         import IQLTrainer
from baselines.dt.dt           import DTTrainer


# DT target returns per environment (used for autoregressive rollout)
DT_TARGET_RETURNS = {
    "hopper-medium-v2":              3600,
    "hopper-medium-expert-v2":       3600,
    "hopper-medium-replay-v2":       3600,
    "halfcheetah-medium-v2":         6000,
    "halfcheetah-medium-expert-v2":  12000,
    "halfcheetah-medium-replay-v2":  6000,
    "walker2d-medium-v2":            5000,
    "walker2d-medium-expert-v2":     5000,
    "walker2d-medium-replay-v2":     5000,
}

# IQL hyperparameters per task type
IQL_HYPERPARAMS = {
    "locomotion": {"expectile": 0.7, "temperature": 3.0},
    "antmaze":    {"expectile": 0.9, "temperature": 10.0},
}


def get_device() -> str:
    if torch.cuda.is_available():   return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env",       type=str, default="hopper-medium-v2")
    parser.add_argument("--baselines", type=str, default="bc,td3bc,cql,iql,dt",
                        help="Comma-separated list of baselines to run")
    parser.add_argument("--n_steps",   type=int, default=1_000_000)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--save_dir",  type=str, default="runs/baselines")
    parser.add_argument("--no_eval",   action="store_true",
                        help="Skip environment evaluation (report train loss only)")
    return parser.parse_args()


def run_baselines(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    print(f"\n{'='*65}")
    print(f"  TabRL Baseline Runner")
    print(f"  Env:      {args.env}")
    print(f"  Device:   {device}")
    print(f"  Baselines: {args.baselines}")
    print(f"{'='*65}\n")

    config      = TabRLConfig()
    config.env_name = args.env

    # ── Load data ─────────────────────────────────────────────────────────────
    minari_id   = config.D4RL_TO_MINARI.get(args.env)
    if minari_id is None:
        raise ValueError(f"Unknown env: {args.env}")
    data        = load_d4rl_dataset(minari_id)
    normalizers = build_normalizers(data)

    # Normalize observations for baselines
    obs_mean = data["observations"].mean(0)
    obs_std  = data["observations"].std(0) + 1e-8
    data_norm = dict(data)
    data_norm["observations"]      = (data["observations"]      - obs_mean) / obs_std
    data_norm["next_observations"] = (data["next_observations"] - obs_mean) / obs_std

    obs_dim = data["observations"].shape[1]
    act_dim = data["actions"].shape[1]
    print(f"obs_dim={obs_dim}  act_dim={act_dim}  N={len(data['observations']):,}\n")

    # ── Optional: gym environment for evaluation ───────────────────────────────
    env = None
    if not args.no_eval:
        try:
            import gymnasium as gym
            env_id = args.env.replace("-v2", "-v4") if "-v2" in args.env else args.env
            env = gym.make(env_id)
            print(f"Environment created: {env_id}")
        except Exception as e:
            print(f"Warning: could not create environment ({e}). Skipping eval.")

    # ── Task type (for IQL hyperparameter selection) ───────────────────────────
    task_type = "antmaze" if "antmaze" in args.env else "locomotion"

    # ── Results collector ──────────────────────────────────────────────────────
    results = {}
    save_dir = os.path.join(args.save_dir, args.env)
    os.makedirs(save_dir, exist_ok=True)

    baselines_to_run = [b.strip() for b in args.baselines.split(",")]

    # ── BC ─────────────────────────────────────────────────────────────────────
    if "bc" in baselines_to_run:
        print("\n" + "─" * 50)
        trainer = BCTrainer(obs_dim, act_dim, config,
                            save_dir=os.path.join(save_dir, "bc"),
                            device=device)
        score = trainer.train(
            data_norm, env=env,
            n_steps=args.n_steps,
            eval_every=50_000,
        )
        results["BC"] = score
        print(f"BC final score: {score:.2f}")

    # ── TD3+BC ─────────────────────────────────────────────────────────────────
    if "td3bc" in baselines_to_run:
        print("\n" + "─" * 50)
        trainer = TD3BCTrainer(obs_dim, act_dim, config,
                               save_dir=os.path.join(save_dir, "td3bc"),
                               device=device)
        score = trainer.train(
            data_norm, env=env,
            n_steps=args.n_steps,
            eval_every=50_000,
        )
        results["TD3+BC"] = score
        print(f"TD3+BC final score: {score:.2f}")

    # ── CQL ────────────────────────────────────────────────────────────────────
    if "cql" in baselines_to_run:
        print("\n" + "─" * 50)
        cql_weight = 5.0 if task_type == "antmaze" else 1.0
        trainer = CQLTrainer(obs_dim, act_dim, config,
                             save_dir=os.path.join(save_dir, "cql"),
                             device=device, cql_weight=cql_weight)
        score = trainer.train(
            data_norm, env=env,
            n_steps=args.n_steps,
            eval_every=50_000,
        )
        results["CQL"] = score
        print(f"CQL final score: {score:.2f}")

    # ── IQL ────────────────────────────────────────────────────────────────────
    if "iql" in baselines_to_run:
        print("\n" + "─" * 50)
        iql_hp = IQL_HYPERPARAMS[task_type]
        trainer = IQLTrainer(obs_dim, act_dim, config,
                             save_dir=os.path.join(save_dir, "iql"),
                             device=device, **iql_hp)
        score = trainer.train(
            data_norm, env=env,
            n_steps=args.n_steps,
            eval_every=50_000,
        )
        results["IQL"] = score
        print(f"IQL final score: {score:.2f}")

    # ── Decision Transformer ────────────────────────────────────────────────────
    if "dt" in baselines_to_run:
        print("\n" + "─" * 50)
        target_ret = DT_TARGET_RETURNS.get(args.env, 3600)
        trainer = DTTrainer(obs_dim, act_dim, config,
                            save_dir=os.path.join(save_dir, "dt"),
                            device=device,
                            context_len=20,
                            target_return=target_ret)
        score = trainer.train(
            data, env=env,          # DT normalizes internally
            n_steps=100_000,        # DT uses fewer steps
            batch_size=64,
            eval_every=10_000,
        )
        results["DT"] = score
        print(f"DT final score: {score:.2f}")

    # ── Print results table ────────────────────────────────────────────────────
    print_results_table(args.env, results)

    # Save JSON
    results_path = os.path.join(save_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump({"env": args.env, "results": results}, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    if env is not None:
        env.close()

    return results


def print_results_table(env_name: str, results: dict):
    """
    Print a LaTeX-style comparison table for the paper.
    """
    # Reference scores from published papers (for comparison)
    PUBLISHED = {
        "hopper-medium-v2": {
            "BC": 52.5, "TD3+BC": 59.3, "CQL": 58.5, "IQL": 66.3, "DT": 67.6,
        },
        "halfcheetah-medium-v2": {
            "BC": 42.6, "TD3+BC": 48.3, "CQL": 44.0, "IQL": 47.4, "DT": 42.6,
        },
        "walker2d-medium-v2": {
            "BC": 75.0, "TD3+BC": 83.7, "CQL": 72.5, "IQL": 78.3, "DT": 74.0,
        },
    }

    print(f"\n{'='*65}")
    print(f"  RESULTS: {env_name}")
    print(f"{'='*65}")
    print(f"  {'Method':<15}  {'Ours':>10}  {'Published':>10}")
    print(f"  {'-'*40}")

    for method, score in results.items():
        pub = PUBLISHED.get(env_name, {}).get(method, "--")
        pub_str = f"{pub:.1f}" if isinstance(pub, float) else pub
        print(f"  {method:<15}  {score:>10.2f}  {pub_str:>10}")

    print(f"  {'-'*40}")
    print(f"\n  TabRL (ours): [run tabrl/train.py to fill this row]")
    print(f"{'='*65}")

    # LaTeX snippet
    print("\n  LaTeX (Table 1 row):")
    methods_str = " & ".join(f"{results.get(m, '--'):.1f}" if m in results else "--"
                              for m in ["BC", "TD3+BC", "CQL", "IQL", "DT"])
    print(f"  {env_name} & {methods_str} & \\textbf{{X.X}} \\\\")


if __name__ == "__main__":
    args = parse_args()
    run_baselines(args)
