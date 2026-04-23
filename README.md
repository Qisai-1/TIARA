# TIARA — Tabular In-Context RL Agent

**A Tabular Foundation Model for Offline Reinforcement Learning**

A NeurIPS-targeted research codebase that fine-tunes TabPFN as a dedicated RL backbone,
using two-pass in-context learning to propose and evaluate actions across multiple environments.

## Core Idea

Unlike conventional offline RL methods that train a separate policy per environment,
TIARA generalizes across tasks at inference time through in-context learning (ICL) —
no gradient updates required at test time. The context window of recent transitions
adapts the model to new tasks purely through attention.

## Architecture

```
Context table:  [(s₀, a₀, RTG₀), (s₁, a₁, RTG₁), ..., (s_{L-1}, a_{L-1}, RTG_{L-1})]
                                    ↓
                         TabPFN Backbone (18-layer dual-attention transformer)
                         • self_attn_between_features  (attends across columns)
                         • self_attn_between_items      (attends across rows)
                         • hidden_dim=384, fine-tuned on RL data
                                    ↓
        ┌──────────────────────────┴──────────────────────────┐
        │ Pass 1 — Proposal                                    │ Pass 2 — Evaluation
        │ query: [s, ?, ?]                                     │ query: [s, âₖ, ?] for k=1..K
        │ → Gaussian head → μ, σ                              │ → Value head → Q̂(s, âₖ)
        │ → sample K candidates                               │
        └──────────────────────────┬──────────────────────────┘
                                    ↓
                    Action selection: argmax_k [Q̂(s,âₖ) + β·σ̂]   (UCB)
```

**Training losses (applied simultaneously):**
```
L_BC  = -log N(a_dataset | μ, σ²)                          # proposal head
L_TD  = MSE(Q̂(s,a), r + γ·max_k Q̂_target(s', â'_k))     # value head
L_CQL = E[Q̂(s, a_random)] - E[Q̂(s, a_dataset)]           # conservatism
L_total = L_BC + λ_TD · L_TD    (λ_TD=0.01)
```

**Key design decisions:**
- **RTG as context label** — Return-to-go replaces raw reward as `context_y`, aligning the ICL task with TabPFN's pretraining (regression from labeled context rows)
- **Two-stage training** — frozen backbone first (stable heads), then end-to-end fine-tuning
- **BC-only warmup** — no TD loss for first 5000 steps; proposal head converges before value head trains

## Project Structure

```
TIARA/
├── configs/
│   └── base_config.py          # All hyperparameters as a dataclass
├── data/
│   ├── d4rl_loader.py          # Minari dataset loading + RTG computation + multi-env padding
│   └── replay_buffer.py        # Replay buffer with context window sampling
├── models/
│   ├── backbone.py             # TabPFN wrapper — batched forward, hook on decoder_dict.standard.0
│   ├── proposal_head.py        # Gaussian proposal head (P2) and MLP ablation (P1)
│   ├── value_head.py           # Value/Q head with CQL loss
│   └── tabrl_agent.py          # Full agent: propose → evaluate → select
├── training/
│   ├── pretrain.py             # Offline BC + TD + CQL pretraining loop
│   └── td_trainer.py           # TD fine-tuning loop
├── evaluation/
│   └── evaluator.py            # Policy rollout with multi-env obs/act padding
├── baselines/
│   └── run_baselines.py        # BC, TD3+BC, CQL, IQL, DT baselines
├── utils/
│   ├── normalizer.py           # Running mean/std for obs/act/rewards
│   └── logger.py               # W&B + CSV logging
├── scripts/
│   ├── pretrain_frozen.sbatch  # SLURM: Stage 1 frozen backbone (48hr)
│   ├── pretrain_finetune.sbatch# SLURM: Stage 2 full fine-tuning (72hr)
│   ├── pretrain_ablation.sbatch# SLURM: Job array for all ablations
│   └── download_data.py        # Download all 9 Minari datasets
├── tests/
│   └── test_smoke.py           # Smoke tests for all components
├── train.py                    # Single-GPU entry point (pretrain + eval)
├── train_ddp.py                # Multi-GPU DDP entry point (cluster)
└── activate_tabrl.sh           # Environment setup (PYTHONPATH, MINARI_DATASETS_PATH)
```

## Datasets

TIARA pretrains on 9 Minari locomotion datasets simultaneously (~9M transitions total):

| Environment | Quality | Minari ID | Transitions |
|---|---|---|---|
| Hopper | Medium | `mujoco/hopper/medium-v0` | 999,404 |
| Hopper | Simple (≈medium-replay) | `mujoco/hopper/simple-v0` | 999,206 |
| Hopper | Expert | `mujoco/hopper/expert-v0` | 999,164 |
| HalfCheetah | Medium | `mujoco/halfcheetah/medium-v0` | 1,000,000 |
| HalfCheetah | Simple | `mujoco/halfcheetah/simple-v0` | 1,000,000 |
| HalfCheetah | Expert | `mujoco/halfcheetah/expert-v0` | 1,000,000 |
| Walker2d | Medium | `mujoco/walker2d/medium-v0` | 999,613 |
| Walker2d | Simple | `mujoco/walker2d/simple-v0` | 999,942 |
| Walker2d | Expert | `mujoco/walker2d/expert-v0` | 999,190 |

Input padding: all environments padded to `max_obs_dim=17, max_act_dim=6` → 23-column context table.

## Quick Start (Local)

```bash
# Install dependencies
pip install tabpfn minari gymnasium[mujoco] torch wandb

# Download datasets
python scripts/download_data.py

# Run smoke tests
python tests/test_smoke.py

# Stage 1 — frozen backbone pretraining
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --use_wandb

# Evaluate checkpoint
PYTHONPATH=.. python train.py --phase eval \
    --checkpoint runs/frozen/pretrain_best.pt \
    --eval_env hopper-medium-v2
```

## Cluster Training (Iowa State Nova)

```bash
# Setup environment
source /work/mech-ai-scratch/supersai/TIARA/activate_tabrl.sh
conda activate dqn

# Download datasets to TIARA storage
python scripts/download_data.py

# Stage 1 — frozen backbone (submit first, ~48hrs on 4x H200)
sbatch scripts/pretrain_frozen.sbatch

# Stage 2 — full fine-tuning (submit after Stage 1 completes)
sbatch scripts/pretrain_finetune.sbatch

# Monitor
squeue -u $USER
tail -f /work/mech-ai-scratch/supersai/TIARA/logs/frozen_<JOBID>.out

# W&B dashboard
# https://wandb.ai/supersai/tabrl
```

## Key Hyperparameters

| Parameter | Frozen Stage | Finetune Stage | Description |
|---|---|---|---|
| `context_len` | 32 | 32 | ICL context window (rows fed to backbone) |
| `n_candidates` (K) | 4 | 4 | Action candidates proposed per step |
| `batch_size` | 256 | 32 | Per-GPU batch size |
| `head_lr` | 3e-4 | 1e-4 | Proposal + value head learning rate |
| `backbone_lr` | N/A | 1e-7 | TabPFN LR (very conservative) |
| `tau` | 0.001 | 0.001 | Target network update rate |
| `td_weight` (λ_TD) | 0.01 | 0.01 | TD loss scale vs BC loss |
| `cql_alpha` | 1.0 | 0.1 | CQL conservatism strength |
| `cql_n_random` | 3 | 3 | Random actions for CQL penalty |
| `gamma` (γ) | 0.99 | 0.99 | Discount factor |
| `beta_ucb` (β) | 0.1 | 0.1 | UCB exploration coefficient |
| `warmup_steps` | 2000 | 2000 | Linear LR warmup steps |
| `bc_only_steps` | 5000 | 5000 | Steps with BC loss only (no TD) |
| `grad_clip` | 1.0 | 1.0 (heads), 0.1 (backbone) | Gradient norm clipping |

## Training Stability Fixes

Several fixes were needed to stabilize training over the original design:

| Fix | Value | Reason |
|---|---|---|
| `tau` reduced | 0.005 → 0.001 | Target network was updating too fast, causing TD instability |
| `td_weight` added | 0.01 | TD loss was 70× larger than BC, overwhelming proposal head |
| BC-only warmup | 5000 steps | Proposal head needs to converge before TD provides signal |
| TD target clamping | ±200 | Prevents Q-value explosion in early training |
| Backbone grad clip | 0.1 | Protects pretrained representations from large updates |
| RTG bug fixed | — | One dataset path was using raw rewards instead of RTG |
| Batched backbone | — | Was looping over batch (0.0 it/s); now single batched call |

## Ablations (reproduce Table 2 in paper)

```bash
# Context length ablation (submit as job array)
sbatch scripts/pretrain_ablation.sbatch

# Or run individually
for ctx in 0 16 32 64; do
    PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --context_len $ctx
done

# Candidate count ablation
for k in 1 4 16; do
    PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --n_candidates $k
done

# Proposal mechanism ablation
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --proposal_type mlp      # P1 ablation
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --proposal_type gaussian # P2 (default)

# Value head design ablation
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --shallow_value  # Design X: no re-encoding
```

## Zero-Shot Experiment

Train on hopper + halfcheetah only, evaluate zero-shot on walker2d:

```bash
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone \
    --envs mujoco/hopper/medium-v0,mujoco/hopper/simple-v0,mujoco/hopper/expert-v0,\
mujoco/halfcheetah/medium-v0,mujoco/halfcheetah/simple-v0,mujoco/halfcheetah/expert-v0

# Evaluate zero-shot on walker2d (never seen during training)
PYTHONPATH=.. python train.py --phase eval \
    --checkpoint runs/zeroshot/pretrain_best.pt \
    --eval_env walker2d-medium-v2
```

## Expected Results (Table 1)

Normalized scores (0 = random, 100 = expert). Baselines from published papers.

| Method | Hopper M | Hopper ME | Hopper MR | HC M | HC ME | HC MR | Walker M | Walker ME | Walker MR | Avg |
|---|---|---|---|---|---|---|---|---|---|---|
| BC | 52.5 | 110.9 | 18.1 | 42.6 | 55.2 | 36.1 | 75.3 | 108.8 | 26.0 | 58.4 |
| TD3+BC | 59.3 | 110.1 | 19.5 | 48.3 | 90.7 | 41.9 | 83.7 | 110.1 | 34.0 | 66.4 |
| CQL | 58.5 | 111.0 | 31.4 | 44.0 | 94.9 | 39.2 | 72.5 | 109.6 | 26.7 | 65.3 |
| IQL | 66.3 | 91.5 | 94.7 | 47.4 | 73.2 | 73.9 | 78.3 | 109.6 | 73.9 | 78.8 |
| DT | 67.6 | 107.9 | 82.7 | 42.6 | 95.0 | 37.4 | 74.0 | 108.1 | 66.6 | 75.8 |
| **TIARA (frozen)** | — | — | — | — | — | — | — | — | — | — |
| **TIARA (finetune)** | — | — | — | — | — | — | — | — | — | — |

M=medium, ME=medium-expert, MR=medium-replay, HC=HalfCheetah

## Citation

```bibtex
@article{tiara2026,
  title   = {TIARA: Tabular In-Context RL Agent},
  author  = {Supersai},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2026}
}
```