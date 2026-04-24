# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**TIARA** (Tabular In-Context RL Agent) — a NeurIPS-targeted research codebase that fine-tunes TabPFN as a dedicated RL backbone using two-pass in-context learning to propose and evaluate actions across multiple environments simultaneously.

Unlike conventional offline RL methods that train a separate policy per environment, TIARA generalizes across tasks at inference time through ICL — no gradient updates required at test time.

The package directory is named `TIARA` on disk. `train.py` and `train_ddp.py` use dynamic `importlib` imports so they work regardless of the directory name.

## Environment Setup

```bash
conda activate dqn
source activate_tabrl.sh   # sets PYTHONPATH, MINARI_DATASETS_PATH, WANDB_PROJECT

pip install tabpfn minari gymnasium[mujoco] torch wandb tqdm
```

`MINARI_DATASETS_PATH` must be set before loading datasets; it points to where Minari caches `.hdf5` files.

## Commands

**Download all datasets (run once before training):**
```bash
python scripts/download_data.py
```

**Smoke tests (no TabPFN or MuJoCo required — uses fallback TransformerBackbone):**
```bash
python tests/test_smoke.py
```

**Stage 1 — frozen backbone pretraining (local single-GPU):**
```bash
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --use_wandb
```

**Stage 2 — fine-tune from frozen checkpoint (local single-GPU):**
```bash
PYTHONPATH=.. python train.py --phase finetune \
    --warmstart runs/.../pretrain_best.pt \
    --head_lr 1e-4 --backbone_lr 1e-7 --pretrain_steps 300000
```

**Evaluate a checkpoint:**
```bash
PYTHONPATH=.. python train.py --phase eval \
    --checkpoint runs/.../pretrain_best.pt \
    --eval_env hopper-medium-v2
```

**Multi-GPU DDP (cluster, recommended for paper runs):**
```bash
# Stage 1
sbatch scripts/pretrain_frozen.sbatch

# Stage 2 — edit FROZEN_CKPT in the script first
sbatch scripts/pretrain_finetune.sbatch

# Ablations
sbatch scripts/pretrain_ablation.sbatch
```

**Single-env debug mode only:**
```bash
PYTHONPATH=.. python train.py --phase pretrain --single_env hopper-medium-v2
```

## Architecture

### Two-Pass Inference

Every forward pass constructs a **context table** of `L` past transitions as rows, with columns `[s, a, RTG]`, and runs two backbone passes:

```
Context table:  [(s₀, a₀, RTG₀), ..., (s_{L-1}, a_{L-1}, RTG_{L-1})]
                              ↓
              TabPFN Backbone (18-layer dual-attention transformer)
              • self_attn_between_features  (attends across columns)
              • self_attn_between_items     (attends across rows)
              • hidden_dim = 384
                         ↓
    Pass 1 — Proposal              Pass 2 — Evaluation
    query: [s, ?, ?]               query: [s, âₖ, ?] for k=1..K
    → Gaussian head → μ, σ         → Value head → Q̂(s, âₖ)
    → sample K candidates
                         ↓
        argmax_k [Q̂(s, âₖ) + β·σ̂]   (UCB action selection)
```

`TabRLAgent.propose()` runs Pass 1; `TabRLAgent.evaluate()` runs Pass 2; `select_action()` chains them. `td_step()` and `bc_step()` are the training methods.

**Training losses (applied simultaneously):**
```
L_BC    = -log N(a_dataset | μ, σ²)                         # proposal head
L_TD    = MSE(Q̂(s,a), r + γ·max_k Q̂_target(s', â'_k))    # value head
L_CQL   = E[Q̂(s, a_random)] - E[Q̂(s, a_dataset)]          # conservatism
L_total = L_BC + 0.01 · L_TD
```

### Backbone

`models/backbone.py` defines two backbones with identical interfaces:

- **`TabPFNBackbone`** — wraps TabPFN's 18-layer dual-attention transformer. Extracts pre-decoder hidden states (dim=384) via a **persistent forward hook** on `decoder_dict.standard.0` (the 5000-bin output head is discarded). Input is batched as `(L+Q, B, feat_dim)` sequence-first for a single efficient forward pass.
- **`TransformerBackbone`** — randomly initialized fallback (hidden_dim=256) when TabPFN is unavailable. Used by smoke tests.

`build_backbone()` tries TabPFN first and silently falls back. Both return `(B, Q, hidden_dim)`.

### Heads

- **`proposal_head`** — Gaussian head produces `K` candidate actions by sampling from `N(μ, σ)` where `μ, σ` come from a small MLP on the backbone hidden state. An MLP ablation (`--proposal_type mlp`) adds a diversity loss.
- **`value_head`** — MLP scoring each candidate action for Q-value estimation. Default "Design Y" re-encodes candidates through the backbone. `--shallow_value` ("Design X") ablation skips re-encoding and feeds `[h1, candidate]` directly.
- **Target networks** — soft-updated copies of `value_head` (and optionally `backbone`) updated via Polyak averaging (`τ=0.001`) each step.

### Multi-Environment Training

Trains on all 9 Minari locomotion datasets simultaneously (~9M transitions, Hopper/HalfCheetah/Walker2d × medium/simple/expert):

- Each environment has its own `EnvNormalizer` (obs/act dims differ across envs).
- All obs/act arrays are zero-padded to `max_obs_dim=17, max_act_dim=6` → 23-column context table.
- `MultiEnvDataset.__getitem__` ignores index and samples uniformly across environments, preventing larger datasets from dominating.
- `context_y` is **return-to-go (RTG)**, not raw reward. RTG aligns the ICL pretraining task with Q-value estimation.

### Two-Stage Training

The recommended workflow per the README:

1. **Frozen backbone** (`pretrain_frozen.sbatch`) — only proposal + value heads train (≈397K params). Fast, stable. Establishes good head weights.
2. **Full fine-tuning** (`pretrain_finetune.sbatch`) — backbone unfrozen, heads warm-started from Stage 1 checkpoint via `--warmstart`. Backbone starts from TabPFN pretrained weights (not randomly); frozen checkpoint has `backbone_state=None`.

**BC-only warmup**: first 5000 steps use BC loss only (no TD). The proposal head needs to converge before TD provides meaningful signal.

### Training Stability Details

These fixes were required to stabilize training (documented in README Table):

| Fix | Detail |
|---|---|
| `tau` reduced to 0.001 | Target network updated too fast → TD instability |
| `td_weight=0.01` | TD loss was 70× larger than BC, overwhelming proposal head |
| BC-only warmup (5000 steps) | Proposal head must converge before TD signal is useful |
| TD target clamping `±200` | Prevents Q-value explosion in early training |
| Backbone grad clip `0.1` | Protects pretrained representations from large updates |
| RTG as `context_y` | One dataset path was using raw rewards — now fixed |

### Key Config (`configs/base_config.py`)

All hyperparameters live in `TabRLConfig` dataclass. Paper-recommended values differ by stage:

| Parameter | Frozen Stage | Finetune Stage |
|---|---|---|
| `context_len` | 32 | 32 |
| `n_candidates` (K) | 4 | 4 |
| `batch_size` | 256 | 32 |
| `head_lr` | 3e-4 | 1e-4 |
| `backbone_lr` | N/A (frozen) | 1e-7 |
| `cql_alpha` | 1.0 | 0.1 |

Non-obvious config details:
- `freeze_backbone`: defaults to `True` in CLI; use `--no_freeze_backbone` to unfreeze
- `cql_n_random`: 3 random actions for CQL penalty (not 10 as in original CQL paper)
- `td_weight=0.01`: TD loss scaled down relative to BC loss
- `context_sampling`: `"recent"` (sliding window) | `"priority"` (TD-error) | `"random"`

### Checkpoints

`agent.save()` stores `proposal_head`, `value_head`, `value_head_target`, `backbone_state`. `backbone_state` is `None` when backbone is frozen — `agent.load()` handles this gracefully, keeping the backbone at TabPFN pretrained weights when loading a frozen checkpoint into an unfrozen agent.
