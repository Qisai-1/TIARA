# TabRL — Tabular Foundation Model for Reinforcement Learning

**Propose-Evaluate-Act with In-Context RL (PEARL)**

A NeurIPS-targeted research codebase that adapts TabPFN as a tabular backbone
for RL, using two-pass in-context learning to propose and evaluate actions.

## Architecture

```
Pass 1 (Proposal):   [context rows: s,a,r] + [query row: s,?,?]  → backbone → Gaussian head → K candidate actions
Pass 2 (Evaluation): [context rows: s,a,r] + [K candidate rows: s,âₖ,?] → backbone → value head → Q̂(s,âₖ)
Action selection:    argmax_k Q̂(s,âₖ)    [or UCB: argmax Q̂ + β·σ̂]
TD training:         MSE(Q̂(s,a), r + γ·max Q̂(s',â'))
```

## Project Structure

```
tabrl/
├── configs/
│   └── base_config.py          # All hyperparameters in one place
├── data/
│   ├── d4rl_loader.py          # Minari-based D4RL dataset loading
│   └── replay_buffer.py        # Replay buffer with context window sampling
├── models/
│   ├── backbone.py             # TabPFN backbone wrapper (extracts internals)
│   ├── proposal_head.py        # Gaussian proposal head (P2)
│   ├── value_head.py           # Value/Q head
│   └── tabrl_agent.py          # Full agent combining all components
├── training/
│   ├── pretrain.py             # Offline BC + CQL pretraining on D4RL
│   └── td_trainer.py           # TD training loop (online fine-tuning)
├── evaluation/
│   └── evaluator.py            # Policy rollout and D4RL score normalization
├── utils/
│   ├── normalizer.py           # Running mean/std for states/actions/rewards
│   └── logger.py               # W&B + CSV logging
└── train.py                    # Main entry point
```

## Quick Start

```bash
pip install tabpfn minari gymnasium[mujoco] torch wandb

# Offline pretraining on hopper-medium-v2
python train.py --env hopper-medium-v2 --phase pretrain --context_len 64 --n_candidates 16

# Then online TD fine-tuning
python train.py --env hopper-medium-v2 --phase online --checkpoint runs/hopper/pretrain_best.pt
```

## Key Hyperparameters

| Parameter | Default | Description |
|---|---|---|
| `context_len` | 64 | ICL context window (rows fed to backbone) |
| `n_candidates` | 16 | K candidate actions proposed per step |
| `gamma` | 0.99 | Discount factor |
| `beta_ucb` | 0.1 | UCB exploration coefficient |
| `cql_alpha` | 1.0 | CQL conservatism weight (offline phase) |
| `backbone_lr` | 1e-5 | Very slow backbone LR (or 0 to freeze) |
| `head_lr` | 3e-4 | Head learning rate |

## Ablations (reproduce Table 1 in paper)

```bash
# Ablation: context length
for ctx in 0 16 64 128 256; do
    python train.py --env hopper-medium-v2 --context_len $ctx
done

# Ablation: number of candidates
for k in 1 4 16 64; do
    python train.py --env hopper-medium-v2 --n_candidates $k
done

# Ablation: proposal mechanism
python train.py --env hopper-medium-v2 --proposal_type mlp      # P1
python train.py --env hopper-medium-v2 --proposal_type gaussian # P2 (default)

# Ablation: value head re-encoding
python train.py --env hopper-medium-v2 --shallow_value          # Design X (no re-encode)
```
