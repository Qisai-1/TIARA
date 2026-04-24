# TIARA — Session Change Report
**Date:** 2026-04-23  
**Author:** Claude Code (claude-sonnet-4-6)

---

## Overview

This session covered four areas of work:

1. **Documentation** — Created `CLAUDE.md` to orient future Claude Code sessions in this repository.
2. **Training performance** — Identified and fixed four bottlenecks causing slow training.
3. **Two-stage training pipeline** — Wired up a frozen-backbone → unfrozen fine-tune workflow across `train.py`, `train_ddp.py`, and the SLURM sbatch scripts.
4. **README sync** — Project author updated `README.md`; `CLAUDE.md` updated to reflect the new content.

---

## 1. CLAUDE.md — New File (later updated)

**File:** `CLAUDE.md`

Created the project documentation file for Claude Code. Contents:

- What the project is (TIARA — Tabular In-Context RL Agent; TabPFN fine-tuned as RL backbone)
- Environment setup: conda env, `activate_tabrl.sh`, required pip packages, `MINARI_DATASETS_PATH`
- All common commands: data download, smoke tests, single-GPU training, DDP multi-GPU training, evaluation, SLURM sbatch scripts
- High-level architecture: two-pass inference loop, backbone internals (TabPFN hook extraction, fallback TransformerBackbone), proposal and value heads, multi-environment training design (RTG as context_y, zero-padding, uniform env sampling)
- Non-obvious config details: `freeze_backbone` CLI flag inversion, `td_weight` scaling, `cql_n_random`, checkpoint format

Later updated in section 4 to reflect the README rewrite.

---

## 2. Training Performance Fixes

### 2.1 `models/backbone.py` — Cached module lookup + persistent hook

**Root cause:** `dict(self.model.named_modules())` was called on every backbone forward pass, traversing the entire TabPFN module graph and building a dict each time. With 4–5 backbone invocations per training step, this was the primary bottleneck. On top of this, `register_forward_hook()` and `handle.remove()` were also called every forward.

**Fix:** Moved the module lookup to `__init__` and registered a single persistent hook (`_capture_hook`) at construction time. The `forward()` method now just calls `self._captured.clear()` before the model call.

```python
# Before (called on every forward pass):
hook_mod = dict(self.model.named_modules())["decoder_dict.standard.0"]
handle   = hook_mod.register_forward_hook(post_hook)
try:
    self.model(...)
finally:
    handle.remove()

# After (registered once in __init__):
self._captured: dict = {}
self._hook_handle = modules["decoder_dict.standard.0"].register_forward_hook(
    self._capture_hook
)
# In forward():
self._captured.clear()
self.model(...)
h = self._captured.get("h")
```

---

### 2.2 `models/tabrl_agent.py` — Merged backbone calls in `td_step`

**Root cause:** `td_step` was running two separate backbone forward passes with the same `context_X` and `query_obs`: one for `Q_dataset` (the single dataset action) and one for `Q_random` (CQL penalty random actions).

**Fix:** Concatenated the dataset action and random actions into a single tensor `(B, 1+cql_n_random, act_dim)` and called `evaluate()` once, then split the result.

```python
# Before: two backbone calls
Q_dataset = self.evaluate(context_X, context_y, query_obs, query_act_expanded)[:, 0]
Q_random  = self.evaluate(context_X, context_y, query_obs, random_acts)

# After: one backbone call
all_acts  = torch.cat([query_act_expanded, random_acts], dim=1)  # (B, 1+K, act)
Q_all     = self.evaluate(context_X, context_y, query_obs, all_acts)
Q_dataset = Q_all[:, 0]
Q_random  = Q_all[:, 1:]
```

The `cql_alpha == 0` path (no CQL) retains a single-call path unchanged.

---

### 2.3 `data/d4rl_loader.py` — `persistent_workers` + pre-existing bug fix

**Performance fix:** The multi-env DataLoader was missing `persistent_workers=True`. Without it, worker processes were killed and respawned at every epoch boundary.

```python
# Before:
dataloader = DataLoader(..., num_workers=min(num_workers, ...), ...)

# After:
n_workers  = min(num_workers, os.cpu_count() or 1)
dataloader = DataLoader(..., num_workers=n_workers, persistent_workers=n_workers > 0, ...)
```

**Pre-existing bug fix:** `ICLTransitionDataset.__getitem__` referenced `self.rtg` but `__init__` never computed it, causing `AttributeError` in the `test_dataset` smoke test. Fixed by computing RTG in `__init__` using the existing static method from `ICLEnvDataset`:

```python
self.rtg = ICLEnvDataset._compute_rtg(data["rewards"], data["terminals"])
```

---

### 2.4 `train_ddp.py` — `persistent_workers`

Same `persistent_workers=True` fix applied to the DDP DataLoader inside `build_ddp_dataloader`.

---

## 3. Two-Stage Training Pipeline

The motivation: train with a frozen backbone first (cheap, fast, good head initialization), then load those head weights as the starting point for full end-to-end fine-tuning with the backbone unfrozen.

The existing `agent.load()` method already handled the key case correctly: when a frozen checkpoint (where `backbone_state=None`) is loaded into an unfrozen agent, the heads are restored and the backbone is left at its TabPFN pretrained weights. The missing piece was the CLI and script wiring.

### 3.1 `train.py` — New `finetune` phase

**Changes:**

- Added `"finetune"` to `--phase` choices (`pretrain | finetune | eval`).
- Added `--warmstart` argument: path to the frozen-backbone checkpoint.
- At parse time, `finetune` phase validates that `--warmstart` is provided and forces `freeze_backbone=False` regardless of other flags.
- New `finetune` phase block in `main()`: builds the agent and dataloader identically to `pretrain`, calls `agent.load(args.warmstart)` to restore head weights, then runs the standard `pretrain()` training loop.
- Run name prefixed with `"finetune_"` so checkpoints land in a separate directory.
- End-of-pretrain message updated to print the finetune command as the suggested next step.
- `make_run_name()` updated to detect `config.phase == "finetune"`.

**Usage:**
```bash
# Stage 1: frozen backbone
PYTHONPATH=.. python train.py --phase pretrain --freeze_backbone --use_wandb

# Stage 2: fine-tune from Stage 1 checkpoint
PYTHONPATH=.. python train.py --phase finetune \
    --warmstart runs/.../pretrain_best.pt \
    --head_lr 1e-4 --backbone_lr 1e-7 --pretrain_steps 300000
```

---

### 3.2 `train_ddp.py` — `--warmstart` flag

**Changes:**

- Added `--warmstart` argument (path to frozen checkpoint), stored as `config.warmstart`.
- Added loading logic before the training loop, structured as `if/elif` so `--warmstart` and `--resume` are mutually exclusive:

```python
if warmstart and os.path.exists(warmstart):
    raw_agent.load(warmstart)          # heads loaded; backbone = TabPFN pretrained
elif config.resume and os.path.exists(config.resume):
    raw_agent.load(config.resume)      # interrupted-run resumption
    start_step = config.resume_step + 1
```

Note: `--resume` is for resuming an interrupted run of the same configuration. `--warmstart` is for the frozen → finetune transition across configurations.

---

### 3.3 `scripts/pretrain_finetune.sbatch` — Completed implementation

The script previously had a placeholder comment block and launched a from-scratch unfrozen run with no checkpoint.

**Changes:**

- Defines `FROZEN_CKPT` pointing to the Stage 1 output directory.
- Validates the checkpoint exists before launching and exits with a clear error message if not.
- Passes `--no_freeze_backbone` and `--warmstart $FROZEN_CKPT` to `torchrun`.
- Matches `pretrain_frozen.sbatch` structure: logging setup, GPU confirmation, `activate_tabrl.sh`, `PYTORCH_CUDA_ALLOC_CONF`.
- Uses lower learning rates appropriate for fine-tuning: `--head_lr 1e-4`, `--backbone_lr 1e-7`, `--batch_size 32`.

**Usage:**
```bash
# Update FROZEN_CKPT path in the script, then:
sbatch scripts/pretrain_finetune.sbatch
```

---

## 4. README.md + CLAUDE.md — Post-session README sync

The project author updated `README.md` after the session changes above were made. `CLAUDE.md` was updated to stay in sync.

### What changed in README.md

| Section | Change |
|---|---|
| Project name | Renamed from "TabRL — Tabular Foundation Model for RL (PEARL)" to **"TIARA — Tabular In-Context RL Agent"** |
| Architecture diagram | Expanded to show 18-layer dual-attention transformer internals (`self_attn_between_features`, `self_attn_between_items`, `hidden_dim=384`) |
| Training losses | Added explicit loss formulas: `L_BC`, `L_TD`, `L_CQL`, `L_total = L_BC + 0.01·L_TD` |
| Key design decisions | New section: RTG as context label, two-stage training, BC-only warmup (5000 steps) |
| Project structure | Added `baselines/run_baselines.py`; updated file descriptions to match current code |
| Datasets table | Added per-environment transition counts and Minari IDs (~9M transitions total) |
| Quick Start | Shows two-stage flow (`pretrain` → `finetune`); uses `PYTHONPATH=..` prefix |
| Cluster Training | Added `sbatch` commands for both stages and `squeue` / `tail` monitoring commands |
| Hyperparameters | Split into Frozen / Finetune columns; updated values (`backbone_lr` 1e-5 → 1e-7, `context_len` 64 → 32, `n_candidates` 16 → 4, finetune `batch_size` = 32); added `bc_only_steps`, `grad_clip` rows |
| Training Stability | New table documenting all stability fixes and the reason for each |
| Ablations | Updated context lengths (0/16/32/64); commands updated to use `PYTHONPATH=..` |
| Zero-shot experiment | New section: train on hopper + halfcheetah only, eval zero-shot on walker2d |
| Expected results | New Table 1 with baseline scores (BC, TD3+BC, CQL, IQL, DT); TIARA rows blank pending results |
| Citation | Added BibTeX entry |

### What changed in CLAUDE.md

- Project name corrected to TIARA throughout
- Architecture updated with 18-layer dual-attention detail and explicit loss formulas
- Added "Two-Stage Training" subsection documenting the `--warmstart` workflow and BC-only warmup
- Added "Training Stability Details" table (prevents re-introducing known-bad configurations)
- Commands updated to use `PYTHONPATH=..` and show both `pretrain` and `finetune` phases
- Hyperparameter table updated: `backbone_lr=1e-7`, `context_len=32`, `n_candidates=4`, finetune `batch_size=32`

---

## Summary Table

| File | Change Type | Description |
|---|---|---|
| `CLAUDE.md` | New → Updated | Created, then updated to match README rewrite |
| `README.md` | External (author) | Major rewrite — project rename, architecture detail, two-stage docs, stability table |
| `models/backbone.py` | Performance | Cache `named_modules()` lookup; register hook once at init |
| `models/tabrl_agent.py` | Performance | Merge `Q_dataset` + `Q_random` into single backbone call in `td_step` |
| `data/d4rl_loader.py` | Performance + Bug | `persistent_workers=True`; fix missing `self.rtg` in `ICLTransitionDataset` |
| `train_ddp.py` | Performance + Feature | `persistent_workers=True`; add `--warmstart` arg and load logic |
| `train.py` | Feature | Add `finetune` phase, `--warmstart` arg, updated next-step messages |
| `scripts/pretrain_finetune.sbatch` | Feature | Complete two-stage warm-start script with checkpoint validation |

---

## Verification

All smoke tests pass after every change:

```
=== Smoke Test: Replay Buffer ===       ✓
=== Smoke Test: ICL Dataset ===         ✓  (was broken before — rtg bug fixed)
=== Smoke Test: Agent Forward Pass ===  ✓
=== Smoke Test: TD Step ===             ✓
=== Smoke Test: BC Step ===             ✓
=== Smoke Test: MLP Proposal Head ===   ✓
=== Smoke Test: Shallow Value Head ===  ✓

ALL SMOKE TESTS PASSED ✓
```
