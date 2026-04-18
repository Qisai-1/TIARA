from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TabRLConfig:
    # ── Environment ──────────────────────────────────────────────────────────
    env_name: str = "hopper-medium-v2"
    # Supported: hopper-medium-v2, halfcheetah-medium-v2, walker2d-medium-v2,
    #            hopper-medium-expert-v2, halfcheetah-medium-expert-v2, walker2d-medium-expert-v2

    # ── ICL Context ──────────────────────────────────────────────────────────
    context_len: int = 64          # N: number of past transitions in context table
    context_sampling: str = "recent"  # "recent" | "priority" | "random"
    # "recent"   = sliding window of last N transitions
    # "priority" = sample by TD-error (prioritized experience replay style)
    # "random"   = uniform random from buffer

    # ── Proposal Head ────────────────────────────────────────────────────────
    n_candidates: int = 16         # K: number of candidate actions proposed
    proposal_type: str = "gaussian"  # "gaussian" (P2, recommended) | "mlp" (P1 ablation)
    proposal_hidden_dim: int = 256
    proposal_n_layers: int = 2
    diversity_loss_weight: float = 0.1  # only used when proposal_type = "mlp"

    # ── Value Head ───────────────────────────────────────────────────────────
    shallow_value: bool = False    # if True, skip backbone re-encoding (Design X ablation)
    value_hidden_dim: int = 256
    value_n_layers: int = 2

    # ── Exploration ──────────────────────────────────────────────────────────
    use_ucb: bool = True           # UCB action selection using σ from Gaussian head
    beta_ucb: float = 0.1          # UCB coefficient: aₜ = argmax Q̂ + beta*σ̂

    # ── Backbone ─────────────────────────────────────────────────────────────
    backbone_model: str = "v2"     # TabPFN version
    freeze_backbone: bool = False  # True = only train heads (faster, less memory)
    backbone_lr: float = 1e-5      # very small if not frozen

    # ── Training ─────────────────────────────────────────────────────────────
    phase: str = "pretrain"        # "pretrain" | "online"
    pretrain_steps: int = 100_000
    online_steps: int = 50_000
    batch_size: int = 256          # number of (context, query) pairs per batch
    head_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005             # soft target network update
    grad_clip: float = 1.0

    # ── CQL (offline conservatism) ───────────────────────────────────────────
    cql_alpha: float = 1.0         # conservatism weight; set 0 to disable
    cql_n_random: int = 10         # random actions for CQL penalty

    # ── Replay Buffer ────────────────────────────────────────────────────────
    buffer_size: int = 1_000_000
    warmup_steps: int = 1000       # collect this many steps before TD updates

    # ── Logging / Checkpointing ──────────────────────────────────────────────
    log_every: int = 1000
    eval_every: int = 5000
    eval_episodes: int = 10
    save_dir: str = "runs"
    use_wandb: bool = False
    wandb_project: str = "tabrl"
    seed: int = 42

    # ── Hardware ─────────────────────────────────────────────────────────────
    device: str = "cuda"           # "cuda" | "cpu" | "mps"

    # D4RL environment → Minari dataset name mapping
    D4RL_TO_MINARI: dict = field(default_factory=lambda: {
        "hopper-medium-v2":              "mujoco/hopper/medium-v0",
        "hopper-medium-expert-v2":       "mujoco/hopper/expert-v0",
        "hopper-medium-replay-v2":       "mujoco/hopper/simple-v0",
        "halfcheetah-medium-v2":         "mujoco/halfcheetah/medium-v0",
        "halfcheetah-medium-expert-v2":  "mujoco/halfcheetah/expert-v0",
        "halfcheetah-medium-replay-v2":  "mujoco/halfcheetah/simple-v0",
        "walker2d-medium-v2":            "mujoco/walker2d/medium-v0",
        "walker2d-medium-expert-v2":     "mujoco/walker2d/expert-v0",
        "walker2d-medium-replay-v2":     "mujoco/walker2d/simple-v0",
    })

    # D4RL normalized score reference values (for reporting)
    D4RL_REF_SCORES: dict = field(default_factory=lambda: {
        "hopper-medium-v2":             {"random": 20.272, "expert": 3234.3},
        "hopper-medium-expert-v2":      {"random": 20.272, "expert": 3234.3},
        "hopper-medium-replay-v2":      {"random": 20.272, "expert": 3234.3},
        "halfcheetah-medium-v2":        {"random": -280.178, "expert": 12135.0},
        "halfcheetah-medium-expert-v2": {"random": -280.178, "expert": 12135.0},
        "halfcheetah-medium-replay-v2": {"random": -280.178, "expert": 12135.0},
        "walker2d-medium-v2":           {"random": 1.629, "expert": 4592.3},
        "walker2d-medium-expert-v2":    {"random": 1.629, "expert": 4592.3},
        "walker2d-medium-replay-v2":    {"random": 1.629, "expert": 4592.3},
    })
