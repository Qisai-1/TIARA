"""
Logger — CSV always, W&B optional.

Usage:
    # Enable W&B via flag
    python train.py --phase pretrain --freeze_backbone --use_wandb

    # Or set env var (useful for long runs)
    WANDB_PROJECT=tabrl python train.py --phase pretrain --freeze_backbone --use_wandb
"""

from __future__ import annotations
import os
import csv
from typing import Dict, Any, Optional


class Logger:
    def __init__(
        self,
        log_dir:  str,
        config,
        use_wandb: bool  = False,
        run_name:  Optional[str] = None,
    ):
        os.makedirs(log_dir, exist_ok=True)
        self.csv_path  = os.path.join(log_dir, "metrics.csv")
        self._writer   = None
        self._file     = None
        self.use_wandb = use_wandb
        self._wandb    = None

        if use_wandb:
            self._init_wandb(config, log_dir, run_name)

    def _init_wandb(self, config, log_dir: str, run_name: Optional[str]):
        try:
            import wandb
        except ImportError:
            print("[Logger] wandb not installed — pip install wandb")
            print("[Logger] Falling back to CSV only")
            self.use_wandb = False
            return

        # Build a clean config dict for W&B
        cfg_dict = {}
        if hasattr(config, "__dataclass_fields__"):
            # dataclass — skip non-serialisable fields
            for k in config.__dataclass_fields__:
                v = getattr(config, k)
                if isinstance(v, (str, int, float, bool)):
                    cfg_dict[k] = v
        elif hasattr(config, "__dict__"):
            cfg_dict = {
                k: v for k, v in vars(config).items()
                if isinstance(v, (str, int, float, bool))
            }

        # Auto-generate run name if not provided
        if run_name is None:
            run_name = (
                f"pretrain"
                f"_ctx{getattr(config, 'context_len', '?')}"
                f"_K{getattr(config, 'n_candidates', '?')}"
                f"_{getattr(config, 'proposal_type', 'gaussian')}"
                f"_{'frozen' if getattr(config, 'freeze_backbone', True) else 'finetune'}"
                f"_seed{getattr(config, 'seed', 42)}"
            )

        try:
            wandb.init(
                project  = getattr(config, "wandb_project", "tabrl"),
                name     = run_name,
                config   = cfg_dict,
                dir      = log_dir,
                # Resume if run crashed and restarted
                resume   = "allow",
                tags     = [
                    "pretrain",
                    f"ctx{getattr(config, 'context_len', '?')}",
                    f"K{getattr(config, 'n_candidates', '?')}",
                    "frozen" if getattr(config, "freeze_backbone", True) else "finetune",
                ],
            )
            self._wandb = wandb
            print(f"[Logger] W&B run: {wandb.run.name}  url: {wandb.run.url}")

        except Exception as e:
            print(f"[Logger] W&B init failed: {e}")
            print("[Logger] Falling back to CSV only")
            self.use_wandb = False

    def log(self, metrics: Dict[str, Any], step: int):
        metrics = dict(metrics)   # don't mutate caller's dict
        metrics["step"] = step

        # ── CSV ───────────────────────────────────────────────────────────────
        if self._writer is None:
            self._file   = open(self.csv_path, "w", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(metrics.keys())
            )
            self._writer.writeheader()
        try:
            self._writer.writerow(
                {k: metrics.get(k, "") for k in self._writer.fieldnames}
            )
            self._file.flush()
        except ValueError:
            # New keys appeared mid-run — reopen with updated fieldnames
            self._file.close()
            self._file   = open(self.csv_path, "a", newline="")
            self._writer = csv.DictWriter(
                self._file, fieldnames=list(metrics.keys())
            )

        # ── W&B ───────────────────────────────────────────────────────────────
        if self.use_wandb and self._wandb is not None:
            # Group metrics into sections for cleaner W&B dashboard
            grouped = {}
            for k, v in metrics.items():
                if k == "step":
                    continue
                if isinstance(v, (int, float)):
                    # Auto-group: "proposal/nll" → section "proposal"
                    grouped[k] = v
            try:
                self._wandb.log(grouped, step=step)
            except Exception as e:
                print(f"[Logger] W&B log error: {e}")

    def log_summary(self, metrics: Dict[str, Any]):
        """Log final summary metrics (shown prominently in W&B)."""
        if self.use_wandb and self._wandb is not None:
            for k, v in metrics.items():
                self._wandb.run.summary[k] = v

    def close(self):
        if self._file:
            self._file.close()
        if self.use_wandb and self._wandb is not None:
            self._wandb.finish()
