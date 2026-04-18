"""
Offline Pretraining — Phase 1.

Trains proposal head (BC) and value head (CQL-style TD) on the
multi-environment D4RL dataset. Backbone is either frozen or fine-tuned.

W&B dashboard sections:
    train/          — per-step losses and learning rate
    proposal/       — proposal head diagnostics (sigma, mu magnitude)
    value/          — value head diagnostics (Q values, CQL penalty)
    throughput/     — steps per second, GPU memory
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from ..models.tabrl_agent import TabRLAgent
from ..utils.logger import Logger


def build_optimizer(agent: TabRLAgent, config) -> torch.optim.Optimizer:
    head_params = (
        list(agent.proposal_head.parameters()) +
        list(agent.value_head.parameters())
    )
    if config.freeze_backbone:
        return torch.optim.AdamW(
            head_params, lr=config.head_lr, weight_decay=1e-4
        )
    else:
        backbone_params = [
            p for p in agent.backbone.parameters() if p.requires_grad
        ]
        return torch.optim.AdamW([
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params,     "lr": config.head_lr},
        ], weight_decay=1e-4)


def pretrain(
    agent:      TabRLAgent,
    dataloader: DataLoader,
    config,
    logger:     Logger,
    save_dir:   str,
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    optimizer = build_optimizer(agent, config)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pretrain_steps, eta_min=1e-6
    )

    agent.train()
    step      = 0
    best_loss = float("inf")
    best_path = os.path.join(save_dir, "pretrain_best.pt")

    data_iter = iter(dataloader)
    t0        = time.time()

    # Track running averages for smoother W&B curves
    running = {}

    print(f"\n{'='*60}")
    print(f"  Offline Pretraining: {config.pretrain_steps:,} steps")
    print(f"  Context len:  {config.context_len}")
    print(f"  Candidates K: {config.n_candidates}")
    print(f"  Backbone:     {'frozen' if config.freeze_backbone else 'fine-tuning'}")
    print(f"  CQL alpha:    {config.cql_alpha}")
    print(f"  Logging to:   {save_dir}/metrics.csv")
    if logger.use_wandb:
        print(f"  W&B:          enabled")
    print(f"{'='*60}\n")

    while step < config.pretrain_steps:

        # ── Get batch ─────────────────────────────────────────────────────────
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch     = next(data_iter)

        # ── Forward ───────────────────────────────────────────────────────────
        optimizer.zero_grad(set_to_none=True)

        bc_loss, bc_info = agent.bc_step(batch)
        td_loss, td_info = agent.td_step(batch)
        total_loss       = bc_loss + td_loss

        # ── Backward ──────────────────────────────────────────────────────────
        total_loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(
            agent.parameters(), config.grad_clip
        ).item()
        optimizer.step()
        scheduler.step()
        agent.soft_update_target()

        step += 1

        # ── Update running averages ───────────────────────────────────────────
        alpha = 0.02   # EMA smoothing
        for k, v in {**bc_info, **td_info,
                     "train/total_loss": total_loss.item(),
                     "train/bc_loss":    bc_loss.item(),
                     "train/grad_norm":  grad_norm}.items():
            running[k] = running.get(k, v) * (1 - alpha) + v * alpha

        # ── Logging ───────────────────────────────────────────────────────────
        if step % config.log_every == 0:
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[-1]
            sps     = config.log_every / elapsed   # steps per second

            # Build full metrics dict
            metrics = {
                # Training losses
                "train/total_loss":    total_loss.item(),
                "train/bc_loss":       bc_loss.item(),
                "train/td_loss":       td_info["value/td_loss"],
                "train/grad_norm":     grad_norm,
                "train/lr":            lr_now,

                # Proposal head diagnostics
                "proposal/sigma_mean": bc_info.get("proposal/sigma_mean", 0),
                "proposal/mu_mean":    bc_info.get("proposal/mu_mean", 0),
                "proposal/nll":        bc_info.get("proposal/nll", 0),

                # Value head diagnostics
                "value/q_pred":        td_info["value/q_pred"],
                "value/q_target":      td_info["value/q_target"],
                "value/td_loss":       td_info["value/td_loss"],
                "value/cql_penalty":   td_info.get("value/cql_penalty", 0),

                # Smoothed versions (less noisy in W&B)
                "smooth/total_loss":   running.get("train/total_loss", 0),
                "smooth/bc_loss":      running.get("train/bc_loss", 0),
                "smooth/sigma":        running.get("proposal/sigma_mean", 0),
                "smooth/q_pred":       running.get("value/q_pred", 0),

                # Throughput
                "throughput/steps_per_sec": sps,
                "throughput/step":          step,
                "throughput/pct_complete":  100 * step / config.pretrain_steps,
            }

            # Add GPU memory if available
            if torch.cuda.is_available():
                metrics["throughput/gpu_mem_gb"] = (
                    torch.cuda.memory_allocated() / 1e9
                )

            logger.log(metrics, step=step)

            # Terminal print (concise)
            print(
                f"[{step:>7d}/{config.pretrain_steps}] "
                f"loss={total_loss.item():.4f}  "
                f"bc={bc_loss.item():.4f}  "
                f"td={td_info['value/td_loss']:.4f}  "
                f"Q={td_info['value/q_pred']:.3f}  "
                f"σ={bc_info.get('proposal/sigma_mean', 0):.3f}  "
                f"lr={lr_now:.2e}  "
                f"({sps:.1f} it/s)"
            )
            t0 = time.time()

            # Save best checkpoint
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                agent.save(best_path)
                if logger.use_wandb:
                    logger.log_summary({
                        "best_loss": best_loss,
                        "best_step": step,
                    })

        # ── Periodic checkpoint ───────────────────────────────────────────────
        if step % max(1, config.pretrain_steps // 10) == 0:
            ckpt = os.path.join(save_dir, f"pretrain_step{step}.pt")
            agent.save(ckpt)
            print(f"  → Checkpoint: {ckpt}")

    print(f"\n[Pretrain] Done. Best loss: {best_loss:.4f}")
    print(f"[Pretrain] Best checkpoint: {best_path}")

    logger.log_summary({"final_best_loss": best_loss})
    return best_path