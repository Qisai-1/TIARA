"""
TabRL — Multi-GPU Training Entry Point (DDP)
=============================================
Uses PyTorch DistributedDataParallel (DDP) via torchrun.

Launch with:
    # Single node, 4 GPUs
    torchrun --standalone --nproc_per_node=4 train_ddp.py --phase pretrain --freeze_backbone

    # Or via SLURM sbatch (recommended):
    sbatch scripts/pretrain_frozen.sbatch

DDP strategy:
    - Each GPU gets a copy of the model
    - DataLoader uses DistributedSampler — each GPU sees different batches
    - Gradients are averaged across GPUs automatically
    - Effective batch size = batch_size × n_gpus
    - Only rank 0 logs to W&B and saves checkpoints

Note on TabPFN backbone with DDP:
    TabPFN's forward pass runs through a forward hook which captures
    intermediate tensors. This is compatible with DDP because:
    1. Frozen backbone: no gradients → no DDP sync needed for backbone
    2. Fine-tuning: DDP wraps the full agent, hook runs inside forward()
       which DDP handles correctly
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

# Fix import path — works from inside tabrl/ directory
_here   = os.path.dirname(os.path.abspath(__file__))
_parent = os.path.dirname(_here)
if _parent not in sys.path:
    sys.path.insert(0, _parent)
if _here not in sys.path:
    sys.path.insert(0, _here)

from tabrl.configs.base_config import TabRLConfig
from tabrl.data.d4rl_loader import (
    build_multi_env_dataloader,
    ALL_PRETRAIN_ENVS,
    MultiEnvDataset,
    ICLEnvDataset,
    EnvNormalizer,
    load_d4rl_dataset,
)
from tabrl.models.tabrl_agent import TabRLAgent
from tabrl.utils.logger import Logger
from tabrl.utils.normalizer import build_normalizers


# ── DDP helpers ───────────────────────────────────────────────────────────────

def setup_ddp():
    """Initialize the DDP process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main_process():
    """Returns True only on rank 0 (the process that logs/saves)."""
    return not dist.is_initialized() or dist.get_rank() == 0


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    config = TabRLConfig()
    parser = argparse.ArgumentParser(description="TabRL DDP training")

    parser.add_argument("--phase",          type=str, default="pretrain")
    parser.add_argument("--envs",           type=str, default="all")
    parser.add_argument("--context_len",    type=int, default=config.context_len)
    parser.add_argument("--n_candidates",   type=int, default=config.n_candidates)
    parser.add_argument("--proposal_type",  type=str, default=config.proposal_type)
    parser.add_argument("--shallow_value",  action="store_true")
    parser.add_argument("--cql_alpha",      type=float, default=config.cql_alpha)
    parser.add_argument("--freeze_backbone",    action="store_true", default=True)
    parser.add_argument("--no_freeze_backbone", dest="freeze_backbone",
                        action="store_false")
    parser.add_argument("--backbone_lr",    type=float, default=config.backbone_lr)
    parser.add_argument("--pretrain_steps", type=int,   default=config.pretrain_steps)
    parser.add_argument("--batch_size",     type=int,   default=config.batch_size)
    parser.add_argument("--head_lr",        type=float, default=config.head_lr)
    parser.add_argument("--save_dir",       type=str,   default="runs")
    parser.add_argument("--use_wandb",      action="store_true")
    parser.add_argument("--seed",           type=int,   default=42)

    args = parser.parse_args()

    config.phase           = args.phase
    config.context_len     = args.context_len
    config.n_candidates    = args.n_candidates
    config.proposal_type   = args.proposal_type
    config.shallow_value   = args.shallow_value
    config.cql_alpha       = args.cql_alpha
    config.freeze_backbone = args.freeze_backbone
    config.backbone_lr     = args.backbone_lr
    config.pretrain_steps  = args.pretrain_steps
    config.batch_size      = args.batch_size
    config.head_lr         = args.head_lr
    config.save_dir        = args.save_dir
    config.use_wandb       = args.use_wandb
    config.seed            = args.seed
    config.device          = "cuda"

    return config, args


# ── DDP DataLoader ────────────────────────────────────────────────────────────

def build_ddp_dataloader(
    minari_ids:  list,
    config,
    rank:        int,
    world_size:  int,
) -> tuple:
    """
    Build a DataLoader with DistributedSampler so each GPU
    processes a disjoint subset of the data each step.

    Effective batch size = config.batch_size × world_size
    """
    from tabrl.data.d4rl_loader import (
        load_d4rl_dataset, EnvNormalizer, ICLEnvDataset, MultiEnvDataset
    )

    if rank == 0:
        print(f"\n[DDP] Building dataset on rank 0 ...")

    all_data    = []
    normalizers = []
    env_names   = []

    for mid in minari_ids:
        data = load_d4rl_dataset(mid)
        norm = EnvNormalizer(data)
        all_data.append(data)
        normalizers.append(norm)
        env_names.append(mid.split("/")[-2] + "-" + mid.split("/")[-1])

    max_obs_dim = max(n.obs_dim for n in normalizers)
    max_act_dim = max(n.act_dim for n in normalizers)

    env_datasets = [
        ICLEnvDataset(
            data=data, normalizer=norm,
            context_len=config.context_len,
            max_obs_dim=max_obs_dim,
            max_act_dim=max_act_dim,
            gamma=0.99,
        )
        for data, norm in zip(all_data, normalizers)
    ]

    dataset = MultiEnvDataset(env_datasets, env_names)

    # DistributedSampler ensures each GPU gets different data
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,     # per-GPU batch size
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    if rank == 0:
        print(f"[DDP] Dataset size: {len(dataset):,}")
        print(f"[DDP] Per-GPU batch: {config.batch_size}")
        print(f"[DDP] Effective batch: {config.batch_size * world_size}")
        print(f"[DDP] Padded feature_dim: {max_obs_dim + max_act_dim}")

    return dataloader, normalizers, env_names, max_obs_dim, max_act_dim, sampler


# ── DDP Training loop ─────────────────────────────────────────────────────────

def train_ddp(config, args):
    # Setup DDP
    local_rank = setup_ddp()
    world_size = dist.get_world_size()
    rank       = dist.get_rank()
    main       = (rank == 0)

    # Seed each rank differently for data diversity
    seed = config.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if main:
        print(f"\n[DDP] World size: {world_size} GPUs")
        print(f"[DDP] Effective batch size: {config.batch_size * world_size}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    minari_ids = ALL_PRETRAIN_ENVS if args.envs == "all" else args.envs.split(",")

    dataloader, normalizers, env_names, max_obs_dim, max_act_dim, sampler = \
        build_ddp_dataloader(minari_ids, config, rank, world_size)

    # ── Build agent ───────────────────────────────────────────────────────────
    config.device = f"cuda:{local_rank}"
    agent = TabRLAgent(config, obs_dim=max_obs_dim, act_dim=max_act_dim)
    agent = agent.to(local_rank)

    # Wrap with DDP — find_unused_parameters=True because frozen backbone
    # params don't produce gradients
    ddp_agent = DDP(
        agent,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Access underlying module for parameter groups
    raw_agent = ddp_agent.module
    head_params = (
        list(raw_agent.proposal_head.parameters()) +
        list(raw_agent.value_head.parameters())
    )
    if config.freeze_backbone:
        optimizer = torch.optim.AdamW(
            head_params, lr=config.head_lr, weight_decay=1e-4
        )
    else:
        backbone_params = [
            p for p in raw_agent.backbone.parameters() if p.requires_grad
        ]
        optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": config.backbone_lr},
            {"params": head_params,     "lr": config.head_lr},
        ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.pretrain_steps, eta_min=1e-6
    )

    # ── Logger (rank 0 only) ──────────────────────────────────────────────────
    save_dir = os.path.join(
        config.save_dir,
        f"ddp{world_size}gpu_ctx{config.context_len}_K{config.n_candidates}"
        f"_{'frozen' if config.freeze_backbone else 'finetune'}"
        f"_seed{config.seed}"
    )

    logger = None
    if main:
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        logger = Logger(
            save_dir, config,
            use_wandb=config.use_wandb,
            run_name=os.path.basename(save_dir),
        )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_loss = float("inf")
    best_path = os.path.join(save_dir, "pretrain_best.pt") if main else ""

    data_iter = iter(dataloader)
    t0        = time.time()
    running   = {}

    if main:
        print(f"\n{'='*60}")
        print(f"  DDP Pretraining: {config.pretrain_steps:,} steps")
        print(f"  GPUs: {world_size}  Local rank: {local_rank}")
        print(f"  Save dir: {save_dir}")
        print(f"{'='*60}\n")

    for step in range(1, config.pretrain_steps + 1):

        # Shuffle sampler at each epoch boundary
        if step % len(dataloader) == 1:
            sampler.set_epoch(step // len(dataloader))

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            sampler.set_epoch(step // len(dataloader))
            batch = next(data_iter)

        # Move batch to local GPU
        batch = {
            k: v.to(local_rank) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        optimizer.zero_grad(set_to_none=True)

        # Use raw_agent for bc_step/td_step (DDP wraps forward only)
        bc_loss, bc_info = raw_agent.bc_step(batch)
        td_loss, td_info = raw_agent.td_step(batch)
        total_loss       = bc_loss + td_loss

        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp_agent.parameters(), config.grad_clip
        ).item()
        optimizer.step()
        scheduler.step()
        raw_agent.soft_update_target()

        # EMA running averages
        alpha = 0.02
        for k, v in {**bc_info, **td_info,
                     "train/total_loss": total_loss.item(),
                     "train/bc_loss":    bc_loss.item(),
                     "train/grad_norm":  grad_norm}.items():
            running[k] = running.get(k, v) * (1 - alpha) + v * alpha

        # Logging — rank 0 only
        if main and step % config.log_every == 0:
            elapsed = time.time() - t0
            lr_now  = scheduler.get_last_lr()[-1]
            sps     = config.log_every / elapsed

            metrics = {
                "train/total_loss":    total_loss.item(),
                "train/bc_loss":       bc_loss.item(),
                "train/td_loss":       td_info["value/td_loss"],
                "train/grad_norm":     grad_norm,
                "train/lr":            lr_now,
                "train/effective_batch": config.batch_size * world_size,
                "proposal/sigma_mean": bc_info.get("proposal/sigma_mean", 0),
                "proposal/mu_mean":    bc_info.get("proposal/mu_mean", 0),
                "proposal/nll":        bc_info.get("proposal/nll", 0),
                "value/q_pred":        td_info["value/q_pred"],
                "value/q_target":      td_info["value/q_target"],
                "value/td_loss":       td_info["value/td_loss"],
                "value/cql_penalty":   td_info.get("value/cql_penalty", 0),
                "smooth/total_loss":   running.get("train/total_loss", 0),
                "smooth/bc_loss":      running.get("train/bc_loss", 0),
                "smooth/sigma":        running.get("proposal/sigma_mean", 0),
                "smooth/q_pred":       running.get("value/q_pred", 0),
                "throughput/steps_per_sec": sps,
                "throughput/pct_complete":  100 * step / config.pretrain_steps,
                "throughput/gpu_mem_gb": torch.cuda.memory_allocated() / 1e9,
            }

            logger.log(metrics, step=step)
            print(
                f"[{step:>7d}/{config.pretrain_steps}] "
                f"loss={total_loss.item():.4f}  "
                f"bc={bc_loss.item():.4f}  "
                f"td={td_info['value/td_loss']:.4f}  "
                f"Q={td_info['value/q_pred']:.3f}  "
                f"σ={bc_info.get('proposal/sigma_mean',0):.3f}  "
                f"lr={lr_now:.2e}  "
                f"({sps:.1f} it/s × {world_size} GPU)"
            )
            t0 = time.time()

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                raw_agent.save(best_path)
                logger.log_summary({"best_loss": best_loss, "best_step": step})

        # Periodic checkpoint — rank 0 only
        if main and step % max(1, config.pretrain_steps // 10) == 0:
            ckpt = os.path.join(save_dir, f"step{step}.pt")
            raw_agent.save(ckpt)
            print(f"  → Checkpoint: {ckpt}")

        # Synchronise all ranks at checkpoint steps
        dist.barrier()

    if main:
        print(f"\n[DDP] Done. Best loss: {best_loss:.4f}")
        logger.close()

    cleanup_ddp()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config, args = parse_args()
    train_ddp(config, args)
