#!/bin/bash
#SBATCH --job-name=tabrl_pretrain_frozen
#SBATCH --output=logs/pretrain_frozen_%j.out
#SBATCH --error=logs/pretrain_frozen_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gpu

# ── Adjust these to match your cluster ──────────────────────────────────────
# Common partition names: gpu, gpu-high, compute, a100, v100
# Check available partitions with: sinfo
# Check available GPUs with: sinfo -o "%N %G"

# ── Environment setup ────────────────────────────────────────────────────────
echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn

# Move to project directory
cd ~/tabrl

# Create log directory
mkdir -p logs

# ── W&B setup ────────────────────────────────────────────────────────────────
# Option 1: set your API key directly (replace with your key)
# export WANDB_API_KEY=your_key_here

# Option 2: use offline mode if no internet on compute nodes
# export WANDB_MODE=offline

# Option 3: use W&B service (recommended for clusters)
export WANDB_PROJECT=tabrl

# ── Multi-GPU training with torchrun ─────────────────────────────────────────
# torchrun handles DDP process spawning automatically
# --nproc_per_node = number of GPUs per node
# --nnodes = number of nodes (1 for single node)

torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp.py \
    --phase pretrain \
    --freeze_backbone \
    --context_len 64 \
    --n_candidates 16 \
    --pretrain_steps 200000 \
    --batch_size 256 \
    --head_lr 3e-4 \
    --use_wandb \
    --save_dir runs/frozen_4gpu

echo "End time: $(date)"
echo "Job complete."
