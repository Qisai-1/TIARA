#!/bin/bash
#SBATCH --job-name=tabrl_finetune
#SBATCH --output=logs/finetune_%j.out
#SBATCH --error=logs/finetune_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --partition=gpu

# Full fine-tuning needs more time and memory than frozen backbone
# TabPFN has 10M params — recommend A100 40GB or better per GPU

echo "Job ID:     $SLURM_JOB_ID"
echo "Node:       $SLURMD_NODENAME"
echo "GPUs:       $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn

cd ~/tabrl
mkdir -p logs

export WANDB_PROJECT=tabrl

# Full fine-tuning: backbone_lr is much smaller than head_lr
# to avoid destroying TabPFN's pretrained representations
torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp.py \
    --phase pretrain \
    --no_freeze_backbone \
    --context_len 64 \
    --n_candidates 16 \
    --pretrain_steps 200000 \
    --batch_size 128 \
    --head_lr 3e-4 \
    --backbone_lr 1e-6 \
    --use_wandb \
    --save_dir runs/finetune_4gpu

echo "End time: $(date)"
echo "Job complete."
