#!/bin/bash
#SBATCH --time=48:00:00                 # walltime limit (HH:MM:SS)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gres=gpu:h200:4              # 4x A100 80GB GPUs
#SBATCH --cpus-per-task=4              # 4 cores per task = 16 cores total
#SBATCH --ntasks=4                      # 1 task per GPU, 4 GPUs total
#SBATCH --mem=180G                      # maximum memory per node
#SBATCH --job-name="tabrl_finetune"    # job name
#SBATCH --account=mech-ai              # Slurm account to use for the job
#SBATCH --mail-user=supersai@iastate.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/finetune_%j.out"
#SBATCH --error="logs/finetune_%j.err"

# Full fine-tuning needs more time and memory than frozen backbone
# TabPFN has 10M params — recommend A100 40GB or better per GPU

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

cd ~/tabrl
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn


export WANDB_PROJECT=tabrl

# Confirm GPUs are visible before training starts
echo "GPUs available:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp.py \
    --phase pretrain \
    --no_freeze_backbone \
    --context_len 64 \
    --n_candidates 16 \
    --pretrain_steps 300000 \
    --batch_size 128 \
    --head_lr 1e-4 \
    --backbone_lr 1e-6 \
    --use_wandb \
    --seed 42 \
    --save_dir runs/finetune_4gpu_h200

echo "End time: $(date)"
echo "Job complete."