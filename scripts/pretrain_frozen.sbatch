#!/bin/bash
#SBATCH --time=72:00:00                 # walltime limit (HH:MM:SS)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gres=gpu:a100:4              # 4x A100 80GB GPUs
#SBATCH --cpus-per-task=4              # 4 cores per task = 16 cores total
#SBATCH --ntasks=4                      # 1 task per GPU, 4 GPUs total
#SBATCH --mem=120G                      # maximum memory per node
#SBATCH --job-name="tabrl_frozen"    # job name
#SBATCH --account=mech-ai              # Slurm account to use for the job
#SBATCH --mail-user=supersai@iastate.edu  # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --output="logs/frozen_%j.out"
#SBATCH --error="logs/frozen_%j.err"


echo "=========================================="
echo "  TabRL — Frozen Backbone Pretraining"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $SLURMD_NODENAME"
echo "  Start:     $(date)"
echo "=========================================="

# Confirm GPUs
echo ""
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

TIARA=/work/mech-ai-scratch/supersai/TIARA
cd $TIARA
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn

source /work/mech-ai-scratch/supersai/TIARA/activate_tabrl.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Frozen backbone ───────────────────────────────────────────────────────────
# Only proposal head + value head train (397K parameters)
# TabPFN weights completely frozen
# Fast: ~8 hours on 4x H200
# This is the BASELINE for the paper — establishes what pretrained
# TabPFN ICL gives us for free before any RL-specific fine-tuning

torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp.py \
    --phase pretrain \
    --freeze_backbone \
    --context_len 32 \
    --n_candidates 4 \
    --pretrain_steps 100000 \
    --batch_size 256 \
    --head_lr 3e-4 \
    --cql_alpha 1.0 \
    --cql_n_random 3 \
    --use_wandb \
    --seed 42 \
    --save_dir $TIARA/frozen_4gpu_h200


echo ""
echo "=========================================="
echo "  Job complete:  $(date)"
echo "  Checkpoint at: $TIARA/frozen_4gpu_A100/"
echo "=========================================="