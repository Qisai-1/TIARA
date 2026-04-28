#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=120G
#SBATCH --job-name="tabrl_frozen_v2"
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output="/work/mech-ai-scratch/supersai/TIARA/logs/frozen_v2_%j.out"
#SBATCH --error="/work/mech-ai-scratch/supersai/TIARA/logs/frozen_v2_%j.err"

TIARA=/work/mech-ai-scratch/supersai/TIARA
cd $TIARA
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn
source $TIARA/activate_tabrl.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# v2 — key changes vs v1:
#   td_weight: 0.01 → 0.001   (TD loss 10x smaller)
#   cql_alpha: 1.0  → 0.0     (disable CQL entirely — value head trains on TD only)
#   This gives the value head the best chance to stabilize before adding CQL back

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
    --cql_alpha 0.0 \
    --cql_n_random 0 \
    --td_weight 0.001 \
    --use_wandb \
    --seed 42 \
    --save_dir $TIARA/frozen_4gpu_h200_v2

echo "End time: $(date)"
