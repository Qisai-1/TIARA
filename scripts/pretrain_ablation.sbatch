#!/bin/bash
#SBATCH --job-name=tabrl_ablation
#SBATCH --output=logs/ablation_%A_%a.out
#SBATCH --error=logs/ablation_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --array=0-7

# Job array: runs 8 ablations in parallel, each on 1 GPU
# Submit with: sbatch pretrain_ablation.sbatch
# Monitor with: squeue -u $USER

echo "Job ID:      $SLURM_JOB_ID"
echo "Array task:  $SLURM_ARRAY_TASK_ID"
echo "Start time:  $(date)"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn

cd ~/tabrl
mkdir -p logs

export WANDB_PROJECT=tabrl

# ── Ablation configurations ───────────────────────────────────────────────────
# Each array task runs one ablation
# Array index → config

case $SLURM_ARRAY_TASK_ID in
    # Context length ablation
    0) CTX=0;   K=16; PROP=gaussian; SHALLOW="";            NAME="ctx0_K16" ;;
    1) CTX=16;  K=16; PROP=gaussian; SHALLOW="";            NAME="ctx16_K16" ;;
    2) CTX=64;  K=16; PROP=gaussian; SHALLOW="";            NAME="ctx64_K16" ;;
    3) CTX=128; K=16; PROP=gaussian; SHALLOW="";            NAME="ctx128_K16" ;;

    # Candidate count ablation
    4) CTX=64;  K=4;  PROP=gaussian; SHALLOW="";            NAME="ctx64_K4" ;;
    5) CTX=64;  K=64; PROP=gaussian; SHALLOW="";            NAME="ctx64_K64" ;;

    # Proposal mechanism ablation
    6) CTX=64;  K=16; PROP=mlp;     SHALLOW="";             NAME="ctx64_mlp" ;;

    # Design X ablation (no backbone re-encoding)
    7) CTX=64;  K=16; PROP=gaussian; SHALLOW="--shallow_value"; NAME="ctx64_shallowV" ;;
esac

echo "Running ablation: $NAME"
echo "  context_len=$CTX  n_candidates=$K  proposal=$PROP  shallow=$SHALLOW"

PYTHONPATH=.. python train.py \
    --phase pretrain \
    --freeze_backbone \
    --context_len $CTX \
    --n_candidates $K \
    --proposal_type $PROP \
    $SHALLOW \
    --pretrain_steps 100000 \
    --batch_size 256 \
    --use_wandb \
    --save_dir runs/ablation_$NAME \
    --seed $SLURM_ARRAY_TASK_ID

echo "End time: $(date)"
