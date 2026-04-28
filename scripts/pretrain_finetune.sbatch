#!/bin/bash
#SBATCH --time=96:00:00                 # walltime limit (HH:MM:SS)
#SBATCH --nodes=1                       # number of nodes
#SBATCH --gres=gpu:a100:4              # 4x A100 80GB GPUs
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

TIARA=/work/mech-ai-scratch/supersai/TIARA
cd $TIARA
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn

source /work/mech-ai-scratch/supersai/TIARA/activate_tabrl.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Warm-start from frozen-backbone checkpoint ────────────────────────────────
# Set this to the pretrain_best.pt produced by pretrain_frozen.sbatch
FROZEN_CKPT=$TIARA/frozen_4gpu_h200/ddp4gpu_ctx32_K4_frozen_seed42/pretrain_best.pt

if [ ! -f "$FROZEN_CKPT" ]; then
    echo "ERROR: frozen checkpoint not found at $FROZEN_CKPT"
    echo "Run pretrain_frozen.sbatch first, then update FROZEN_CKPT above."
    exit 1
fi

echo "Frozen checkpoint: $FROZEN_CKPT"
echo ""

# Confirm GPUs are visible before training starts
echo "GPUs available:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ── Fine-tuning: backbone unfrozen, heads warm-started from frozen run ─────────
# Backbone LR is very small (1e-6) to avoid destroying pretrained representations.
# Head LR is reduced (1e-4) since heads are already partially trained.
# Batch size reduced (32) to fit backbone gradients on GPU.

torchrun \
    --standalone \
    --nproc_per_node=4 \
    --nnodes=1 \
    train_ddp.py \
    --phase pretrain \
    --no_freeze_backbone \
    --warmstart $FROZEN_CKPT \
    --context_len 32 \
    --n_candidates 4 \
    --pretrain_steps 300000 \
    --batch_size 32 \
    --head_lr 1e-4 \
    --backbone_lr 1e-7 \
    --td_weight 0.001 \
    --cql_alpha 0.0 \
    --cql_n_random 3 \
    --use_wandb \
    --seed 42 \
    --save_dir $TIARA/runs/finetune_4gpu_h200

echo ""
echo "=========================================="
echo "  Job complete:  $(date)"
echo "  Checkpoint at: $TIARA/runs/finetune_4gpu_h200/"
echo "=========================================="