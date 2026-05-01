#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --job-name="tabrl_eval"
#SBATCH --account=mech-ai
#SBATCH --partition=nova
#SBATCH --mail-user=supersai@iastate.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --output="/work/mech-ai-scratch/supersai/TIARA/logs/eval_finetune_%j.out"
#SBATCH --error="/work/mech-ai-scratch/supersai/TIARA/logs/eval_finetune_%j.err"

TIARA=/work/mech-ai-scratch/supersai/TIARA
cd $TIARA
mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate dqn
source $TIARA/activate_tabrl.sh

CHECKPOINT="$TIARA/runs/finetune_4gpu_h200/ddp4gpu_ctx32_K4_finetune_seed42/pretrain_best.pt"

echo "Evaluating checkpoint: $CHECKPOINT"
echo "Start time: $(date)"
echo ""

for ENV in hopper-medium-v2 hopper-medium-expert-v2 hopper-medium-replay-v2 \
           halfcheetah-medium-v2 halfcheetah-medium-expert-v2 halfcheetah-medium-replay-v2 \
           walker2d-medium-v2 walker2d-medium-expert-v2 walker2d-medium-replay-v2; do
    echo "=== $ENV ==="
    PYTHONPATH=/work/mech-ai-scratch/supersai python train.py \
        --phase eval \
        --checkpoint $CHECKPOINT \
        --eval_env $ENV \
        --device cpu
    echo ""
done 2>&1 | tee $TIARA/logs/eval_finetune_results.txt

echo "End time: $(date)"
echo "Results saved to: $TIARA/logs/eval_finetune_results.txt"
