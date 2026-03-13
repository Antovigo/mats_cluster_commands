#!/bin/bash

#SBATCH --job-name=val_ablations
#SBATCH --output=val_ablations_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00

echo "========================================================"
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"
echo

cd ~/SPD/spd_alt

uv run python -m spd.scripts.decomposition_stress_test.validate_ablations \
  ~/spd_out/spd/s-2b9c00f0/model_30000.pth \
  --prompts ~/SPD/spd/spd/experiments/lm/prompts/import_numpy_and_pandas.txt \
  --n-batches 10 \
  --device cuda

echo "Job finished at: $(date)"
