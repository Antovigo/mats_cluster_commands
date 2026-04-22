#!/bin/bash

#SBATCH --job-name=resid_mlp3_targeted
#SBATCH --output=resid_mlp3_targeted_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00

echo "========================================================"
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"
echo

cd ~/SPD/spd

uv run python -m spd.scripts.run_variations \
  spd/experiments/resid_mlp/resid_mlp_decomposition.py \
  /mnt/nw/home/a.vigouroux/SPD/batch_commands/toy_models/resid_mlp3-targeted_config.yaml \
  /mnt/nw/home/a.vigouroux/SPD/batch_commands/toy_models/targeted_overrides.yaml

echo "Job finished at: $(date)"
