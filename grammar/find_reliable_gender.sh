#!/bin/bash

#SBATCH --job-name=gender_pronouns
#SBATCH --output=gender_pronouns_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00

echo "========================================================"
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"
echo

cd ~/SPD/spd_alt

uv run python -m spd.scripts.find_reliable_patterns \
  --config-path spd/experiments/lm/pile_llama_simple_mlp-4L.yaml \
  --tokens he she his her him himself herself \
  --batch-size 16 \
  --n-batches 5000 \
  --thr 0.5 \
  --output /mnt/nw/home/a.vigouroux/SPD/batch_commands/gender/gender_pronouns.tsv

echo "Job finished at: $(date)"
