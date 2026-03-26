#!/bin/bash

#SBATCH --job-name=css_flat_pgd
#SBATCH --output=css_flat_pgd_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
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

cd ~/SPD/spd_alt

# uv run python -m torch.distributed.run \
#   --standalone \
#   --nproc_per_node 2 \
#   spd/experiments/lm/lm_decomposition.py \
#   /mnt/nw/home/a.vigouroux/SPD/batch_commands/css_targeted/config_flat.yaml

uv run spd/experiments/lm/lm_decomposition.py /mnt/nw/home/a.vigouroux/SPD/batch_commands/css_targeted/config_flat_PGD.yaml

echo "Job finished at: $(date)"
