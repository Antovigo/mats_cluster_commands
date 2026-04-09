#!/bin/bash

#SBATCH --job-name=p_anneal_1.6+   # Name for your job
#SBATCH --output=p_anneal_1.6+_%j.log  # Log file for stdout/stderr (%j = Job ID)
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=32GB                    # Request 32GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=24:00:00               # Request 24 hour runtime

# --- Your job commands start here ---
echo "========================================================"
echo "Job Started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on host: $(hostname)"
echo "Partition: $SLURM_JOB_PARTITION"
echo "QoS: $SLURM_JOB_QOS"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Allocated GPU(s): $CUDA_VISIBLE_DEVICES"
echo "========================================================"
echo

cd ~/SPD/spd_alt

uv run spd/scripts/run_variations.py \
  spd/experiments/lm/lm_decomposition.py \
  /mnt/nw/home/a.vigouroux/SPD/batch_commands/numpy_impmin_p_annealing/pile_llama_targeted.yaml \
  /mnt/nw/home/a.vigouroux/SPD/batch_commands/numpy_impmin_p_annealing/experimental_plan_coeff_1.6e-3_plus.yaml

echo "Job finished at: $(date)"
