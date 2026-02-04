#!/bin/bash

#SBATCH --job-name=spd   # Name for your job
#SBATCH --output=run_output_%j.log # Log file for stdout/stderr (%j = Job ID)
#SBATCH --partition=compute           # The partition is always 'compute'
#SBATCH --nodes=1                     # Request one node
#SBATCH --ntasks=1                    # Request one task 
#SBATCH --cpus-per-task=4             # Request 4 CPUs (always a good practice)
#SBATCH --mem=32GB                    # Request 16GB of memory
#SBATCH --gres=gpu:1                  # Request 1 L40 GPU
#SBATCH --time=04:00:00               # Request 1 hour runtime (format: HH:MM:SS or D-HH:MM:SS)

# --- To use the debug QoS, uncomment the line below ---
# -- It has a 2-hour time limit and allows max 1 GPU.
##SBATCH --qos=debug

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

# Load any necessary modules (e.g., conda, cuda)
# module load cuda/12.2

# echo "GPU info:"
# nvidia-smi

cd ~/spd

# Pythia transformer training
# uv run spd/experiments/mem/train_mem.py

# Pythia transformer decomposition
uv run spd/experiments/lm/lm_decomposition.py spd/experiments/lm/pythia_70m_targeted_config.yaml

# Sweep
#uv run spd/scripts/run_variations.py spd/experiments/lm/lm_decomposition.py spd/experiments/lm/pythia_70m_targeted_config.yaml spd/experiments/lm/experimental_plan.yaml

echo "Job finished at: $(date)"

