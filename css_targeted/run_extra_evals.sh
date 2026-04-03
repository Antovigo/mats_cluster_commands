#!/bin/bash
#SBATCH --job-name=css-stress-test
#SBATCH --partition=h200-reserved
#SBATCH --gpus-per-node=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/polished-lake/artifacts/mechanisms/spd/slurm_logs/slurm-%j.out

#MODEL_PATH=~/spd_out/spd/s-e790005d/model_50000.pth
MODEL_PATH=~/Documents/MATS/spd_out/spd/s-e790005d/model_50000.pth

cd ~/Code/SPD/spd_alt

uv run python -m spd.scripts.decomposition_stress_test.recon_distribution "$MODEL_PATH" \
    --n-batches 512 --batch-size 2

# uv run python -m spd.scripts.decomposition_stress_test.per_matrix_recon "$MODEL_PATH" \
#     --n-batches 256 --batch-size 2

uv run python -m spd.scripts.decomposition_stress_test.entropy_vs_metrics "$MODEL_PATH" \
    --n-batches 128 --batch-size 2
