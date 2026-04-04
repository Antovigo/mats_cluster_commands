#!/bin/bash
#
#SBATCH --job-name=hybridize
#SBATCH --output=hybridize_%j.log
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00

MODEL_PATH=~/spd_out/spd/s-e790005d/model_50000.pth

cd ~/SPD/spd_alt

uv run python -m spd.scripts.decomposition_stress_test.hybridize_matrices "$MODEL_PATH" \
    --n-batches 1 --batch-size 256 --n-masks 32
