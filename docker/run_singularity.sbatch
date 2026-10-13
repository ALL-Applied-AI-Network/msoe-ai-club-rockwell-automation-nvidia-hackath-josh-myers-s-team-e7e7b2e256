#!/usr/bin/env bash

#SBATCH --job-name=innovation-lab
# Rosie teaching-node T4 allocation.
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=16G
#SBATCH --output=innovation-lab-%j.out
#SBATCH --error=innovation-lab-%j.err

set -euo pipefail
ROOT_DIR="$(pwd -P)"
cd "$ROOT_DIR"
mkdir -p logs
if [[ $# -eq 0 ]]; then
  COMMAND=(uv run python main.py)
else
  COMMAND=("$@")
fi
USE_GPU=1 exec srun --ntasks=1 docker/run_singularity.sh "${COMMAND[@]}"
