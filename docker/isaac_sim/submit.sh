#!/usr/bin/env bash

#SBATCH --job-name=isaac-sim
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1
#SBATCH --output=logs/isaac-sim-%j.out
#SBATCH --error=logs/isaac-sim-%j.err

set -euo pipefail

# Batch entrypoint for the shared standalone Isaac Sim launcher.
SCRIPT_PATH="$(realpath "$0")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs

[[ -n "${SLURM_JOB_ID:-}" ]] || {
  echo "Submit this script with: sbatch $SCRIPT_PATH [python-script] [args...]" >&2
  exit 2
}

exec "$ROOT_DIR/docker/isaac_sim/run.sh" "$@"
