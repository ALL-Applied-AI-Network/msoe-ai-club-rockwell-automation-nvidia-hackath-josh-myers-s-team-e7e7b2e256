#!/usr/bin/env bash

#SBATCH --job-name=problem2-render
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$ROOT_DIR/logs"
  exec sbatch --wait --chdir="$ROOT_DIR" \
    --output="$ROOT_DIR/logs/problem2-%j.out" \
    --error="$ROOT_DIR/logs/problem2-%j.err" \
    "$0" "$@"
fi

TWIN_CAMERA="${TWIN_CAMERA:-FillCam}"
TWIN_STAGE="$ROOT_DIR/data/csi_digital_twin_v1/stations/fill_station/fill_station.usda"
TWIN_FRAME="${TWIN_FRAME:-686}"
TWIN_OUTPUT="${TWIN_OUTPUT:-$ROOT_DIR/outputs/problem2_${TWIN_CAMERA,,}.png}"
export TWIN_CAMERA TWIN_STAGE TWIN_FRAME TWIN_OUTPUT
exec "$SCRIPT_DIR/../run_csi_twin.sh" "$@"

