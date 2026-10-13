#!/usr/bin/env bash
set -euo pipefail

# Submit the Isaac Sim SIF build to Slurm. The build itself runs in
# build_sif.sbatch so large image extraction does not happen on the login node.

SCRIPT_PATH="$(realpath "$0")"
if [[ -n "${ISAAC_SIM_ROOT_DIR:-}" ]]; then
  ROOT_DIR="$ISAAC_SIM_ROOT_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR="$SLURM_SUBMIT_DIR"
else
  ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
fi
mkdir -p "$ROOT_DIR/logs"

exec sbatch \
  --chdir="$ROOT_DIR" \
  --export="ALL,ISAAC_SIM_ROOT_DIR=$ROOT_DIR" \
  --output="$ROOT_DIR/logs/isaac-sim-build-%j.out" \
  --error="$ROOT_DIR/logs/isaac-sim-build-%j.err" \
  "$ROOT_DIR/docker/isaac_sim/build_sif.sbatch" "$@"
