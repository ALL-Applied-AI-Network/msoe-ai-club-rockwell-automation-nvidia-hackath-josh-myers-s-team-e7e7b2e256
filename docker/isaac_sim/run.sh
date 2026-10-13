#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
ROOT_DIR="${ISAAC_SIM_ROOT_DIR:-$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
ISAAC_SIM_SIF="${ISAAC_SIM_SIF:-$ROOT_DIR/isaac-sim-6.0.0.sif}"
ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON_IN_CONTAINER:-/isaac-sim/python.sh}"
ISAAC_SIM_SHELL="${ISAAC_SIM_SHELL_IN_CONTAINER:-/bin/bash}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  SRUN_ARGS=(
    --partition=teaching \
    --gres=gpu:t4:1 \
    --cpus-per-gpu=8 \
    --chdir="$ROOT_DIR" \
    --ntasks=1
  )
  [[ "${1:-}" == shell ]] && SRUN_ARGS+=(--pty)
  exec srun "${SRUN_ARGS[@]}" "$SCRIPT_PATH" "$@"
fi

if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else echo "Singularity or Apptainer is required." >&2; exit 1; fi
[[ -f "$ISAAC_SIM_SIF" ]] || { echo "Isaac Sim SIF not found: $ISAAC_SIM_SIF" >&2; exit 1; }

BIND_ARGS=(-B "$ROOT_DIR:/workspace")
[[ -d /data ]] && BIND_ARGS+=(-B /data:/data)
ENV_ARGS=(--env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y)

if [[ "${1:-}" == shell ]]; then
  exec "$RUNTIME" shell --nv "${ENV_ARGS[@]}" "${BIND_ARGS[@]}" "$ISAAC_SIM_SIF"
fi

[[ $# -gt 0 ]] || {
  exec "$ROOT_DIR/docker/problem2/render_twin.sh"
}
exec "$RUNTIME" exec --nv "${ENV_ARGS[@]}" "${BIND_ARGS[@]}" "$ISAAC_SIM_SIF" \
  "$ISAAC_SIM_SHELL" "$ISAAC_SIM_PYTHON" "$@"

