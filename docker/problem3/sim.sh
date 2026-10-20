#!/usr/bin/env bash

#SBATCH --job-name=problem3-fanuc
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

set -euo pipefail

# Run the in-repository FANUC ER-4iA starter pipeline with shared Isaac Sim.
# Override FANUC_SIM_ROOT only when using a separate starter checkout.

SCRIPT_PATH="$(realpath "$0")"
PROJECT_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
FANUC_SIM_ROOT="${FANUC_SIM_ROOT:-$PROJECT_ROOT/fanuc_er4ia_sim}"
ISAAC="${ISAAC:-$PROJECT_ROOT/models/isaacsim_6.1}"
COSMOS="${COSMOS:-${COSMOS3_PYTHON:-$HOME/.venvs/cosmos3/bin/python}}"
COSMOS3_NANO_PATH="${COSMOS3_NANO_PATH:-$PROJECT_ROOT/models/Cosmos3-Nano}"
HOST_PYTHON="${HOST_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"
ISAAC_SIM_ALLOCATE="${ISAAC_SIM_ALLOCATE:-1}"

# Use an explicit count when supplied. Otherwise determine the count from the
# current allocation or, on a management node, from the teaching partition.
if [[ -z "${PROBLEM3_GPUS:-}" ]]; then
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      PROBLEM3_GPUS="$(grep -oE '[0-9]+' <<< "$SLURM_GPUS_ON_NODE" | tail -n 1 || true)"
    elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      PROBLEM3_GPUS="$(awk -F, '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
    fi
  elif command -v sinfo >/dev/null 2>&1; then
    PROBLEM3_GPUS="$(sinfo -h -p teaching -o '%G' 2>/dev/null \
      | grep -oE 'gpu:[^:]+:[0-9]+' \
      | grep -oE '[0-9]+$' \
      | sort -nr \
      | head -n 1 || true)"
  fi
fi
PROBLEM3_GPUS="${PROBLEM3_GPUS:-2}"

if [[ "${1:-}" == --in-allocation ]]; then
  shift
fi

[[ -d "$FANUC_SIM_ROOT" ]] || {
  echo "FANUC starter not found: $FANUC_SIM_ROOT" >&2
  echo "Expected in-repository path: $PROJECT_ROOT/fanuc_er4ia_sim" >&2
  exit 1
}
[[ -x "$ISAAC/python.sh" ]] || {
  echo "Shared Isaac Sim not found: $ISAAC/python.sh" >&2
  echo "Run: $PROJECT_ROOT/scripts/install_isaac_sim_standalone.sh" >&2
  exit 1
}
[[ -x "$HOST_PYTHON" ]] || {
  echo "Host Python not found: $HOST_PYTHON" >&2
  echo "Set HOST_PYTHON or create the project environment with uv sync." >&2
  exit 1
}
[[ "$PROBLEM3_GPUS" =~ ^[1-9][0-9]*$ ]] || {
  echo "PROBLEM3_GPUS must be a positive integer: $PROBLEM3_GPUS" >&2
  exit 1
}

if [[ "${1:-}" == --host-tests ]]; then
  cd "$FANUC_SIM_ROOT"
  exec "$HOST_PYTHON" -m unittest \
    test_sort_planning test_cube_perception test_cube_pose test_runtime_config
fi

[[ -x "$COSMOS" ]] || {
  echo "COSMOS is not executable: $COSMOS" >&2
  echo "Run scripts/setup_linux.sh or set COSMOS=/path/to/cosmos-environment/bin/python." >&2
  exit 1
}
[[ -f "$COSMOS3_NANO_PATH/config.json" ]] || {
  echo "Cosmos3-Nano model is missing: $COSMOS3_NANO_PATH/config.json" >&2
  echo "Run scripts/setup_linux.sh after accepting the Hugging Face model license." >&2
  exit 1
}

if [[ -z "${SLURM_JOB_ID:-}" && "$ISAAC_SIM_ALLOCATE" == 1 ]]; then
  mkdir -p "$PROJECT_ROOT/logs"
  export FANUC_SIM_ROOT ISAAC HOST_PYTHON COSMOS COSMOS3_NANO_PATH PROBLEM3_GPUS
  exec sbatch --wait \
    --partition=teaching \
    --gres="gpu:t4:$PROBLEM3_GPUS" \
    --cpus-per-gpu=8 \
    --ntasks=1 \
    --chdir="$PROJECT_ROOT" \
    --output="$PROJECT_ROOT/logs/problem3-%j.out" \
    --error="$PROJECT_ROOT/logs/problem3-%j.err" \
    "$SCRIPT_PATH" --in-allocation "$@"
fi

if [[ "$#" -eq 0 ]]; then
  set -- "$PROJECT_ROOT/output/runs/cube_stack_$(date +%Y%m%d_%H%M%S)"
fi

export ISAAC FANUC_SIM_ROOT HOST_PYTHON COSMOS COSMOS3_NANO_PATH PROBLEM3_GPUS
exec "$FANUC_SIM_ROOT/run_sort_pipeline.sh" "$@"
