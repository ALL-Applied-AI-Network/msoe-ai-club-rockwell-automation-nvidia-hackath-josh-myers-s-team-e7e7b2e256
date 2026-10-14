#!/usr/bin/env bash
set -euo pipefail

# Run the shared standalone Isaac Sim installation inside an existing ROSIE
# allocation. Submit docker/isaac_sim/submit.sh from a management node.
# This launcher intentionally does not use a Singularity/Apptainer Isaac image.

SCRIPT_PATH="$(realpath "$0")"
ROOT_DIR="${ISAAC_SIM_ROOT_DIR:-$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
ISAAC_SIM_PYTHON="${ISAAC_SIM_PYTHON:-$ROOT_DIR/models/isaacsim_6.1/python.sh}"

if [[ ! -x "$ISAAC_SIM_PYTHON" ]]; then
  echo "Shared Isaac Sim launcher not found or not executable: $ISAAC_SIM_PYTHON" >&2
  echo "Install it with: $ROOT_DIR/scripts/install_isaac_sim_standalone.sh" >&2
  exit 1
fi

cd "$ROOT_DIR"
export ACCEPT_EULA="${ACCEPT_EULA:-Y}"
export PRIVACY_CONSENT="${PRIVACY_CONSENT:-Y}"
export ISAAC_SIM_PYTHON

if [[ "${1:-}" == shell ]]; then
  exec bash -l
fi

[[ $# -gt 0 ]] || {
  echo "Usage: $0 [shell] <python-script> [args...]" >&2
  echo "Example: $0 data/csi_digital_twin_v1/tools/render_example.py --camera FillCam" >&2
  exit 2
}

exec "$ISAAC_SIM_PYTHON" "$@"
