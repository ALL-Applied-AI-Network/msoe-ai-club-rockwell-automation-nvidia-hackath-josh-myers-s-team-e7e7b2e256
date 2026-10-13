#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper. run.sh now handles the Rosie srun allocation itself.
SCRIPT_PATH="$(realpath "$0")"
ROOT_DIR="$(cd "$(dirname "$SCRIPT_PATH")/../.." && pwd)"
exec "$ROOT_DIR/docker/isaac_sim/run.sh" "$@"
