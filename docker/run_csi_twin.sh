#!/usr/bin/env bash
set -euo pipefail

# Render the CSI USD twin with Isaac Sim. The project SIF is intentionally a
# development image; Isaac Sim must be supplied separately through
# ISAAC_SIM_SIF or an installed ISAAC_SIM_PYTHON executable.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TWIN_ROOT="/workspace/data/csi_digital_twin_v1"
CAMERA="${TWIN_CAMERA:-FillCam}"
STAGE="${TWIN_STAGE:-$TWIN_ROOT/csi_line.usda}"
FRAME="${TWIN_FRAME:-1}"
OUTPUT="${TWIN_OUTPUT:-/workspace/outputs/csi_${CAMERA,,}_${FRAME}.png}"
SUBFRAMES="${TWIN_SUBFRAMES:-8}"

TOOL="$TWIN_ROOT/tools/render_example.py"
ARGS=("$TOOL" --camera "$CAMERA" --stage "$STAGE" --frame "$FRAME" --out "$OUTPUT" --subframes "$SUBFRAMES")

if [[ -n "${ISAAC_SIM_PYTHON:-}" ]]; then
  HOST_TOOL="$ROOT_DIR/data/csi_digital_twin_v1/tools/render_example.py"
  HOST_STAGE="${STAGE#/workspace/}"
  HOST_OUTPUT="${OUTPUT#/workspace/}"
  ARGS=("$HOST_TOOL" --camera "$CAMERA" --stage "$ROOT_DIR/$HOST_STAGE" --frame "$FRAME" --out "$ROOT_DIR/$HOST_OUTPUT" --subframes "$SUBFRAMES")
  exec "$ISAAC_SIM_PYTHON" "${ARGS[@]}"
fi

ISAAC_SIM_SIF="${ISAAC_SIM_SIF:-$ROOT_DIR/isaac-sim-6.0.0.sif}"
[[ -f "$ISAAC_SIM_SIF" ]] || { echo "Isaac Sim SIF not found: $ISAAC_SIM_SIF" >&2; exit 1; }

export ISAAC_SIM_SIF
exec "$ROOT_DIR/docker/isaac_sim/run.sh" "${ARGS[@]}"

