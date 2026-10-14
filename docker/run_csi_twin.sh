#!/usr/bin/env bash

#SBATCH --job-name=csi-twin
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-gpu=8
#SBATCH --ntasks=1

set -euo pipefail

# Render a CSI USD twin with the shared standalone Isaac Sim installation.
# Paths are kept on the host; no Isaac Sim container or /workspace bind is used.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  mkdir -p "$ROOT_DIR/logs"
  exec sbatch --wait --chdir="$ROOT_DIR" \
    --output="$ROOT_DIR/logs/csi-twin-%j.out" \
    --error="$ROOT_DIR/logs/csi-twin-%j.err" \
    "$0" "$@"
fi

TOOL="$ROOT_DIR/data/csi_digital_twin_v1/tools/render_example.py"
CAMERA="${TWIN_CAMERA:-FillCam}"
STAGE="${TWIN_STAGE:-$ROOT_DIR/data/csi_digital_twin_v1/csi_line.usda}"
FRAME="${TWIN_FRAME:-1}"
OUTPUT="${TWIN_OUTPUT:-$ROOT_DIR/outputs/csi_${CAMERA,,}_${FRAME}.png}"
SUBFRAMES="${TWIN_SUBFRAMES:-8}"

[[ -f "$TOOL" ]] || { echo "Twin renderer not found: $TOOL" >&2; exit 1; }
[[ -f "$STAGE" ]] || { echo "Twin stage not found: $STAGE" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT")"

exec "$ROOT_DIR/docker/isaac_sim/run.sh" \
  "$TOOL" \
  --camera "$CAMERA" \
  --stage "$STAGE" \
  --frame "$FRAME" \
  --out "$OUTPUT" \
  --subframes "$SUBFRAMES" \
  "$@"
