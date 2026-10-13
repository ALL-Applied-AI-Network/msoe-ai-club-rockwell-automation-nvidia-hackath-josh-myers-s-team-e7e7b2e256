#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWIN_CAMERA="${TWIN_CAMERA:-FillCam}"
TWIN_STAGE=/workspace/data/csi_digital_twin_v1/stations/fill_station/fill_station.usda
TWIN_FRAME="${TWIN_FRAME:-686}"
TWIN_OUTPUT="${TWIN_OUTPUT:-/workspace/outputs/problem2_${TWIN_CAMERA,,}.png}"
export TWIN_CAMERA TWIN_STAGE TWIN_FRAME TWIN_OUTPUT
exec "$SCRIPT_DIR/../run_csi_twin.sh" "$@"

