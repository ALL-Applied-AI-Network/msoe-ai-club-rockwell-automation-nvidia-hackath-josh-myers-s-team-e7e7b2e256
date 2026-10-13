#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWIN_CAMERA=CapSealCam \
TWIN_STAGE=/workspace/data/csi_digital_twin_v1/stations/tray_inspection/tray_inspection.usdc \
TWIN_FRAME="${TWIN_FRAME:-1}" \
TWIN_OUTPUT="${TWIN_OUTPUT:-/workspace/outputs/problem1_cap_seal.png}" \
  exec "$SCRIPT_DIR/../run_csi_twin.sh" "$@"

