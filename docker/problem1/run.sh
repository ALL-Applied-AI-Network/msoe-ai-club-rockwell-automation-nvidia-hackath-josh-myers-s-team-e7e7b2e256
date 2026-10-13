#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="configs/problem1_vial_inspection.yaml" \
  exec "$SCRIPT_DIR/../run_singularity.sh" "${@:-shell}"

