#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="configs/problem3_grasping_sorting.yaml" \
  exec "$SCRIPT_DIR/../run_singularity.sh" "${@:-shell}"

