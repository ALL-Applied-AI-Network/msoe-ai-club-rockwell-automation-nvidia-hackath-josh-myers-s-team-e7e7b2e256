#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${SIF_PATH:-$ROOT_DIR/innovation-lab.sif}"

echo "== Singularity digital-twin setup =="

if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
  fi
  if command -v module >/dev/null 2>&1; then
    module load singularity 2>/dev/null || module load apptainer 2>/dev/null || true
  fi
fi

if command -v apptainer >/dev/null 2>&1; then
  RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then
  RUNTIME=singularity
else
  echo "Singularity/Apptainer is not available." >&2
  echo "Ask the ROSIE administrator which module to load, then rerun this script." >&2
  exit 1
fi

echo "Runtime: $RUNTIME ($($RUNTIME --version 2>&1 | head -n 1))"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA driver detected; runtime jobs can use --nv."
else
  echo "NVIDIA driver is not visible in this node; build will continue, but GPU jobs need a GPU allocation."
fi

if [[ -e "$SIF_PATH" && "${FORCE:-0}" != 1 ]]; then
  echo "SIF already exists: $SIF_PATH"
  echo "Use FORCE=1 to rebuild it."
else
  echo "Converting the Docker image locally through $RUNTIME..."
  FORCE="${FORCE:-0}" SIF_PATH="$SIF_PATH" "$ROOT_DIR/docker/build_singularity.sh"
fi

echo "Validating the image..."
"$RUNTIME" inspect "$SIF_PATH" >/dev/null
echo "Digital twin image is ready: $SIF_PATH"