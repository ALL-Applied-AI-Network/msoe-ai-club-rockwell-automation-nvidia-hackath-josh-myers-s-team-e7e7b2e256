#!/usr/bin/env bash
set -euo pipefail

# Convert a Docker Hub image to a SIF directly on HPC. Singularity's Docker
# transport does not require a Docker daemon, fakeroot, or image transfer.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${SIF_PATH:-$ROOT_DIR/innovation-lab.sif}"
IMAGE_URI="${IMAGE_URI:-docker://nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04}"
BUILD_ROOT="${SINGULARITY_BUILD_DIR:-$ROOT_DIR/docker/.cache/project-build}"
CACHE_ROOT="${SINGULARITY_CACHE_DIR:-$ROOT_DIR/docker/.cache/project-cache}"

if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else echo "Singularity or Apptainer is required." >&2; exit 1; fi
if [[ -e "$SIF_PATH" && "${FORCE:-0}" != 1 ]]; then
  echo "Refusing to overwrite existing image: $SIF_PATH" >&2
  echo "Re-run with FORCE=1 to replace it." >&2
  exit 1
fi

mkdir -p "$(dirname "$SIF_PATH")" "$BUILD_ROOT/tmp" "$CACHE_ROOT"
export TMPDIR="$BUILD_ROOT/tmp"
export SINGULARITY_TMPDIR="$BUILD_ROOT/tmp"
export APPTAINER_TMPDIR="$BUILD_ROOT/tmp"
export SINGULARITY_CACHEDIR="$CACHE_ROOT"
export APPTAINER_CACHEDIR="$CACHE_ROOT"
if [[ "${FORCE:-0}" == 1 && -e "$SIF_PATH" ]]; then rm -f -- "$SIF_PATH"; fi
echo "Using temporary space: $TMPDIR"
echo "Using Singularity cache: $CACHE_ROOT"
"$RUNTIME" pull --name "$SIF_PATH" "$IMAGE_URI"
echo "Created $SIF_PATH"
