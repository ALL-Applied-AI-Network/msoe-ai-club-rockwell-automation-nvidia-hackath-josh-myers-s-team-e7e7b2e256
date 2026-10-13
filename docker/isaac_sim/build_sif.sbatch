#!/usr/bin/env bash

#SBATCH --job-name=isaac-sim-build
#SBATCH --partition=teaching
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/isaac-sim-build-%j.out
#SBATCH --error=logs/isaac-sim-build-%j.err


set -euo pipefail

ROOT_DIR="${ISAAC_SIM_ROOT_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"
cd "$ROOT_DIR"

ISAAC_SIM_VERSION="${ISAAC_SIM_VERSION:-6.0.0}"
IMAGE_URI="${IMAGE_URI:-docker://nvcr.io/nvidia/isaac-sim:${ISAAC_SIM_VERSION}}"
SIF_PATH="${ISAAC_SIM_SIF:-$ROOT_DIR/isaac-sim-${ISAAC_SIM_VERSION}.sif}"
DEF_FILE="${ISAAC_SIM_DEF:-$ROOT_DIR/docker/isaac_sim/Singularity.def}"
BUILD_ROOT="${ISAAC_SIM_BUILD_DIR:-$ROOT_DIR/docker/.cache/isaac-sim-build}"
CACHE_ROOT="${ISAAC_SIM_CACHE_DIR:-$ROOT_DIR/docker/.cache/isaac-sim-cache}"
# Rosie accounts without subuid/subgid mappings cannot use local fakeroot.
# The remote builder preserves the permission-normalization step needed by
# the runtime image, so it is the safe default for those accounts. Set this
# explicitly to fakeroot on hosts that provide the mappings.
BUILD_MODE="${ISAAC_SIM_BUILD_MODE:-remote}"

if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else echo "Singularity or Apptainer is required." >&2; exit 1; fi

if [[ -e "$SIF_PATH" && "${FORCE:-0}" != 1 ]]; then
  echo "Refusing to overwrite existing image: $SIF_PATH" >&2
  echo "Submit again with FORCE=1 to replace it." >&2
  exit 1
fi

if [[ -n "${NGC_API_KEY:-}" ]]; then
  export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
  export SINGULARITY_DOCKER_PASSWORD="$NGC_API_KEY"
fi

[[ -f "$DEF_FILE" ]] || { echo "Singularity definition not found: $DEF_FILE" >&2; exit 1; }

mkdir -p "$(dirname "$SIF_PATH")" "$BUILD_ROOT" "$CACHE_ROOT"
export TMPDIR="$BUILD_ROOT/tmp"
export SINGULARITY_TMPDIR="$BUILD_ROOT/tmp"
export APPTAINER_TMPDIR="$BUILD_ROOT/tmp"
export SINGULARITY_CACHEDIR="$CACHE_ROOT"
export APPTAINER_CACHEDIR="$CACHE_ROOT"
mkdir -p "$TMPDIR"
[[ "${FORCE:-0}" == 1 ]] && rm -f -- "$SIF_PATH"

if [[ "$BUILD_MODE" == auto ]]; then
  if [[ -r /etc/subuid ]] && grep -qE "(^|:)${USER:-$(id -un)}:" /etc/subuid; then
    BUILD_MODE=fakeroot
  else
    cat >&2 <<'EOF'
Rosie has no /etc/subuid mapping for this account, so local --fakeroot builds
are unavailable. Choose one of:
  1. Ask the Rosie administrator to add subuid/subgid mappings.
  2. Set ISAAC_SIM_BUILD_MODE=remote and authenticate Singularity's remote builder.
  3. Set ISAAC_SIM_BUILD_MODE=direct only for a non-runtime diagnostic build.
EOF
    exit 4
  fi
fi

echo "Using temporary space: $TMPDIR"
echo "Using OCI cache: $CACHE_ROOT"
case "$BUILD_MODE" in
  fakeroot)
    echo "Building with local fakeroot permission normalization"
    "$RUNTIME" build --fakeroot --fix-perms "$SIF_PATH" "$DEF_FILE"
    ;;
  remote)
    echo "Building with Singularity remote builder"
    echo "Authenticate first with: singularity remote login"
    "$RUNTIME" build --remote --fix-perms "$SIF_PATH" "$DEF_FILE"
    ;;
  direct)
    echo "Building directly from OCI image (permission fix unavailable)"
    "$RUNTIME" build --fix-perms "$SIF_PATH" "$IMAGE_URI"
    ;;
  *)
    echo "Invalid ISAAC_SIM_BUILD_MODE: $BUILD_MODE (use fakeroot, remote, or direct)" >&2
    exit 2
    ;;
esac
chmod 0770 -- "$SIF_PATH"
echo "Set permissions 770 on $SIF_PATH"
echo "Created $SIF_PATH"
