#!/usr/bin/env bash
set -euo pipefail

# Install NVIDIA's standalone Isaac Sim alongside this repository.
# Usage: ./scripts/install_isaac_sim_standalone.sh [target-directory]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET_DIR="${1:-$REPO_ROOT/models/isaacsim_6.1}"
DOWNLOAD_DIR="${ISAAC_SIM_DOWNLOAD_DIR:-${HOME}/Downloads}"
ZIP_URL="${ISAAC_SIM_ZIP_URL:-https://downloads.isaacsim.nvidia.com/isaac-sim-standalone-6.1.0-linux-x86_64.zip}"
ZIP_FILE="$DOWNLOAD_DIR/$(basename "$ZIP_URL")"

mkdir -p "$DOWNLOAD_DIR" "$TARGET_DIR"

if [[ ! -f "$ZIP_FILE" ]] || ! unzip -tq "$ZIP_FILE" >/dev/null 2>&1; then
  echo "Downloading Isaac Sim standalone 6.1.0 (~12.4 GB) to $ZIP_FILE"
  wget -c --progress=dot:giga "$ZIP_URL" -O "$ZIP_FILE"
else
  echo "Using existing download: $ZIP_FILE"
fi

echo "Extracting to $TARGET_DIR"
unzip -q -o "$ZIP_FILE" -d "$TARGET_DIR"

cd "$TARGET_DIR"
if [[ -x ./post_install.sh ]]; then
  ./post_install.sh
else
  echo "post_install.sh was not found in $TARGET_DIR" >&2
  exit 1
fi

if [[ ! -x ./kit/python/bin/python3 ]]; then
  echo "Isaac Sim extraction is incomplete: $TARGET_DIR/kit/python/bin/python3 is missing" >&2
  exit 1
fi

echo "Installed Isaac Sim at $TARGET_DIR"
