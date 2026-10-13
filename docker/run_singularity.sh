#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIF_PATH="${SIF_PATH:-$ROOT_DIR/innovation-lab.sif}"
USE_GPU="${USE_GPU:-1}"
CONFIG_FILE="${CONFIG_FILE:-}"
if command -v apptainer >/dev/null 2>&1; then RUNTIME=apptainer
elif command -v singularity >/dev/null 2>&1; then RUNTIME=singularity
else echo "Singularity or Apptainer is required." >&2; exit 1; fi
[[ -f "$SIF_PATH" ]] || { echo "SIF image not found: $SIF_PATH" >&2; exit 1; }
BIND_ARGS=(-B "$ROOT_DIR:/workspace")
if [[ -d /data ]]; then
  BIND_ARGS+=(-B /data:/data)
fi

ENV_ARGS=()
if [[ -n "$CONFIG_FILE" ]]; then
  [[ -f "$ROOT_DIR/$CONFIG_FILE" ]] || {
    echo "Config file not found: $ROOT_DIR/$CONFIG_FILE" >&2
    exit 1
  }
  ENV_ARGS+=(--env "INNOVATION_LAB_CONFIG=/workspace/$CONFIG_FILE")
fi

if [[ "${1:-}" == shell ]]; then
  SHELL_ARGS=()
  [[ "$USE_GPU" == 1 ]] && SHELL_ARGS+=(--nv)
  exec "$RUNTIME" shell "${SHELL_ARGS[@]}" "${ENV_ARGS[@]}" "${BIND_ARGS[@]}" "$SIF_PATH"
fi

ARGS=(exec)
[[ "$USE_GPU" == 1 ]] && ARGS+=(--nv)
ARGS+=("${ENV_ARGS[@]}" "${BIND_ARGS[@]}" "$SIF_PATH")
exec "$RUNTIME" "${ARGS[@]}" "${@:-uv run python main.py}"
