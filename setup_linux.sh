#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}"

echo "== Physical AI Innovation Lab setup (Linux) =="

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to install uv. Install curl with your system package manager and rerun this script."
    exit 1
  fi
  echo "uv not found; installing it for the current user..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv installation completed but uv is not on PATH. Add ~/.local/bin to PATH and rerun this script."
  exit 1
fi

mkdir -p data/problem1 data/problem2 data/problem3 configs artifacts models outputs logs
touch data/problem1/.gitkeep data/problem2/.gitkeep data/problem3/.gitkeep

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example; add credentials only if you need NVIDIA services."
fi

uv sync --dev
uv run python main.py

echo "Setup complete. Activate with: source .venv/bin/activate"
