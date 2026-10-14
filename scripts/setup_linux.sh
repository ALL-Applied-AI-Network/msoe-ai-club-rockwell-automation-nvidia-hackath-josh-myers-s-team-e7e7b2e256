#!/usr/bin/env bash
set -euo pipefail

# Complete Linux/ROSIE setup for Problems 1, 2 and 3.
# Large runtimes and model weights live under the ignored models/ directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$ROOT_DIR/logs"
LOG_FILE="$LOG_DIR/setup-$(date +%Y%m%d-%H%M%S).log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

PASS=0
WARN=0
step() { printf '\n[%s] %s\n' "$1" "$2"; }
ok() { PASS=$((PASS + 1)); printf '[OK] %s\n' "$1"; }
warn() { WARN=$((WARN + 1)); printf '[WARN] %s\n' "$1" >&2; }
die() {
  printf '[ERROR] %s\n' "$1" >&2
  printf 'Setup failed. Full log: %s\n' "$LOG_FILE" >&2
  exit 1
}

cd "$ROOT_DIR"
printf 'Physical AI Innovation Lab setup\nRepository: %s\nLog: %s\n' "$ROOT_DIR" "$LOG_FILE"

step 1 'Checking Linux prerequisites'
[[ "$(uname -s)" == Linux ]] || die 'This script is for Linux/ROSIE. Use scripts/setup_windows.ps1 on Windows.'
command -v python3 >/dev/null || die 'python3 is required.'
command -v curl >/dev/null || die 'curl is required.'
command -v wget >/dev/null || die 'wget is required for Isaac Sim installation.'
command -v unzip >/dev/null || die 'unzip is required for Isaac Sim installation.'
ok 'Linux prerequisites found.'

step 2 'Installing or locating uv'
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die 'uv is not on PATH after installation.'
ok "uv: $(uv --version)"

step 3 'Creating repository directories and local environment file'
mkdir -p data/problem1 data/problem2 data/problem3 configs artifacts models outputs logs
if [[ ! -f .env ]]; then
  cp .env.example .env
  ok 'Created .env from .env.example.'
else
  ok 'Kept existing .env.'
fi

step 4 'Installing the project Python environment'
UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.uv-cache}" uv sync --dev
uv run python main.py
ok 'Project environment and smoke check are ready.'

step 5 'Running FANUC host regression tests'
uv run python -m unittest discover -s fanuc_er4ia_sim -p 'test_*.py'
ok 'FANUC host tests passed.'

COSMOS_ENV="${COSMOS_ENV:-$HOME/.venvs/cosmos3}"
COSMOS_PYTHON="$COSMOS_ENV/bin/python"
step 6 "Preparing Cosmos3 environment at $COSMOS_ENV"
mkdir -p "$(dirname "$COSMOS_ENV")"
if [[ ! -x "$COSMOS_PYTHON" ]]; then
  uv venv "$COSMOS_ENV" --python python3
fi
uv pip install --python "$COSMOS_PYTHON" \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu129
uv pip install --python "$COSMOS_PYTHON" \
  -r "$ROOT_DIR/fanuc_er4ia_sim/requirements-cosmos.txt" \
  'transformers>=4.57' accelerate safetensors huggingface_hub
ok 'Cosmos Python dependencies are installed.'

COSMOS3_ROOT="${COSMOS3_ROOT:-$HOME/cosmos-framework}"
step 7 "Installing NVIDIA Cosmos framework at $COSMOS3_ROOT"
if [[ ! -d "$COSMOS3_ROOT/.git" ]]; then
  git clone https://github.com/NVIDIA/cosmos-framework.git "$COSMOS3_ROOT"
fi
[[ -d "$COSMOS3_ROOT/packages/transformers-cosmos3" ]] || \
  die "Cosmos framework checkout is missing packages/transformers-cosmos3: $COSMOS3_ROOT"
uv pip install --python "$COSMOS_PYTHON" -e "$COSMOS3_ROOT/packages/transformers-cosmos3"
ok 'NVIDIA Cosmos framework is installed.'

MODEL_DIR="$ROOT_DIR/models/Cosmos3-Nano"
step 8 "Checking Cosmos3-Nano model at $MODEL_DIR"
mkdir -p "$ROOT_DIR/models"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  HF_TOKEN_VALUE="${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGING_FACE_HUB_TOKEN:-}}}"
  if [[ -n "$HF_TOKEN_VALUE" ]]; then
    export HF_TOKEN="$HF_TOKEN_VALUE"
  elif "$COSMOS_ENV/bin/hf" auth whoami >/dev/null 2>&1; then
    ok 'Using the existing Hugging Face CLI login.'
  elif [[ -t 0 ]]; then
    echo "Cosmos3-Nano is gated. Accept NVIDIA's model license on Hugging Face first."
    "$COSMOS_ENV/bin/hf" auth login
  else
    warn 'Cosmos3-Nano weights are not downloaded because setup is noninteractive and Hugging Face is not authenticated.'
    warn "Accept the NVIDIA model license, run '$COSMOS_ENV/bin/hf auth login', then rerun setup_linux.sh."
  fi
  if [[ -n "$HF_TOKEN_VALUE" || "$($COSMOS_ENV/bin/hf auth whoami >/dev/null 2>&1; echo $?)" == 0 ]]; then
    "$COSMOS_ENV/bin/hf" download nvidia/Cosmos3-Nano --local-dir "$MODEL_DIR"
    ok 'Downloaded Cosmos3-Nano weights.'
  fi
else
  ok 'Cosmos3-Nano weights already exist.'
fi

step 9 'Installing shared Isaac Sim under models/isaacsim_6.1'
ISAAC_DIR="$ROOT_DIR/models/isaacsim_6.1"
if [[ -x "$ISAAC_DIR/python.sh" && -x "$ISAAC_DIR/kit/python/bin/python3" ]]; then
  ok 'Shared Isaac Sim is already installed.'
else
  "$ROOT_DIR/scripts/install_isaac_sim_standalone.sh" "$ISAAC_DIR"
  ok 'Shared Isaac Sim installation completed.'
fi

step 10 'Validating all three problem launch prerequisites'
[[ -x "$ISAAC_DIR/python.sh" ]] || die "Missing Isaac launcher: $ISAAC_DIR/python.sh"
[[ -x "$COSMOS_PYTHON" ]] || die "Missing Cosmos Python: $COSMOS_PYTHON"
[[ -f "$MODEL_DIR/config.json" ]] || die "Cosmos3-Nano is incomplete. Run '$COSMOS_ENV/bin/hf auth login', accept NVIDIA's license, and rerun setup_linux.sh."
[[ -x "$ROOT_DIR/docker/problem1/render_twin.sh" ]] || die 'Problem 1 launcher is not executable.'
[[ -x "$ROOT_DIR/docker/problem2/render_twin.sh" ]] || die 'Problem 2 launcher is not executable.'
[[ -x "$ROOT_DIR/docker/problem3/run.sh" ]] || die 'Problem 3 launcher is not executable.'
ok 'Problem launchers and runtime paths validated.'

printf '\nSetup finished: %d checks passed, %d warnings.\n' "$PASS" "$WARN"
printf 'Log: %s\n' "$LOG_FILE"
printf '\nRun problems from the repository root:\n'
printf '  ./docker/problem1/render_twin.sh\n'
printf '  ./docker/problem2/render_twin.sh\n'
printf '  PROBLEM3_GPUS=2 ./docker/problem3/run.sh output/runs/stack_01\n'
