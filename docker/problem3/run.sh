#!/usr/bin/env bash
set -euo pipefail

# Problem 3 is the FANUC ER-4iA simulation, not a project-SIF command.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/sim.sh" "$@"
