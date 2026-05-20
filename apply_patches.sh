#!/usr/bin/env bash
# Apply the 10 patches our fork makes to upstream code.
# Idempotent: safe to re-run after fresh installs.
#
# Usage:
#   bash apply_patches.sh          # full mode (Ada/48GB)
#   bash apply_patches.sh h200     # lean mode (skips 4 memory band-aids)
#   bash apply_patches.sh --dry-run     # show what would change
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)
MODE="${1:-full}"

ARGS=()
case "$MODE" in
    full|h200) ARGS=(--mode "$MODE") ;;
    --dry-run) ARGS=(--dry-run) ;;
    *) echo "usage: $0 [full|h200|--dry-run]"; exit 1 ;;
esac
# Allow second arg as --dry-run
if [ "${2:-}" = "--dry-run" ]; then ARGS+=(--dry-run); fi

# We don't need any heavy Python deps; use the system interpreter
# (or the OpenRLHF venv if it's present and we want stable Python).
PYTHON=python3
if [ -x "$REPO/OpenRLHF_SFT/.venv/bin/python" ]; then
    PYTHON="$REPO/OpenRLHF_SFT/.venv/bin/python"
fi

exec "$PYTHON" "$REPO/apply_patches.py" "${ARGS[@]}"
