#!/usr/bin/env bash
# Apply the RL-side patches to rllm/verl.
# Idempotent — safe to re-run.
#
# Usage:
#   bash apply_patches_rl.sh           # apply
#   bash apply_patches_rl.sh --dry-run # report status only
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"

LOG_DIR=$REPO/rl_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/apply_patches_rl_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/apply_patches_rl_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

# Prefer the RL venv's Python if it exists; otherwise fall back to system python3.
PYTHON=python3
if [ -x "$REPO/DeepSWE_RL/.venv/bin/python" ]; then
    PYTHON="$REPO/DeepSWE_RL/.venv/bin/python"
fi
echo "==> Using $PYTHON"

exec "$PYTHON" "$REPO/apply_patches_rl.py" "$@"
