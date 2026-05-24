#!/usr/bin/env bash
# Single-command "where are we" snapshot for the RL pipeline.
# Output is meant to be pasted back to whoever is debugging.
#
# Captures:
#   - active processes, GPU utilization, disk
#   - venv sanity (via verify_rl_setup.py)
#   - latest dry-run log tail + last errors
#   - the unresolved-debug section (currently: WorkerDict dispatch)
#
# The full snapshot goes to a fresh log file under rl_smoke/logs/.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"

LOG_DIR=$REPO/rl_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/rl_status_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/rl_status_latest.log"
exec > >(tee -a "$LOG") 2>&1

banner() {
    echo
    echo "════════════════════════════════════════════════════════════════════"
    echo "  $*"
    echo "════════════════════════════════════════════════════════════════════"
}

banner "snapshot at $(date -Iseconds) — host $(hostname) — repo $REPO"

banner "1. processes"
pgrep -af "train_agent_ppo|ray::|vllm serve" 2>/dev/null | grep -v "rl_status\|pgrep" | head -20 || echo "(none)"

banner "2. GPU"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader || echo "(nvidia-smi failed)"

banner "3. disk"
df -h / /data 2>/dev/null | grep -v tmpfs | head

banner "4. env / venv state"
echo "PATH includes /usr/local/cuda/bin? $(echo \"$PATH\" | tr : '\n' | grep -c cuda)"
echo "SWE_MASTER_CACHE_ROOT=${SWE_MASTER_CACHE_ROOT:-<unset>}"
echo "HF_HOME=${HF_HOME:-<unset>}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-<unset>}"
echo
if [ -x "$REPO/DeepSWE_RL/.venv/bin/python" ]; then
    echo "rllm venv python: $REPO/DeepSWE_RL/.venv/bin/python"
    "$REPO/DeepSWE_RL/.venv/bin/python" -c "import sys; print('  ', sys.version.split()[0])"
else
    echo "rllm venv: MISSING (run: bash install_rl_env.sh)"
fi

banner "5. verify_rl_setup.py"
if [ -x "$REPO/DeepSWE_RL/.venv/bin/python" ]; then
    "$REPO/DeepSWE_RL/.venv/bin/python" "$REPO/verify_rl_setup.py" || true
else
    echo "(skipped — no venv)"
fi

banner "6. patches applied"
bash "$REPO/apply_patches_rl.sh" --dry-run 2>&1 | tail -30 || true

banner "7. latest log files (last 5)"
ls -lt "$LOG_DIR" 2>/dev/null | head -7

banner "8. latest dry-run log: tail + tracebacks"
DRY=$(ls -t "$LOG_DIR"/rl_dryrun_*.log 2>/dev/null | head -1)
if [ -n "$DRY" ]; then
    echo "file: $DRY ($(stat -c %s "$DRY") bytes, last write $(stat -c %y "$DRY"))"
    echo
    echo "--- last 40 lines ---"
    tail -40 "$DRY"
    echo
    echo "--- tracebacks/errors ---"
    grep -nE "Traceback|^Error |ImportError|ModuleNotFoundError|AttributeError|OutOfMemoryError|ValueError|RuntimeError|raise [A-Z]" "$DRY" 2>/dev/null | tail -10
else
    echo "(no rl_dryrun_*.log yet — run: bash run_rl_dryrun.sh)"
fi

banner "9. things to keep an eye on"
cat <<'EOF'
- rllm uses actor_rollout_ref.rollout.mode=async to pick async_rollout_manager;
  run_rl_dryrun.sh now sets it. If you build a custom run script, include both
  rollout.mode=async AND
  rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler
- patch_rllm_wake_up_dispatch.py is NO LONGER part of apply_patches_rl.py.
  If you find it still applied to agent_execution_engine.py, revert it:
    cd DeepSWE_RL/rllm && git checkout rllm/engine/agent_execution_engine.py
- 2-GPU PPO at any non-trivial scale will OOM. The dry-run is intentionally
  tiny (1.5B base, prompt_len 2048, batch 2, 1 step). Real training needs
  8-64 GPUs per the official deepswe_32b.sh.
EOF

banner "snapshot written to: $LOG"
echo "(paste this whole log when you want help)"
