#!/usr/bin/env bash
# Smoke-test the R2E-Gym teacher-rollout pipeline.
#
# What this runs end-to-end:
#   1. Loads instance 0 (astropy__astropy-12907) from SWE-Bench-Verified.
#   2. Pulls the slimshetty Docker image lazily on first use.
#   3. Spins up the agent (openhands scaffold + fn_calling) talking to our
#      local vLLM serving Qwen2.5-Coder-7B-Instruct on http://localhost:8000.
#   4. Writes a trajectory JSONL to $TRAJ_DIR.
#
# Goal: validate the pipeline mechanics (dataset → docker → agent → API → traj).
# We do NOT expect the 7B teacher to actually solve the bug; the trajectory
# shape is what we're after.
set -euo pipefail

# Override via env vars:
#   TEACHER_NAME=qwen25-coder-14b-instruct bash run_rollout_smoke.sh
#   USE_FN_CALLING=False USED_YAML=./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml \
#     bash run_rollout_smoke.sh
#   START_IDX=0 K=5 MAX_STEPS=15 MAX_WORKERS=2 bash run_rollout_smoke.sh
TEACHER_NAME=${TEACHER_NAME:-qwen25-coder-7b-instruct}
START_IDX=${START_IDX:-0}
K=${K:-1}
MAX_STEPS=${MAX_STEPS:-10}
MAX_WORKERS=${MAX_WORKERS:-1}
USE_FN_CALLING=${USE_FN_CALLING:-True}
USED_YAML=${USED_YAML:-./src/r2egym/agenthub/config/openhands/openhands_sp_fn_calling.yaml}

REPO=/data/nikunj/SWE-Master
LOG_DIR=$REPO/sft_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/rollout_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/rollout_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

# Confirm vLLM is up before we start an agent that'll fail without it.
if ! curl -sf --max-time 3 http://localhost:8000/v1/models >/dev/null; then
    echo "ERROR: vLLM not reachable at http://localhost:8000. Start it first:"
    echo "  bash $REPO/serve_vllm.sh"
    exit 1
fi
echo "==> vLLM reachable at http://localhost:8000"

# R2E-Gym venv (client; not the vLLM one).
# shellcheck disable=SC1091
source "$REPO/R2E-Gym/.venv/bin/activate"
cd "$REPO/R2E-Gym"

# litellm requires OPENAI_API_KEY to be set even when the endpoint doesn't check.
export OPENAI_API_KEY="vllm-dummy-key"
# OPENAI_API_BASE is read by the agent (agent.py:99-100); default already
# points at localhost:8000 so we don't strictly need to set it.
export OPENAI_API_BASE="http://localhost:8000/v1"

TRAJ_DIR="$REPO/rollout_smoke/results"
mkdir -p "$TRAJ_DIR"

# `hosted_vllm/<served-model-name>` is the litellm prefix that routes to a
# self-hosted vLLM endpoint. The served name comes from --served-model-name
# in serve_vllm.sh.
python src/r2egym/agenthub/run/edit.py runagent_multiple \
    --traj_dir "$TRAJ_DIR" \
    --max_workers "$MAX_WORKERS" \
    --start_idx "$START_IDX" \
    --k "$K" \
    --dataset "/data/nikunj/SWE-Master-backup/datasets/SWE-Bench-Verified" \
    --split "test" \
    --llm_name "hosted_vllm/$TEACHER_NAME" \
    --use_fn_calling "$USE_FN_CALLING" \
    --exp_name "smoke-${TEACHER_NAME}-${TS}" \
    --temperature 0.6 \
    --max_steps "$MAX_STEPS" \
    --max_steps_absolute "$MAX_STEPS" \
    --backend "docker" \
    --prepull_images False \
    --scaffold "openhands" \
    --ip "127.0.0.1" \
    --use_lsp False \
    --used_yaml "$USED_YAML"

echo "==> Done at $(date -Iseconds)"
echo "==> Trajectories in: $TRAJ_DIR"
echo "==> Full log:        $LOG"
