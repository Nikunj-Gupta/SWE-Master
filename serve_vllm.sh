#!/usr/bin/env bash
# Serve a Qwen2.5-Coder model via vLLM in the background.
#
# Defaults to the 7B Instruct on GPU 0. Override via env vars:
#   TEACHER_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct \
#   TEACHER_NAME=qwen25-coder-14b-instruct \
#   bash serve_vllm.sh
#
# For multi-GPU (tensor parallel), also set:
#   GPUS=0,1   TP_SIZE=2
set -euo pipefail

TEACHER_MODEL=${TEACHER_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
TEACHER_NAME=${TEACHER_NAME:-qwen25-coder-7b-instruct}
GPUS=${GPUS:-0}
TP_SIZE=${TP_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
GPU_MEM=${GPU_MEM:-0.85}

# Auto-discover repo root: where this script lives.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=$REPO/sft_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/vllm_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/vllm_latest.log"

# shellcheck disable=SC1091
source "$REPO/vllm_venv/bin/activate"

# Pin teacher to the requested GPU(s).
export CUDA_VISIBLE_DEVICES=$GPUS
# Re-use a shared HF cache so the model is already on disk between runs.
# Default to <repo>/hf_cache for portability; override by exporting HF_HOME
# (recommended: point to a partition with ≥ 300 GB free).
export HF_HOME=${HF_HOME:-$REPO/hf_cache}
mkdir -p "$HF_HOME"

# vLLM's inductor pipeline JIT-compiles a Triton helper that links against
# -lcuda. The system has libcuda.so.1 (driver) but no build-time libcuda.so
# symlink. CUDA ships a linker stub at $CUDA_HOME/lib64/stubs/libcuda.so —
# put it on LIBRARY_PATH so gcc's `-lcuda` resolves at compile time.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}

echo "==> vllm log:      $LOG"
echo "==> Model:         $TEACHER_MODEL"
echo "==> Served as:     $TEACHER_NAME"
echo "==> GPUs:          $GPUS (TP=$TP_SIZE)  max_model_len=$MAX_MODEL_LEN"
echo "==> Started at $(date -Iseconds)"

# Background launch. Stop with:  kill $(cat $REPO/vllm_venv/vllm.pid)
nohup vllm serve "$TEACHER_MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len "$MAX_MODEL_LEN" \
    --served-model-name "$TEACHER_NAME" \
    >> "$LOG" 2>&1 &

PID=$!
echo $PID > "$REPO/vllm_venv/vllm.pid"
echo "==> vllm pid: $PID"
echo "==> Tail log:   tail -f $LOG"
echo "==> Health URL: curl http://localhost:8000/v1/models  (ready when this returns)"
echo "==> Stop with:  kill \$(cat $REPO/vllm_venv/vllm.pid)"
