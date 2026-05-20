#!/usr/bin/env bash
# End-to-end orchestrator for the SWE-Master pipeline.
#
# Runs: verify env → serve vLLM teacher → rollout sweep → R2E→SFT convert
#       → (optional) SFT training on the resulting reward==1 corpus.
#
# All knobs are env vars with sensible defaults — override individually.
# Defaults pick a small smoke (K=1) so first-time runs finish in ~15 min.
# Bump K + MAX_WORKERS for real sweeps.
#
# Example real run:
#   export WANDB_API_KEY=...
#   K=50 MAX_WORKERS=16 bash run_full_pipeline.sh
#
# Example tighter smoke (single instance):
#   export WANDB_API_KEY=...
#   bash run_full_pipeline.sh
#
# Skip the SFT training pass:
#   SKIP_SFT=1 bash run_full_pipeline.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)

# ===================================================================
# Knobs
# ===================================================================
SKIP_VERIFY="${SKIP_VERIFY:-0}"
SKIP_SFT="${SKIP_SFT:-0}"

# Teacher
TEACHER_MODEL="${TEACHER_MODEL:-RUC-AIBOX/SWE-Master-32B-SFT}"
TEACHER_NAME="${TEACHER_NAME:-swe-master-32b-sft}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
TP_SIZE="${TP_SIZE:-8}"
GPU_MEM="${GPU_MEM:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

# Rollout
START_IDX="${START_IDX:-0}"
K="${K:-1}"
MAX_STEPS="${MAX_STEPS:-30}"
MAX_WORKERS="${MAX_WORKERS:-1}"
USE_FN_CALLING="${USE_FN_CALLING:-False}"
USED_YAML="${USED_YAML:-./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml}"

# SFT
SFT_BASE_MODEL="${SFT_BASE_MODEL:-Qwen/Qwen2.5-Coder-3B-Instruct}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-$REPO/sft_pipeline}"

# ===================================================================
# Logging
# ===================================================================
TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$REPO/sft_smoke/logs"
mkdir -p "$LOG_DIR"
PIPE_LOG="$LOG_DIR/pipeline_${TS}.log"
ln -sfn "$(basename "$PIPE_LOG")" "$LOG_DIR/pipeline_latest.log"
exec > >(tee -a "$PIPE_LOG") 2>&1

banner() {
    echo
    echo "========================================================================"
    echo "  $*"
    echo "========================================================================"
}

banner "SWE-Master pipeline orchestrator — $(date -Iseconds)"
echo "  log:        $PIPE_LOG"
echo "  teacher:    $TEACHER_NAME ($TEACHER_MODEL)"
echo "  TP=$TP_SIZE  GPUs=$GPUS  max_model_len=$MAX_MODEL_LEN"
echo "  rollout:    start_idx=$START_IDX  K=$K  max_steps=$MAX_STEPS  max_workers=$MAX_WORKERS"
echo "  scaffold:   use_fn_calling=$USE_FN_CALLING  yaml=$USED_YAML"
echo "  SFT:       $([ \"$SKIP_SFT\" = \"1\" ] && echo SKIPPED || echo \"$SFT_BASE_MODEL\")"

# ===================================================================
# 0. Verify environment
# ===================================================================
if [ "$SKIP_VERIFY" != "1" ]; then
    banner "0. Verify setup"
    python3 "$REPO/verify_setup.py" --quiet || {
        echo
        echo "verify_setup.py found problems. Fix them or set SKIP_VERIFY=1 to bypass."
        exit 1
    }
fi

: "${WANDB_API_KEY:?WANDB_API_KEY must be exported before running this pipeline.}"

# ===================================================================
# 1. Start vLLM (or detect existing)
# ===================================================================
banner "1. Start vLLM teacher endpoint"

if curl -sf --max-time 3 http://localhost:8000/v1/models >/dev/null; then
    cur_id=$(curl -sf --max-time 3 http://localhost:8000/v1/models | python3 -c "import sys,json; print(json.loads(sys.stdin.read())['data'][0]['id'])")
    if [ "$cur_id" = "$TEACHER_NAME" ]; then
        echo "vLLM already serving '$TEACHER_NAME' — reusing."
    else
        echo "vLLM is serving '$cur_id' but we want '$TEACHER_NAME' — stopping it."
        kill "$(cat "$REPO/vllm_venv/vllm.pid")" 2>/dev/null || true
        sleep 5
    fi
fi

if ! curl -sf --max-time 3 http://localhost:8000/v1/models >/dev/null; then
    TEACHER_MODEL="$TEACHER_MODEL" \
    TEACHER_NAME="$TEACHER_NAME" \
    GPUS="$GPUS" \
    TP_SIZE="$TP_SIZE" \
    GPU_MEM="$GPU_MEM" \
    MAX_MODEL_LEN="$MAX_MODEL_LEN" \
    bash "$REPO/serve_vllm.sh"

    echo "Waiting for vLLM to come up (up to 30 min for first-time model download)..."
    for i in $(seq 1 360); do
        if curl -sf --max-time 3 http://localhost:8000/v1/models >/dev/null 2>&1; then
            echo "vLLM is up after ${i}×5s."
            break
        fi
        sleep 5
        if [ $((i % 12)) -eq 0 ]; then
            echo "  ... still waiting (${i}×5s, tail of vllm log:)"
            tail -1 "$LOG_DIR/vllm_latest.log" 2>/dev/null | head -c 200
            echo
        fi
    done

    curl -sf --max-time 3 http://localhost:8000/v1/models >/dev/null || {
        echo "vLLM failed to come up. See $LOG_DIR/vllm_latest.log"
        exit 1
    }
fi

# ===================================================================
# 2. Rollout sweep
# ===================================================================
banner "2. Rollout sweep"
TEACHER_NAME="$TEACHER_NAME" \
MAX_STEPS="$MAX_STEPS" \
MAX_WORKERS="$MAX_WORKERS" \
START_IDX="$START_IDX" \
K="$K" \
USE_FN_CALLING="$USE_FN_CALLING" \
USED_YAML="$USED_YAML" \
bash "$REPO/run_rollout_smoke.sh"

# ===================================================================
# 3. Convert + filter
# ===================================================================
banner "3. Convert + filter trajectories"

# Find the most recent trajectory file matching this teacher
LATEST_TRAJ=$(ls -t "$REPO/rollout_smoke/results"/smoke-"$TEACHER_NAME"-*.jsonl 2>/dev/null | head -1)
if [ -z "$LATEST_TRAJ" ]; then
    echo "No trajectory file found for teacher '$TEACHER_NAME'."
    exit 1
fi
echo "trajectory file: $LATEST_TRAJ"

source "$REPO/R2E-Gym/.venv/bin/activate"
PREFIX="$REPO/rollout_smoke/sft_data/pipeline_${TS}"
python "$REPO/rollout_smoke/convert_rollouts_to_sft.py" \
    --src "$LATEST_TRAJ" \
    --dst-prefix "$PREFIX"
deactivate || true

FILTERED="${PREFIX}.openrlhf.filtered.jsonl"
N_REWARD1=$(wc -l <"$FILTERED" | tr -d ' ')
echo
echo "filtered corpus: $FILTERED"
echo "reward==1 count: $N_REWARD1"

if [ "$N_REWARD1" -eq 0 ]; then
    echo "WARNING: no reward==1 trajectories — SFT step would produce nothing."
    if [ "$SKIP_SFT" != "1" ]; then
        echo "Auto-skipping SFT. (Pass --K with more instances to get a real corpus.)"
        SKIP_SFT=1
    fi
fi

# ===================================================================
# 4. SFT (optional)
# ===================================================================
if [ "$SKIP_SFT" = "1" ]; then
    banner "4. SFT — SKIPPED"
else
    banner "4. SFT training"

    # Stop vLLM to free GPUs for training
    if [ -f "$REPO/vllm_venv/vllm.pid" ]; then
        echo "stopping vLLM to free GPUs..."
        kill "$(cat "$REPO/vllm_venv/vllm.pid")" 2>/dev/null || true
        sleep 5
    fi

    # Replicate the filtered corpus so the trainer can fill a batch.
    REPS=5
    REPLICATED="${PREFIX}.openrlhf.filtered.x${REPS}.jsonl"
    awk -v reps=$REPS '{for(i=0;i<reps;i++) print}' "$FILTERED" > "$REPLICATED"
    echo "replicated x$REPS → $REPLICATED ($(wc -l <"$REPLICATED" | tr -d ' ') samples)"

    # Make sure CUDA env is right (in case shell didn't have it).
    export CUDA_HOME=/usr/local/cuda
    export PATH=$CUDA_HOME/bin:$PATH
    export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    source "$REPO/OpenRLHF_SFT/.venv/bin/activate"
    mkdir -p "$SFT_OUTPUT_DIR"/{ckpt,hf,logs}

    SFT_LOG="$SFT_OUTPUT_DIR/logs/train_${TS}.log"
    deepspeed --module openrlhf.cli.train_sft \
        --data.max_len 32768 \
        --data.dataset "$REPLICATED" \
        --data.input_key input \
        --data.max_samples 500 \
        --data.apply_chat_template \
        --data.multiturn \
        --model.model_name_or_path "$SFT_BASE_MODEL" \
        --model.gradient_checkpointing_enable \
        --ckpt.output_dir "$SFT_OUTPUT_DIR/hf" \
        --ckpt.path "$SFT_OUTPUT_DIR/ckpt" \
        --ckpt.save_steps -1 \
        --ckpt.max_num 1 \
        --ckpt.save_hf \
        --train.batch_size 8 \
        --train.micro_batch_size 1 \
        --train.max_epochs 2 \
        --eval.steps -1 \
        --ds.zero_stage 3 \
        --ds.adam_offload \
        --ds.param_dtype bf16 \
        --ds.attn_implementation flash_attention_2 \
        --ds.use_liger_kernel \
        --ds.packing_samples \
        --optim adam \
        --adam.lr 2e-5 \
        --logger.logging_steps 1 \
        --logger.wandb.key "$WANDB_API_KEY" \
        --logger.wandb.org _nikunj \
        --logger.wandb.project swe-master-sft-smoke \
        --logger.wandb.run_name "pipeline-${TS}" \
        2>&1 | tee "$SFT_LOG"

    echo "SFT done. HF ckpt at $SFT_OUTPUT_DIR/hf"
fi

banner "Pipeline complete"
echo "  rollout JSONL:  $LATEST_TRAJ"
echo "  SFT corpus:     $FILTERED"
echo "  SFT model:      $([ \"$SKIP_SFT\" = \"1\" ] && echo SKIPPED || echo \"$SFT_OUTPUT_DIR/hf\")"
echo "  pipeline log:   $PIPE_LOG"
echo "  finished:       $(date -Iseconds)"
