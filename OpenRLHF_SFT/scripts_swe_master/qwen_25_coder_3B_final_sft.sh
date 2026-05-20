#!/usr/bin/env bash
# Final SFT pass on the combined 56-reward==1 trajectory corpus
# (Phase 1 SWE-Master-32B-SFT + Phase 2 Qwen3-Coder-30B-A3B teachers, multi-repo).
#
# Uses all 8 GPUs (ZeRO-3, world_size=8) — same memory pattern as the
# successful 6-trajectory run. 56 unique × 5 reps = 280 training samples.
set -euo pipefail

# Auto-discover repo root: this script lives at <repo>/OpenRLHF_SFT/scripts_swe_master/
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$REPO/sft_final

# shellcheck disable=SC1091
source "$REPO/OpenRLHF_SFT/.venv/bin/activate"

mkdir -p "$OUT"/{ckpt,hf,logs}

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running.}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Force CUDA 12.6 toolchain (avoids picking up Ubuntu's nvidia-cuda-toolkit
# /usr/bin/nvcc -> 11.5 wrapper if it's earlier on PATH).
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}

TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUT/logs/train_${TS}.log
ln -sfn "$(basename "$LOG")" "$OUT/logs/train_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

deepspeed --module openrlhf.cli.train_sft \
    --data.max_len 32768 \
    --data.dataset "$REPO/rollout_smoke/sft_data/all_reward1_56_x5.jsonl" \
    --data.input_key input \
    --data.max_samples 500 \
    --data.apply_chat_template \
    --data.multiturn \
    --model.model_name_or_path Qwen/Qwen2.5-Coder-3B-Instruct \
    --model.gradient_checkpointing_enable \
    --ckpt.output_dir "$OUT/hf" \
    --ckpt.path "$OUT/ckpt" \
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
    --logger.wandb.run_name "qwen25-coder-3b-final-56traj-${TS}"

echo "==> Done at $(date -Iseconds)"
echo "==> HF checkpoint: $OUT/hf"
echo "==> Full log:      $LOG"
