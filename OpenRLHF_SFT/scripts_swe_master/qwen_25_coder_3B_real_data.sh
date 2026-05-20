#!/usr/bin/env bash
# SFT for Qwen2.5-Coder-3B-Instruct on REAL rollout-derived data.
#
# Differences vs. the smoke run (qwen_25_coder_3B_smoke.sh):
#   - Dataset is the filtered reward==1 corpus produced by our K=5 + partial
#     K=50 sweeps with SWE-Master-32B-SFT as teacher (6 unique × 10 reps).
#   - Patches still in effect from the smoke (CE inplace_backward, deepspeed
#     gas==1 short-circuit, actor.py fp32-upcast disabled).
set -euo pipefail

# Auto-discover repo root: this script lives at <repo>/OpenRLHF_SFT/scripts_swe_master/
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
OUT=$REPO/sft_real_data

# shellcheck disable=SC1091
source "$REPO/OpenRLHF_SFT/.venv/bin/activate"

mkdir -p "$OUT"/{ckpt,hf,logs}

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running.}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUT/logs/train_${TS}.log
ln -sfn "$(basename "$LOG")" "$OUT/logs/train_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

deepspeed --module openrlhf.cli.train_sft \
    --data.max_len 32768 \
    --data.dataset "$REPO/rollout_smoke/sft_data/all_reward1_x10.jsonl" \
    --data.input_key input \
    --data.max_samples 200 \
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
    --logger.wandb.run_name "qwen25-coder-3b-real-data-${TS}"

echo "==> Done at $(date -Iseconds)"
echo "==> HF checkpoint: $OUT/hf"
echo "==> Full log:      $LOG"
