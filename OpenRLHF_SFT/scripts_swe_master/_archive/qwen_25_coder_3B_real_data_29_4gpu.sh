#!/usr/bin/env bash
# SFT on the 29-reward==1 trajectory corpus, RESTRICTED to GPUs 4-7.
#
# Designed to run in parallel with vLLM serving on GPUs 0-3 (Phase 2 teacher
# rollouts). CUDA_VISIBLE_DEVICES=4,5,6,7 makes DeepSpeed see exactly 4 GPUs.
#
# Key knob changes vs the 8-GPU 6-traj run:
#   - train.batch_size 8 → 4 (keeps grad_accum_steps=1, our DeepSpeed patch fits)
#   - dataset: all_reward1_x5.jsonl (29 unique × 5 = 145 samples)
#   - separate output dir to avoid colliding with the 6-traj run's HF ckpt
set -euo pipefail

REPO=/data/nikunj/SWE-Master
OUT=$REPO/sft_real_data_29

# shellcheck disable=SC1091
source "$REPO/OpenRLHF_SFT/.venv/bin/activate"

mkdir -p "$OUT"/{ckpt,hf,logs}

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running.}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pin to GPUs 4-7 (the idle half). DeepSpeed will see these remapped as 0-3.
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Force CUDA 12.6 toolchain. Without this, DeepSpeed's JIT compile of cpu_adam
# (triggered by --ds.adam_offload) can pick up an older nvcc on PATH and fail
# with "CUDA version 11.5 does not match torch's 12.6".
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}

TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUT/logs/train_${TS}.log
ln -sfn "$(basename "$LOG")" "$OUT/logs/train_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"
echo "==> Using GPUs: $CUDA_VISIBLE_DEVICES"

deepspeed --module openrlhf.cli.train_sft \
    --data.max_len 32768 \
    --data.dataset "$REPO/rollout_smoke/sft_data/all_reward1_x5.jsonl" \
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
    --train.batch_size 4 \
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
    --logger.wandb.run_name "qwen25-coder-3b-real-data-29traj-${TS}"

echo "==> Done at $(date -Iseconds)"
echo "==> HF checkpoint: $OUT/hf"
echo "==> Full log:      $LOG"
