#!/usr/bin/env bash
# Smoke-test SFT for Qwen2.5-Coder-7B-Instruct on 8x RTX 6000 Ada (48GB).
#
# Goal: prove the training loop runs end-to-end and loss decreases on a
# replicated copy of the shipped demo trajectories. Not a real training run.
#
# Differences vs. the 32B reference (qwen_25_coder_32B_new_remove_01_not_dedep.sh):
#   - 7B Instruct from HF instead of a local 32B-80K checkpoint
#   - max_len 32768 (was 81920)
#   - no ring attention (flash-attn alone is plenty at 7B/32K)
#   - train_batch_size 8 (was 256), max_epochs 1 (was 5), max_samples 200
#   - learning_rate 2e-5 (was 5e-5)
#   - all paths under /data/nikunj/SWE-Master/sft_smoke
set -euo pipefail

REPO=/data/nikunj/SWE-Master
OUT=$REPO/sft_smoke

# shellcheck disable=SC1091
source "$REPO/OpenRLHF_SFT/.venv/bin/activate"

mkdir -p "$OUT"/{data,ckpt,hf,logs}

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running.}"

# Reduce CUDA allocator fragmentation — PyTorch's own OOM message recommends this
# whenever "reserved but unallocated" memory is non-trivial.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TS=$(date +%Y%m%d_%H%M%S)
LOG=$OUT/logs/train_${TS}.log
ln -sfn "$(basename "$LOG")" "$OUT/logs/train_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

# Build the smoke corpus if it isn't there yet.
if [ ! -f "$OUT/data/demo_x20.jsonl" ]; then
    python "$REPO/OpenRLHF_SFT/scripts_swe_master/replicate_demo_data.py" \
        --dst "$OUT/data/demo_x20.jsonl" \
        --copies 20
fi

# OpenRLHF 0.10.3 uses a namespaced/dotted CLI. The 32B reference script in
# the repo predates this refactor — every flag below has been translated.
deepspeed --module openrlhf.cli.train_sft \
    --data.max_len 32768 \
    --data.dataset "$OUT/data/demo_x20.jsonl" \
    --data.input_key input \
    --data.max_samples 200 \
    --data.apply_chat_template \
    --data.multiturn \
    --model.model_name_or_path Qwen/Qwen2.5-Coder-7B-Instruct \
    --model.gradient_checkpointing_enable \
    --ckpt.output_dir "$OUT/hf" \
    --ckpt.path "$OUT/ckpt" \
    --ckpt.save_steps -1 \
    --ckpt.max_num 1 \
    --ckpt.save_hf \
    --train.batch_size 8 \
    --train.micro_batch_size 1 \
    --train.max_epochs 1 \
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
    --logger.wandb.run_name "qwen25-coder-7b-smoke-${TS}"

echo "==> Done at $(date -Iseconds)"
echo "==> Full log: $LOG"
