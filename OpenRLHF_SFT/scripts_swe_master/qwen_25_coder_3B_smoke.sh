#!/usr/bin/env bash
# Smoke-test SFT for Qwen2.5-Coder-3B-Instruct on 8x RTX 6000 Ada (48GB).
#
# Why 3B: the 7B variant (qwen_25_coder_7B_smoke.sh) completed step 1 but
# OOM'd on step 2 — the (seq_len × vocab_size) fp32 logits cost the same
# regardless of model size, and at 7B there was no slack left after the
# first optimizer step's persistent buffers. Dropping to 3B frees enough
# everywhere else to run multiple steps with the same patches.
#
# Patches still in effect (from the 7B effort):
#   - openrlhf/models/utils.py:    inplace_backward=True in CE call
#   - deepspeed/runtime/engine.py: short-circuit `grad / gas` when gas==1
set -euo pipefail

REPO=/data/nikunj/SWE-Master
OUT=$REPO/sft_smoke

# shellcheck disable=SC1091
source "$REPO/OpenRLHF_SFT/.venv/bin/activate"

mkdir -p "$OUT"/{data,ckpt,hf,logs}

: "${WANDB_API_KEY:?Set WANDB_API_KEY in your environment before running.}"

# Reduce CUDA allocator fragmentation.
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

deepspeed --module openrlhf.cli.train_sft \
    --data.max_len 32768 \
    --data.dataset "$OUT/data/demo_x20.jsonl" \
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
    --logger.wandb.run_name "qwen25-coder-3b-smoke-${TS}"

echo "==> Done at $(date -Iseconds)"
echo "==> Full log: $LOG"
