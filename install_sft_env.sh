#!/usr/bin/env bash
# Install the OpenRLHF SFT training environment for the smoke run.
# Creates a uv venv at OpenRLHF_SFT/.venv, installs OpenRLHF (which pulls
# torch + deepspeed), then flash-attn and ring-flash-attn on top.
#
# Assumes: nvcc 12.x available, uv on PATH, internet access for PyPI/GitHub.
set -euo pipefail

# Auto-discover repo root: where this script lives.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=$REPO/sft_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/install_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/install_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

cd "$REPO/OpenRLHF_SFT"

echo "==> Creating venv with Python 3.11 (--clear, idempotent)"
uv venv --python 3.11 --clear
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> Pre-installing torch (cu126 wheel) so flash-attn/deepspeed builds can import it"
# We pin to PyTorch's cu126 channel because:
#   1. Local nvcc is 12.6 — torch must be compiled for CUDA 12.x or
#      flash-attn's build refuses with a CUDA major-version mismatch.
#   2. flash-attn 2.8.3 publishes prebuilt wheels only for cu12*torch2.5/2.6/2.7;
#      the default cu13 torch 2.12 has no prebuilt wheel and forces a long source compile.
uv pip install torch --index-url https://download.pytorch.org/whl/cu126

echo "==> Installing build essentials (needed under --no-build-isolation)"
# Without these, setup.py-style builds (OpenRLHF, flash-attn, deepspeed) fail
# inside the venv: wheel/packaging missing, ninja absent for C++ compilation,
# numpy required by flash-attn's setup imports.
uv pip install wheel packaging ninja numpy

echo "==> Cloning OpenRLHF"
if [ ! -d OpenRLHF ]; then
    git clone https://github.com/OpenRLHF/OpenRLHF.git
fi

echo "==> Installing OpenRLHF (--no-build-isolation; pulls flash-attn + deepspeed)"
(cd OpenRLHF && uv pip install -e . --no-build-isolation)

echo "==> Pinning transformers <5"
# OpenRLHF unpinned dep pulled transformers 5.7.0, which (a) was almost
# certainly not what OpenRLHF 0.10.3 was tested against, and (b) removed
# `is_flash_attn_greater_or_equal_2_10`, breaking ring-flash-attn import.
# 4.46 is the floor for Qwen2.5; <5 keeps the flash-attn helpers intact.
uv pip install 'transformers>=4.46,<5'

echo "==> Installing ring-flash-attn (kept available for longer-context runs)"
uv pip install ring-flash-attn --no-build-isolation

echo "==> Installing liger-kernel (fused/chunked CE; avoids OOM on logits backward)"
uv pip install liger-kernel

echo "==> Sanity check"
python - <<'PY'
import torch, deepspeed, flash_attn, ring_flash_attn, openrlhf
print(f"torch          {torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
print(f"deepspeed      {deepspeed.__version__}")
print(f"flash_attn     {flash_attn.__version__}")
print("ring_flash_attn ok")
print(f"openrlhf       {openrlhf.__file__}")
PY

echo "==> Done at $(date -Iseconds)"
echo "==> Activate with: source $REPO/OpenRLHF_SFT/.venv/bin/activate"
echo "==> Full log: $LOG"
