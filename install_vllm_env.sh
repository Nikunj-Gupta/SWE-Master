#!/usr/bin/env bash
# Install vLLM in its own venv at <repo>/vllm_venv.
# Kept separate from the OpenRLHF SFT and R2E-Gym client venvs so each can
# pin its own torch / transformers / huggingface_hub without colliding.
set -euo pipefail

# Auto-discover repo root: where this script lives.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=$REPO/sft_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/install_vllm_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/install_vllm_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

cd "$REPO"

echo "==> Creating vllm venv at $REPO/vllm_venv (Python 3.11)"
uv venv --python 3.11 --clear vllm_venv
# shellcheck disable=SC1091
source vllm_venv/bin/activate

echo "==> Installing vLLM with torch from cu126 channel"
# We force the cu126 wheel index because nvcc on this box is 12.6.
# The default PyPI torch resolves to cu13 and would crash with "CUDA version mismatch"
# (same trap we hit during the OpenRLHF install).
uv pip install vllm --extra-index-url https://download.pytorch.org/whl/cu126

echo "==> Pinning transformers<5"
# vLLM 0.11.0 expects the transformers 4.x tokenizer API (uses
# `all_special_tokens_extended`); transformers 5.x removed it.
# Same medicine we applied in the OpenRLHF and R2E-Gym venvs.
uv pip install 'transformers>=4.46,<5'

echo "==> Sanity check"
python - <<'PY'
import vllm, torch
print(f"vllm   {vllm.__version__}")
print(f"torch  {torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
PY

# CLI smoke
vllm --help | head -3 || true

echo "==> Done at $(date -Iseconds)"
echo "==> Activate with: source $REPO/vllm_venv/bin/activate"
echo "==> Full log: $LOG"
