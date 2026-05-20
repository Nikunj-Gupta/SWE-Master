#!/usr/bin/env bash
# Push the final SFT-trained model to HuggingFace Hub.
#
# Required env vars:
#   HF_TOKEN          Your HuggingFace write-access token
#   HF_USERNAME       Your HF account or org name (e.g. nikolamusk)
#
# Optional:
#   MODEL_DIR         Path to the safetensors checkpoint dir
#                     (default: ./sft_final/hf)
#   REPO_NAME         What to call it on HF (default: SWE-Master-Coder-3B-SFT-fork)
#   PRIVATE           '1' to create as private (default: 0 = public)
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   export HF_USERNAME=yourname
#   bash push_to_hf.sh
set -euo pipefail

REPO=$(cd "$(dirname "$0")" && pwd)

: "${HF_TOKEN:?HF_TOKEN must be set (your HuggingFace write token)}"
: "${HF_USERNAME:?HF_USERNAME must be set (your HF account/org name)}"

export MODEL_DIR="${MODEL_DIR:-$REPO/sft_final/hf}"
export REPO_NAME="${REPO_NAME:-SWE-Master-Coder-3B-SFT-fork}"
export PRIVATE="${PRIVATE:-0}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: MODEL_DIR not found: $MODEL_DIR"
    echo "  Either run the SFT pipeline first (qwen_25_coder_3B_final_sft.sh)"
    echo "  or set MODEL_DIR to your trained checkpoint."
    exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "ERROR: $MODEL_DIR doesn't look like a HF checkpoint (no config.json)"
    ls "$MODEL_DIR" | head
    exit 1
fi

# Copy the model card next to the model so it lands as README on the Hub
if [ -f "$REPO/MODEL_CARD.md" ]; then
    cp "$REPO/MODEL_CARD.md" "$MODEL_DIR/README.md"
    echo "==> staged MODEL_CARD.md → $MODEL_DIR/README.md"
fi

# Use the OpenRLHF venv since it has huggingface_hub installed via transformers
source "$REPO/OpenRLHF_SFT/.venv/bin/activate"

python - <<PY
import os
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = f"{os.environ['HF_USERNAME']}/{os.environ['REPO_NAME']}"
private = os.environ.get('PRIVATE', '0') == '1'

print(f"==> creating repo: {repo_id}  private={private}")
create_repo(
    repo_id=repo_id,
    token=os.environ['HF_TOKEN'],
    private=private,
    exist_ok=True,
)

print(f"==> uploading {os.environ['MODEL_DIR']!r}")
api.upload_folder(
    folder_path=os.environ['MODEL_DIR'],
    repo_id=repo_id,
    token=os.environ['HF_TOKEN'],
    commit_message="Upload SFT-trained model (Qwen2.5-Coder-3B-Instruct + 56 reward==1 trajectories)",
)
print()
print(f"==> done: https://huggingface.co/{repo_id}")
PY
