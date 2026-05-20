#!/usr/bin/env bash
# Install the R2E-Gym rollout environment in a single resolution pass.
#
# Why one-shot: R2E-Gym's pyproject leaves most deps unbounded, so every
# subsequent `uv pip install ...` re-resolves and pulls in incompatible
# latest versions (transformers 5, huggingface_hub 1, openai 2, scipy 1.16,
# etc.). Doing the editable install, the 3 SWE wheels, and all compat pins
# in ONE call gives uv the full constraint set up front.
set -euo pipefail

# Auto-discover repo root: where this script lives.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
LOG_DIR=$REPO/sft_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/install_rollout_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/install_rollout_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"

cd "$REPO/R2E-Gym"

echo "==> Creating fresh venv at $(pwd)/.venv (Python 3.11)"
uv venv --python 3.11 --clear
# shellcheck disable=SC1091
source .venv/bin/activate

echo "==> One-shot install: r2egym (editable) + 3 SWE wheels + compat pins"
# Pin reasoning:
#   huggingface_hub<1.0  : R2E-Gym uses HfFolder (removed in 1.0)
#   transformers<5       : transformers 5 changes tokenizer API (breaks vLLM + r2egym),
#                          and requires huggingface_hub>=1
#   openai<2             : litellm 1.x imports openai._models (gone in openai 2)
#   datasets==2.19       : R2E-Gym pyproject pins this exact version
#   pandas<3             : pandas 3 removed pandas.api.extensions (datasets 2.19 needs it)
#   pyarrow>=17          : earlier pyarrow wheels were built against numpy 1 ABI
#   scipy<1.15           : scipy 1.15 removed scipy.sparse._sputils (sklearn 1.5 needs it)
uv pip install -e . \
    "$REPO/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
    "$REPO/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
    "$REPO/swesmith-0.0.7-py3-none-any.whl" \
    'huggingface_hub<1.0' \
    'transformers>=4.46,<5' \
    'openai<2' \
    'datasets==2.19' \
    'pandas<3' \
    'pyarrow>=17' \
    'scipy<1.15'

echo "==> Sanity check"
python - <<'PY'
import r2egym
from r2egym.agenthub.agent.agent import Agent, AgentArgs
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from importlib.metadata import version
for p in ['r2e-gym','litellm','swebench','transformers','huggingface_hub','numpy','pandas','pyarrow','openai','datasets','scipy','scikit-learn']:
    try: print(f"{p:18s} {version(p)}")
    except Exception as e: print(f"{p:18s} <err {e}>")
print("Agent + Env classes import OK")
print("(Note: dataset load is intentionally NOT tested here — set SWEBENCH_DATASET_PATH")
print(" before running run_rollout_smoke.sh.)")
PY

echo "==> Done at $(date -Iseconds)"
echo "==> Activate with: source $REPO/R2E-Gym/.venv/bin/activate"
echo "==> Full log: $LOG"
