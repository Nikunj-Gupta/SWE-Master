#!/usr/bin/env bash
# Install the rLLM (DeepSWE) RL training environment.
# Creates a uv venv at DeepSWE_RL/.venv, clones the verl submodule (which is
# referenced by rllm/.gitmodules but never pulled in the inlined fork), and
# installs verl + rllm in editable mode plus all transitive deps the trainer
# actually needs at runtime.
#
# Pinning notes:
#   - torch:  cu126 channel (>=2.7,<2.9). Matches our existing CUDA toolkit
#             and vllm's expected runtime. Newer cu13 torch will be silently
#             pulled by unbounded pyproject deps if we don't pin first.
#   - vllm:   >=0.8.3, <0.11. The rllm-pinned verl fork was developed against
#             vllm 0.8–0.10 and uses APIs (`vllm.worker.worker_base`,
#             `vllm.lora.models.LoRAModel`, `vllm.inputs.SingletonInputs`)
#             that were removed in vllm 0.13+. Bumping past 0.11 means deep
#             rewrites of verl's rollout + sharding code; out of scope.
#   - transformers: <5. Same tokenizer-API constraint as our SFT venv.
#   - swebench: ==3.0.2. R2E-Gym needs `get_eval_type` which was dropped in
#             swebench 4.x.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"

# Logs go under rl_smoke/ to keep them separate from sft_smoke/.
LOG_DIR=$REPO/rl_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/install_rl_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/install_rl_latest.log"
exec > >(tee -a "$LOG") 2>&1
echo "==> Logging to $LOG"
echo "==> Started at $(date -Iseconds)"
echo "==> HF_HOME=$HF_HOME  TRITON_CACHE_DIR=$TRITON_CACHE_DIR"

# --------------------------------------------------------------------
# 1. Clone verl submodule (rllm/.gitmodules references it; never pulled
#    because the fork inlined DeepSWE_RL/ as a regular directory)
# --------------------------------------------------------------------
cd "$REPO/DeepSWE_RL/rllm"
if [ ! -d verl/.git ]; then
    echo "==> Cloning verl (agentica-project/verl @ main)"
    git clone -b main https://github.com/agentica-project/verl verl
else
    echo "==> verl already cloned"
fi

# --------------------------------------------------------------------
# 2. Create venv (Python 3.10 per rllm README)
# --------------------------------------------------------------------
cd "$REPO/DeepSWE_RL"
echo "==> Creating venv at $(pwd)/.venv (Python 3.10, --clear, idempotent)"
uv venv --python 3.10 --clear .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# --------------------------------------------------------------------
# 3. Pre-install torch + vllm pinned together so downstream resolves
#    can't drift the stack to cu13/newer-vllm.
# --------------------------------------------------------------------
echo "==> Pre-installing torch + vllm (cu126, vllm < 0.11)"
uv pip install \
    'torch>=2.7,<2.9' \
    'vllm>=0.8.3,<0.11' \
    --index-url https://download.pytorch.org/whl/cu126 \
    --extra-index-url https://pypi.org/simple

# Build essentials (needed under --no-build-isolation).
# setuptools<81: verl's __init__.py uses `import pkg_resources` which
# setuptools 81 deprecated and 82 removed. Pinning to 80.x keeps the API.
echo "==> Installing build essentials"
uv pip install wheel packaging ninja numpy 'setuptools<81'

# --------------------------------------------------------------------
# 4. Install verl (editable). Uses already-installed torch + vllm.
# --------------------------------------------------------------------
echo "==> Installing verl (editable)"
(cd rllm/verl && uv pip install -e . --no-build-isolation)

# --------------------------------------------------------------------
# 5. Install rllm (editable). Brings ray, hydra, omegaconf, etc.
# --------------------------------------------------------------------
echo "==> Installing rllm (editable)"
(cd rllm && uv pip install -e . --no-build-isolation)

# --------------------------------------------------------------------
# 6. Pin transformers<5 (verl + rllm tested against 4.x tokenizer API)
# --------------------------------------------------------------------
echo "==> Pinning transformers<5"
uv pip install 'transformers>=4.46,<5'

# --------------------------------------------------------------------
# 7. Pin swebench to the version R2E-Gym was tested against.
#    swebench 4.x dropped swebench.harness.log_parsers.get_eval_type.
# --------------------------------------------------------------------
echo "==> Pinning swebench == 3.0.2"
uv pip install 'swebench==3.0.2'

# --------------------------------------------------------------------
# 8. Trainer runtime deps that the original ray.init(runtime_env={"pip": [...]})
#    would have pip-installed into an isolated worker venv. We patch ray.init
#    to skip that (see step 10) and install these directly here instead.
#    --no-deps avoids each one pulling its own preferred torch/transformers.
# --------------------------------------------------------------------
echo "==> Installing trainer runtime deps (no-deps)"
uv pip install --no-deps \
    'litellm>=1.58.2' \
    'seaborn>=0.13.2' \
    'gpustat>=1.1.1' \
    'simple-parsing>=0.1.6' \
    'together>=1.3.5' \
    'pexpect>=4.9.0' \
    'libtmux>=0.40.1' \
    'bashlex>=0.18' \
    sentence-transformers \
    chardet unidiff \
    "$REPO/swebench_fork_swegym-2.0.13-py3-none-any.whl" \
    "$REPO/swebench_fork_swerebench-4.0.3-py3-none-any.whl" \
    "$REPO/swesmith-0.0.7-py3-none-any.whl"

# --------------------------------------------------------------------
# 9. Install r2egym as a proper package (--no-deps to keep our cu126
#    stack intact). The trainer imports r2egym for the sweagent rollout.
# --------------------------------------------------------------------
echo "==> Installing r2egym (no deps)"
uv pip install -e "$REPO/R2E-Gym" --no-deps

# --------------------------------------------------------------------
# 10. Apply all RL-side patches via the unified entry point.
#     See apply_patches_rl.py for the patch registry. All patches are
#     idempotent (sentinel-tagged), so re-running this is safe.
# --------------------------------------------------------------------
echo "==> Applying RL patches"
bash "$REPO/apply_patches_rl.sh"

# --------------------------------------------------------------------
# 11. Sanity check — versions + the legacy vllm imports verl needs
# --------------------------------------------------------------------
echo "==> Sanity check"
python - <<'PY'
import torch
print(f"torch          {torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
import vllm
print(f"vllm           {vllm.__version__}")
import rllm
print(f"rllm package   {rllm.__path__[0]}")
import verl
print(f"verl           {verl.__file__}")
try:
    import flash_attn
    print(f"flash_attn     {flash_attn.__version__}")
except Exception as e:
    print(f"flash_attn     <err {e}>")
try:
    import deepspeed
    print(f"deepspeed      {deepspeed.__version__}")
except Exception as e:
    print(f"deepspeed      <err {e}>")

# The three imports that fail on vllm >= 0.13 — confirm they resolve here.
from vllm.worker.worker_base import WorkerWrapperBase  # noqa: F401
from vllm.lora.models import LoRAModel  # noqa: F401
from vllm.inputs import SingletonInputs  # noqa: F401
print("legacy vllm imports OK (WorkerWrapperBase, LoRAModel, SingletonInputs)")

import r2egym
print(f"r2egym         {r2egym.__file__}")
import swebench
print(f"swebench       {swebench.__version__}")
from importlib.metadata import version
for p in ['ray', 'transformers', 'datasets', 'sglang']:
    try: print(f"{p:14s} {version(p)}")
    except Exception as e: print(f"{p:14s} <err {e}>")
print()
import rllm.trainer.verl.train_agent_ppo as t
print(f"trainer entry  {t.__file__}")
PY

echo "==> Done at $(date -Iseconds)"
echo "==> Activate with: source $REPO/DeepSWE_RL/.venv/bin/activate"
echo "==> Full log: $LOG"
