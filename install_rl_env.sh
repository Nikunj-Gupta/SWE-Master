#!/usr/bin/env bash
# Install the rLLM (DeepSWE) RL training environment.
# Creates a uv venv at DeepSWE_RL/.venv, clones the verl submodule (which is
# referenced by rllm/.gitmodules but never pulled in the inlined fork), and
# installs verl + rllm in editable mode plus all transitive deps the trainer
# actually needs at runtime.
#
# Pinning notes:
#   - torch:  hard-pinned to torch==2.8.0+cu126. The +cu126 local-version
#             label is exclusive to the pytorch.org cu126 channel; if any
#             subsequent uv resolve tried to substitute a plain (cu128-bundled)
#             PyPI wheel, that wheel does NOT carry +cu126 and the pin fails.
#             We also export UV_INDEX_URL globally so EVERY uv pip install in
#             this script sees the cu126 channel as primary.
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

# --------------------------------------------------------------------
# Preflight: refuse to start unless nvcc is 12.6.x. We pin torch to a
# cu126 build below; running this script against a different toolkit
# (e.g. 12.4, 12.8, 13.x) produces ABI mismatches at flash-attn build
# time or silent runtime crashes in kernels — both 45+ min into the
# install. Failing fast here costs 5 s.
# --------------------------------------------------------------------
NVCC_VER=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9.]+' | head -1 || true)
case "${NVCC_VER:-}" in
    12.6*) echo "==> nvcc preflight OK: $NVCC_VER" ;;
    "")    echo "FATAL: nvcc not on PATH. Install CUDA 12.6 toolkit and re-run." >&2
           echo "       hint: export PATH=/usr/local/cuda-12.6/bin:\$PATH" >&2
           exit 1 ;;
    *)     echo "FATAL: nvcc is ${NVCC_VER}, this script requires 12.6.x." >&2
           echo "       torch wheels are pinned to cu126; different toolkit -> ABI mismatches." >&2
           echo "       hint: export PATH=/usr/local/cuda-12.6/bin:\$PATH" >&2
           exit 1 ;;
esac

# --------------------------------------------------------------------
# Force EVERY uv pip install in this script to see the cu126 channel
# alongside PyPI. We use --index-strategy=unsafe-best-match because uv's
# default `first-index` strategy refuses to "fall through" to a second
# index for a package that exists on the first — i.e. with PyPI in the
# mix, uv finds plain `torch==2.8.0` on PyPI and rejects our pin to
# `torch==2.8.0+cu126` instead of looking at the cu126 channel:
#   "torch was found on https://pypi.org/simple, but not at the
#    requested version (torch==2.8.0+cu126)."
# `unsafe-best-match` tells uv to consider ALL versions from ALL indexes;
# the `+cu126` local-version label in the pin then guarantees only the
# right wheel matches. Safe here because both indexes are first-party.
# --------------------------------------------------------------------
export UV_INDEX_URL="https://download.pytorch.org/whl/cu126"
export UV_EXTRA_INDEX_URL="https://pypi.org/simple"
export UV_INDEX_STRATEGY="unsafe-best-match"

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
# 3a. Install torch FIRST, alone, hard-pinned to a cu126 build.
#     Pinning to the local-version label (`+cu126`) is what makes this
#     robust: the cu128/cu13x wheels on the default PyPI channel are
#     plain `2.8.0` (no local label) and can't match `==2.8.0+cu126`,
#     so even if a later joint resolve re-considers torch it can't
#     silently substitute a cu128 wheel.
# --------------------------------------------------------------------
echo "==> Installing torch==2.8.0+cu126 (cu126 channel, hard-pinned)"
uv pip install 'torch==2.8.0+cu126'

# Assert the install matches our toolkit BEFORE we spend 30 min on a
# flash-attn build that would otherwise mismatch.
echo "==> Verifying torch picked up cu126"
python - <<'PY'
import sys, torch
v = torch.version.cuda
if v != "12.6":
    sys.stderr.write(f"FATAL: torch installed with cuda={v}, expected 12.6.\n")
    sys.stderr.write("       cu126 pin failed — check UV_INDEX_URL and the wheel actually fetched.\n")
    sys.exit(2)
print(f"OK: torch {torch.__version__} cuda={v}")
PY

# --------------------------------------------------------------------
# 3. Install vllm now that torch is locked. vllm is PyPI-only so uv
#    will fall through to the extra index automatically; with torch
#    already satisfied, vllm's resolver leaves it alone.
# --------------------------------------------------------------------
echo "==> Installing vllm (PyPI; torch already pinned)"
uv pip install 'vllm>=0.8.3,<0.11'

# Build essentials (needed under --no-build-isolation).
# setuptools<81: verl's __init__.py uses `import pkg_resources` which
# setuptools 81 deprecated and 82 removed. Pinning to 80.x keeps the API.
echo "==> Installing build essentials"
uv pip install wheel packaging ninja numpy 'setuptools<81'

# --------------------------------------------------------------------
# 3b. Build flash-attn FROM SOURCE against the just-installed torch.
#     If we let verl's editable install resolve it transitively, uv
#     picks the prebuilt flash-attn 2.8.3 wheel (built against torch
#     2.7) → ABI mismatch at import time with our torch 2.8+cu128:
#     `undefined symbol: _ZN3c104cuda29c10_cuda_check_implementation...`
#     Installing it explicitly first with --no-build-isolation forces
#     a source build that links against the venv's actual torch.
#     ~5-10 min compile.
# --------------------------------------------------------------------
echo "==> Building flash-attn from source against installed torch"
uv pip install --no-build-isolation --no-cache-dir flash-attn

# --------------------------------------------------------------------
# 4. Install verl (editable). Uses already-installed torch + vllm
#    + flash-attn (we pre-built above, uv will leave it alone).
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

# swesmith.profiles imports swesmith.bug_gen.adapters at module load, which
# unconditionally imports every language adapter (c/cpp/go/js/php/ruby/rust/...).
# Each pulls its tree_sitter_<lang> binding. The --no-deps above skips them,
# so the trainer crashes on `import r2egym.agenthub.environment.env` at
# `from swesmith.bug_gen.adapters.c import ...`. Install them explicitly.
echo "==> Installing tree-sitter bindings needed by swesmith.bug_gen.adapters"
uv pip install --no-deps \
    'tree-sitter>=0.25' \
    tree-sitter-c \
    tree-sitter-cpp \
    tree-sitter-c-sharp \
    tree-sitter-go \
    tree-sitter-java \
    tree-sitter-javascript \
    'tree-sitter-php>=0.23.11' \
    tree-sitter-ruby \
    tree-sitter-rust

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
import sys, torch
print(f"torch          {torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}")
# Final guardrail: a downstream install must not have replaced our cu126
# torch with a plain cu128 wheel. Hard-fail if it has.
if torch.version.cuda != "12.6":
    sys.stderr.write(f"FATAL: post-install torch cuda is {torch.version.cuda}, expected 12.6.\n")
    sys.stderr.write("       Something later in this script clobbered the cu126 pin.\n")
    sys.exit(2)
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
