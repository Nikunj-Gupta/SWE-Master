# Source this for SWE-Master work — keeps every framework's scratch dir off
# space-constrained partitions (e.g. a small / NVMe).
#
# Cache root resolution (in order of precedence):
#   1. $SWE_MASTER_CACHE_ROOT if already set
#   2. /data/$USER/cache if /data exists and is writable (typical lab box)
#   3. $HOME/.cache/swe-master   (portable fallback)
#
# Operators on a specific box can override by exporting SWE_MASTER_CACHE_ROOT
# in ~/.bashrc *before* this file is sourced.

_env_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)

if [ -z "${SWE_MASTER_CACHE_ROOT:-}" ]; then
    if [ -d /data ] && [ -w /data ]; then
        SWE_MASTER_CACHE_ROOT=/data/$USER/cache
    else
        SWE_MASTER_CACHE_ROOT=$HOME/.cache/swe-master
    fi
    export SWE_MASTER_CACHE_ROOT
fi
mkdir -p "$SWE_MASTER_CACHE_ROOT"/{hf,triton,torch_extensions,xdg,tmp,pip,uv,wandb,vllm,torch}

export HF_HOME=$SWE_MASTER_CACHE_ROOT/hf
export TRITON_CACHE_DIR=$SWE_MASTER_CACHE_ROOT/triton
export TORCH_EXTENSIONS_DIR=$SWE_MASTER_CACHE_ROOT/torch_extensions
export PIP_CACHE_DIR=$SWE_MASTER_CACHE_ROOT/pip
export UV_CACHE_DIR=$SWE_MASTER_CACHE_ROOT/uv
export XDG_CACHE_HOME=$SWE_MASTER_CACHE_ROOT/xdg
export TMPDIR=$SWE_MASTER_CACHE_ROOT/tmp

# CUDA + uv on PATH so future shells don't fall back to a system nvcc that's
# older than what our cu126/cu130 wheels expect.
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH=$HOME/.local/bin:$CUDA_HOME/bin:$PATH

# Build-time libcuda.so stub on LIBRARY_PATH so Inductor (PyTorch JIT) and
# vLLM's Triton compile path can link `-lcuda` successfully. The runtime
# libcuda.so.1 comes from the driver; the stub is what the linker needs.
# Without this you'll see: `/usr/bin/ld: cannot find -lcuda`.
export LIBRARY_PATH=$CUDA_HOME/lib64/stubs:${LIBRARY_PATH:-}

unset _env_dir
