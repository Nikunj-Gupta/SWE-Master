"""Patch DeepSWE_RL/rllm's `train_agent_ppo.py` to stop using ray's pip
runtime-env (which spins up an isolated virtualenv per worker and re-installs
two dozen packages from scratch every run).

Why: the original `ray.init(runtime_env={"pip": [...], "working_dir": ..., "env_vars": ...})`
call (a) silently fails when Ray's auto-virtualenv has no pip module installed,
(b) wastes minutes per launch re-resolving packages we already have in the
parent venv, and (c) embeds a placeholder `WANDB_API_KEY: "xx"` that overwrites
the real key in the parent env. We drop only the "pip" key and the dummy
WANDB_API_KEY, keeping working_dir + the rest of env_vars.

Idempotent: safe to re-run.
"""
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(REPO, "DeepSWE_RL", "rllm", "rllm", "trainer", "verl", "train_agent_ppo.py")
SENTINEL = "# patched-by: patch_rllm_ray_init.py"


def main() -> int:
    if not os.path.exists(TARGET):
        print(f"ERROR: target not found: {TARGET}")
        return 1
    with open(TARGET) as f:
        src = f.read()
    if SENTINEL in src:
        print(f"already patched: {TARGET}")
        return 0

    # The whole multi-line ray.init(runtime_env=...) call is one Python expression.
    # We replace it with a leaner one that drops the pip clause and the
    # placeholder WANDB_API_KEY, and tags itself with SENTINEL so we can detect
    # re-runs. We anchor on the literal "ray.init(runtime_env=" prefix and the
    # closing `)` at the end of the original call.
    pattern = re.compile(
        r"ray\.init\(runtime_env=\{[^}]*\"pip\"[^}]*\}[^}]*\}\)",
        re.DOTALL,
    )
    new_call = (
        f'# {SENTINEL}\n'
        '        # Build env_vars: fixed defaults + selected inheritances from the\n'
        '        # parent shell. The inherited keys are critical for vLLM\'s Inductor\n'
        '        # JIT compile path (needs CUDA_HOME, LIBRARY_PATH=$CUDA_HOME/lib64/stubs\n'
        '        # to link `-lcuda`, LD_LIBRARY_PATH for the cu* runtime libs).\n'
        '        _ray_env = {"TOKENIZERS_PARALLELISM": "true",\n'
        '                    "NCCL_DEBUG": "WARN",\n'
        '                    "PYTHONPATH": "./DeepSWE_RL/rllm"}\n'
        '        for _k in ("CUDA_HOME", "LIBRARY_PATH", "LD_LIBRARY_PATH",\n'
        '                   "HF_HOME", "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR",\n'
        '                   "VLLM_USE_V1", "VLLM_ATTENTION_BACKEND",\n'
        '                   "VLLM_ENGINE_ITERATION_TIMEOUT_S",\n'
        '                   "PYTORCH_CUDA_ALLOC_CONF"):\n'
        '            _v = os.environ.get(_k)\n'
        '            if _v is not None:\n'
        '                _ray_env[_k] = _v\n'
        '        ray.init(runtime_env={"working_dir": "./DeepSWE_RL/rllm/verl",\n'
        '                              "env_vars": _ray_env})'
    )
    new_src, n = pattern.subn(new_call, src)
    if n == 0:
        print("ERROR: ray.init(runtime_env=...) block not found — script may be out of date.")
        return 2
    if n > 1:
        print(f"ERROR: matched {n} ray.init blocks; refusing to patch ambiguously.")
        return 3

    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"patched: {TARGET}")
    print("  - dropped 'pip' runtime_env clause (we reuse the parent venv)")
    print("  - dropped placeholder WANDB_API_KEY env var")
    return 0


if __name__ == "__main__":
    sys.exit(main())
