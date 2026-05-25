#!/usr/bin/env python3
"""Sanity-check the rllm/verl RL training environment.

Mirrors verify_setup.py but for the RL pipeline. Probes the DeepSWE_RL
venv, the parquets we generate via prepare_swe_data.py, the legacy vllm
imports verl needs, and the patches we apply. Non-zero exit on any red.

Usage:
    python verify_rl_setup.py
"""
import importlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
DIM = "\033[2m"
NC = "\033[0m"


def check(ok: Optional[bool], label: str, detail: str = "") -> int:
    if ok is True:
        mark = f"{GREEN}✓{NC}"
    elif ok is None:
        mark = f"{YELLOW}?{NC}"
    else:
        mark = f"{RED}✗{NC}"
    suffix = f": {DIM}{detail}{NC}" if detail else ""
    print(f"  {mark} {label}{suffix}")
    return 0 if ok in (True, None) else 1


def section(title: str) -> None:
    print(f"\n{BLUE}== {title}{NC}")


def import_version(modname: str) -> tuple[Optional[bool], str]:
    """Returns (ok, version_or_error_string)."""
    try:
        m = importlib.import_module(modname)
        return True, getattr(m, "__version__", "?")
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def main() -> int:
    fails = 0

    section("System prerequisites")
    nvcc_path = shutil.which("nvcc")
    fails += check(nvcc_path is not None, "nvcc on PATH",
                   subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.splitlines()[-1]
                   if nvcc_path else "")
    # Hard-check the nvcc release matches our pinned cu126 torch.
    if nvcc_path:
        try:
            out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
            import re
            m = re.search(r"release (\d+\.\d+)", out)
            nvcc_rel = m.group(1) if m else "?"
            fails += check(nvcc_rel == "12.6",
                           f"nvcc release {nvcc_rel}",
                           "must be 12.6.x — torch is hard-pinned to cu126")
        except Exception as e:
            fails += check(False, "nvcc version parse", f"{type(e).__name__}: {e}")
    try:
        n_gpus = int(subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"]
        ).decode().strip().count("\n") + 1)
        fails += check(n_gpus >= 1, f"nvidia-smi reports {n_gpus} GPU(s)",
                       "need ≥1, official rllm scripts target 8/16+")
    except Exception as e:
        fails += check(False, "nvidia-smi callable", f"{type(e).__name__}: {e}")

    section("DeepSWE_RL venv")
    venv = REPO / "DeepSWE_RL" / ".venv"
    fails += check(venv.is_dir(), f"venv exists at {venv.relative_to(REPO)}/")
    if not venv.is_dir():
        print(f"\n{RED}venv missing — run `bash install_rl_env.sh` first.{NC}")
        return 1

    section("Core RL packages")
    # Force the diagnostic to use the venv's python so we report on the right env.
    venv_python = venv / "bin" / "python"
    if venv_python.exists() and venv_python.resolve() != Path(sys.executable).resolve():
        print(f"  {YELLOW}!{NC} verify is using {sys.executable}, but the RL venv python is {venv_python}.")
        print(f"  Re-running self under venv python for accurate results...")
        os.execv(str(venv_python), [str(venv_python), __file__])  # never returns

    for pkg, expect in [
        ("torch", "==2.8.0+cu126 (hard-pinned)"),
        ("vllm", "≥0.8.3,<0.11 (verl-compat)"),
        ("rllm", "editable from DeepSWE_RL/rllm"),
        ("verl", "editable from DeepSWE_RL/rllm/verl"),
        ("flash_attn", "≥2.8"),
        ("deepspeed", "≥0.18"),
        ("ray", "≥2.0"),
        ("transformers", "≥4.46,<5"),
        ("r2egym", "editable from R2E-Gym (--no-deps)"),
        ("swebench", "==3.0.2 (R2E-Gym needs get_eval_type)"),
        ("pkg_resources", "shipped by setuptools<81"),
    ]:
        ok, info = import_version(pkg)
        if pkg == "torch" and ok:
            import torch
            info = f"{torch.__version__}  cuda={torch.version.cuda}  gpus={torch.cuda.device_count()}"
            # Hard fail if torch's bundled cuda isn't 12.6 — running with
            # a cu128/cu13x wheel against an nvcc 12.6 host is the exact
            # failure mode that surfaced on the other server.
            fails += check(torch.version.cuda == "12.6",
                           f"torch.version.cuda == 12.6",
                           f"got {torch.version.cuda} — re-run install_rl_env.sh; cu126 pin was lost")
        fails += check(ok, f"{pkg} {info}", f"expect {expect}")

    section("Legacy vllm imports verl requires (broken on vllm ≥0.13)")
    for path, sym in [
        ("vllm.worker.worker_base", "WorkerWrapperBase"),
        ("vllm.lora.models", "LoRAModel"),
        ("vllm.inputs", "SingletonInputs"),
    ]:
        try:
            m = importlib.import_module(path)
            has = hasattr(m, sym)
            fails += check(has, f"{path}.{sym}")
        except Exception as e:
            fails += check(False, f"{path}.{sym}", f"{type(e).__name__}: {e}")

    section("Patches applied")
    for tag, target, required in [
        ("# patched-by: patch_rllm_ray_init.py",
         "DeepSWE_RL/rllm/rllm/trainer/verl/train_agent_ppo.py", True),
    ]:
        p = REPO / target
        if not p.exists():
            fails += check(False, target, "file missing")
            continue
        try:
            applied = tag in p.read_text()
        except OSError as e:
            fails += check(False, target, f"unreadable: {e}")
            continue
        patch_name = tag.split(": ")[1]
        fails += check(applied, f"{patch_name} → {target}",
                       "" if applied else "run `bash apply_patches_rl.sh`")

    # Also warn (not fail) if the old wake_up_dispatch patch is still present —
    # it shouldn't be, see SETUP_RL.md "Open question / resolved" section.
    stale_tag = "# patched-by: patch_rllm_wake_up_dispatch.py"
    stale_target = REPO / "DeepSWE_RL/rllm/rllm/engine/agent_execution_engine.py"
    if stale_target.exists() and stale_tag in stale_target.read_text():
        check(False, f"stale patch present: {stale_tag.split(': ')[1]}",
              "revert: `cd DeepSWE_RL/rllm && git checkout rllm/engine/agent_execution_engine.py`")

    section("Parquets from prepare_swe_data.py")
    parquet_root = REPO / "DeepSWE_RL/rllm/rllm/data/datasets"
    for name, kind in [
        ("R2E_Gym_Subset", "train_verl.parquet"),
        ("R2E_Gym_Lite", "train_verl.parquet"),
        ("R2E_Gym_V1", "train_verl.parquet"),
        ("SWE_Bench_Verified", "test_verl.parquet"),
        ("SWE_Bench_Lite", "test_verl.parquet"),
        ("SweSmith_RL_Dataset", "train_verl.parquet"),
    ]:
        f = parquet_root / name / kind
        fails += check(f.exists(), f"{name}/{kind}",
                       f"{f.stat().st_size // (1024*1024)} MB" if f.exists() else "missing — run prepare_swe_data.py")

    section("Docker daemon (needed for sweagent rollouts)")
    try:
        out = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        ok = out.returncode == 0
        detail = "" if ok else out.stderr.splitlines()[0] if out.stderr else "non-zero exit"
        fails += check(ok, "docker info OK", detail)
    except Exception as e:
        fails += check(False, "docker info", f"{type(e).__name__}: {e}")

    print()
    if fails:
        print(f"{RED}{fails} check(s) failed.{NC}")
        print(f"See {REPO}/SETUP_RL.md for what to run.")
    else:
        print(f"{GREEN}All checks passed.{NC}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
