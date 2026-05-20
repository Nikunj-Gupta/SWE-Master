#!/usr/bin/env python3
"""
Sanity-check the SWE-Master fork setup.

Probes each of the 3 venvs (OpenRLHF, R2E-Gym, vLLM), checks the patches
are applied, verifies Docker daemon is reachable, and tests vLLM HTTP
endpoint if it's running. Prints a checklist; non-zero exit if any red
items found.

Usage:
    python verify_setup.py
"""
import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import urllib.request
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
    mark = f"{GREEN}✓{NC}" if ok else (f"{YELLOW}?{NC}" if ok is None else f"{RED}✗{NC}")
    print(f"  {mark} {label}{(': ' + DIM + detail + NC) if detail else ''}")
    return 0 if (ok or ok is None) else 1


def section(title: str) -> None:
    print(f"\n{BLUE}== {title}{NC}")


def venv_python(venv_dir: Path) -> Optional[Path]:
    p = venv_dir / "bin" / "python"
    return p if p.is_file() else None


def venv_pkg_version(venv_dir: Path, pkg: str) -> Optional[str]:
    py = venv_python(venv_dir)
    if not py:
        return None
    try:
        r = subprocess.run(
            [str(py), "-c", f"from importlib.metadata import version; print(version('{pkg}'))"],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def venv_can_import(venv_dir: Path, module: str) -> bool:
    py = venv_python(venv_dir)
    if not py:
        return False
    try:
        r = subprocess.run(
            [str(py), "-c", f"import {module}"],
            capture_output=True, text=True, timeout=30,
        )
        return r.returncode == 0
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="suppress detail strings")
    args = ap.parse_args()
    failures = 0

    # -----------------------------------------------------------------
    # System prerequisites
    # -----------------------------------------------------------------
    section("System prerequisites")

    nvcc = shutil.which("nvcc")
    nvcc_ver = None
    if nvcc:
        try:
            r = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=5)
            for line in r.stdout.splitlines():
                if "release" in line:
                    nvcc_ver = line.strip()
                    break
        except Exception:
            pass
    failures += check(
        nvcc is not None and nvcc_ver is not None and "12." in (nvcc_ver or ""),
        f"nvcc on PATH (CUDA 12.x)",
        f"{nvcc} — {nvcc_ver}" if nvcc else "not found",
    )
    if nvcc and "11." in (nvcc_ver or ""):
        print(f"      {YELLOW}↳ nvcc resolves to CUDA 11.x — see SETUP.md / Troubleshooting{NC}")

    # nvidia-smi
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_count = int(r.stdout.strip().splitlines()[0]) if r.returncode == 0 else 0
    except Exception:
        gpu_count = 0
    failures += check(gpu_count > 0, f"nvidia-smi reports {gpu_count} GPU(s)",
                      "" if args.quiet else f"need ≥1, recommended 8")

    # Docker
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        docker_ok = r.returncode == 0
        docker_root = next(
            (l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if "Docker Root Dir" in l),
            "?",
        )
    except Exception:
        docker_ok = False
        docker_root = "?"
    failures += check(docker_ok, "Docker daemon reachable", f"root={docker_root}" if not args.quiet else "")

    # uv
    failures += check(shutil.which("uv") is not None, "uv installed", shutil.which("uv") or "")

    # -----------------------------------------------------------------
    # Venvs
    # -----------------------------------------------------------------
    venvs = {
        "OpenRLHF SFT venv": REPO / "OpenRLHF_SFT/.venv",
        "R2E-Gym client venv": REPO / "R2E-Gym/.venv",
        "vLLM serve venv": REPO / "vllm_venv",
    }
    for name, vdir in venvs.items():
        section(name)
        py = venv_python(vdir)
        failures += check(py is not None, f"venv exists at {vdir.relative_to(REPO)}/")
        if not py:
            continue
        # Common probes
        probes_per_venv = {
            "OpenRLHF SFT venv": [
                ("torch", "torch", "2.x+cu126"),
                ("deepspeed", "deepspeed", "≥0.18"),
                ("flash_attn", "flash_attn", "≥2.8"),
                ("openrlhf", "openrlhf", ""),
                ("transformers", "transformers", "4.46–4.99"),
                ("liger-kernel", "liger_kernel", "≥0.8"),
            ],
            "R2E-Gym client venv": [
                ("r2e-gym", "r2egym", ""),
                ("litellm", "litellm", "≥1.80"),
                ("swebench", "swebench", "≥3.0"),
                ("huggingface_hub", "huggingface_hub", "<1.0"),
                ("transformers", "transformers", "<5"),
                ("datasets", "datasets", "2.19"),
            ],
            "vLLM serve venv": [
                ("vllm", "vllm", "≥0.11"),
                ("torch", "torch", "2.x+cu126"),
                ("transformers", "transformers", "<5"),
            ],
        }[name]
        for pkg, module, expect in probes_per_venv:
            ver = venv_pkg_version(vdir, pkg)
            importable = venv_can_import(vdir, module) if ver else False
            failures += check(
                ver is not None and importable,
                f"{pkg} {ver or '(not installed)'}",
                f"expect {expect}" if (expect and not args.quiet) else "",
            )

    # -----------------------------------------------------------------
    # Patches applied
    # -----------------------------------------------------------------
    section("Patches applied (subset check)")
    patch_probes = [
        (REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
         "spec_data = json.loads(self.ds['make_test_spec'])",
         "make_test_spec shadowing fix"),
        (REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
         'self.docker_host = "unix:///var/run/docker.sock"',
         "Unix-socket Docker support"),
        (REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
         'self.run("pip install chardet")',
         "PyPI mirror swap (3 sites)"),
        (REPO / "R2E-Gym/src/r2egym/agenthub/agent/agent.py",
         "lite_llm_max_token = 4096",
         "max_tokens lowered to 4096 (2 sites)"),
        (REPO / "OpenRLHF_SFT/OpenRLHF/openrlhf/models/utils.py",
         "inplace_backward=True",
         "CE inplace_backward (memory)"),
        (REPO / "OpenRLHF_SFT/OpenRLHF/openrlhf/models/actor.py",
         '# output["logits"] = output["logits"].to(torch.float32)',
         "fp32 upcast disabled (memory)"),
    ]
    for fp, needle, label in patch_probes:
        if fp.exists():
            applied = needle in fp.read_text()
            failures += check(applied, label)
        else:
            print(f"  {YELLOW}?{NC} {label} {DIM}(source file missing — install first){NC}")

    # -----------------------------------------------------------------
    # Local services
    # -----------------------------------------------------------------
    section("Local services (optional)")
    # vLLM endpoint
    try:
        with urllib.request.urlopen("http://localhost:8000/v1/models", timeout=3) as resp:
            data = json.loads(resp.read())
            mid = data["data"][0]["id"]
            mlen = data["data"][0].get("max_model_len", "?")
            check(True, f"vLLM serving '{mid}' at :8000", f"max_model_len={mlen}")
    except Exception:
        check(None, "vLLM endpoint at :8000 (skip if not started yet)")

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------
    print()
    if failures == 0:
        print(f"{GREEN}all green — pipeline should be runnable.{NC}")
    else:
        print(f"{RED}{failures} check(s) failed — see SETUP.md for remediation.{NC}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
