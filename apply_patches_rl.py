#!/usr/bin/env python3
"""Apply the RL-side patches to the rllm/verl tree.

This is the RL counterpart to apply_patches.py. Each patch is a small
self-contained Python script in `data_preparation/`; we just invoke them in
order. They're all idempotent (sentinel-tagged) so re-running this is safe.

Patches:
  1. patch_rllm_ray_init.py            — drop the pip clause from rllm's
                                          ray.init(runtime_env=...) call.
                                          Without this, Ray creates an
                                          isolated virtualenv per worker
                                          and tries to pip-install ~20
                                          packages with no pip in the venv.
  2. patch_rllm_wake_up_dispatch.py    — route rllm's direct
                                          self.rollout_engine.wake_up()
                                          calls through RayWorkerGroup's
                                          execute_all_sync(). Without this,
                                          AttributeError at first rollout.

Usage:
    python apply_patches_rl.py [--dry-run]

The --dry-run flag just prints which patches would run; the individual
patch scripts don't currently have their own dry-run mode, so we honour
it by inspecting their sentinel comments to report status.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent

PATCHES = [
    {
        "id": "rllm_ray_init",
        "script": "data_preparation/patch_rllm_ray_init.py",
        "target": "DeepSWE_RL/rllm/rllm/trainer/verl/train_agent_ppo.py",
        "sentinel": "# patched-by: patch_rllm_ray_init.py",
        "reason": "Drop ray.init pip runtime_env clause + placeholder WANDB_API_KEY",
    },
    # NOTE: patch_rllm_wake_up_dispatch.py was tried but is the wrong fix.
    # The actual cure is to set actor_rollout_ref.rollout.mode=async in the
    # trainer config, which makes rllm pick `self.async_rollout_manager`
    # (which has a working .wake_up()) instead of the bare RayWorkerGroup.
    # The dispatch patch would *break* the async path because
    # AsyncLLMServerManager has no `execute_all_sync` method. If you find
    # this patch still applied, revert it:
    #     cd DeepSWE_RL/rllm && git checkout rllm/engine/agent_execution_engine.py
]


def patch_status(p: dict) -> str:
    target = REPO / p["target"]
    if not target.exists():
        return "missing"
    try:
        return "applied" if p["sentinel"] in target.read_text() else "pending"
    except OSError:
        return "unreadable"


def run_patch(p: dict) -> int:
    script = REPO / p["script"]
    if not script.exists():
        print(f"  ERROR script missing: {script}")
        return 1
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        print(f"    {line}")
    for line in proc.stderr.splitlines():
        print(f"    [stderr] {line}")
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--dry-run", action="store_true",
                    help="Just report which patches are pending/applied; don't run.")
    args = ap.parse_args()

    n_applied = 0
    n_already = 0
    n_pending = 0
    n_error = 0

    for p in PATCHES:
        status = patch_status(p)
        print(f"== [{p['id']}] {p['reason']}")
        print(f"     target: {p['target']}")
        print(f"     status: {status}")
        if args.dry_run:
            n_pending += int(status == "pending")
            n_already += int(status == "applied")
            n_error += int(status in ("missing", "unreadable"))
            continue
        if status == "applied":
            print("     -> already applied, skipping")
            n_already += 1
            continue
        if status in ("missing", "unreadable"):
            print(f"     -> {status}: cannot apply")
            n_error += 1
            continue
        print("     -> applying")
        rc = run_patch(p)
        if rc == 0:
            n_applied += 1
        else:
            n_error += 1

    print()
    print(f"summary: {n_applied} applied, {n_already} already-applied, "
          f"{n_pending} pending (dry-run), {n_error} error")
    return 1 if n_error else 0


if __name__ == "__main__":
    sys.exit(main())
