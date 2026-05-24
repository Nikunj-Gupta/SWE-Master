# SETUP_RL

Setup for the RL (rLLM/verl/DeepSWE) pipeline in this fork.

For the SFT pipeline see [SETUP.md](SETUP.md). For RL, the official rllm scripts
target 8–64 GPUs; on a 2-GPU box you can install + verify + smoke up to the
rollout phase, but actual training-at-scale needs more hardware.

---

## What works on 2 GPUs

| Step | Status |
|---|---|
| Install rllm/verl venv | ✅ |
| Data prep (R2E-Gym, SWE-Bench-Verified, etc. → parquet) | ✅ |
| Imports + Ray launch + worker spawn | ✅ |
| vLLM (V1 AsyncLLM) init + FSDP wrap + sleep mode | ✅ |
| Trainer reaches first rollout batch | ✅ |
| First rollout actually fires (vllm wake_up via async path) | ✅ |
| Docker rollout containers + sweagent + reward calc | ✅ |
| **End-to-end PPO step on 2× H100 80GB** | ✅ **47 s/step on 1.5B-Coder model with 8K/8K context, 65K ppo_max_token** |
| Real training (multi-step PPO that learns something) | ⚠️ needs 8–64 GPUs per the official deepswe_32b.sh; rewards are ~0 on this tiny model |

The smoke output we landed on (`step:1` metrics from `run_rl_dryrun.sh`):
- `traj/steps_mean: 3` (3 agent steps × 2 trajectories)
- `timing_s/step: 47.1` (rollout 40s + log_prob 0.9s + adv 0.9s + actor update 6s)
- `perf/max_memory_allocated_gb: 62` of 80
- `actor/grad_norm: 0` (because `batch/solve_none: 2` → all rewards 0 → no signal)
- All trajectory machinery (R2E-Gym docker containers, openhands scaffold,
  reward calc inside container) ran cleanly.

---

## Prerequisites

Same as SFT (see SETUP.md), plus:

- `git` (to clone the `verl` submodule from `agentica-project/verl`)
- A wandb account or `wandb login` (not strictly required if you pass `trainer.logger=['console']`)

---

## Step 1 — Source env.sh (every new shell)

```bash
source /path/to/SWE-Master/env.sh
```

This redirects every framework's scratch dir (HF, Triton, torch_extensions, pip,
uv, XDG, TMPDIR) under `$SWE_MASTER_CACHE_ROOT`. Default: `/data/$USER/cache` if
`/data` is writable, else `~/.cache/swe-master`. Override by exporting
`SWE_MASTER_CACHE_ROOT` before sourcing.

---

## Step 2 — Install the RL venv

```bash
bash install_rl_env.sh
```

Wall: ~30–60 min (flash-attn source build is the long pole). Logs to
`rl_smoke/logs/install_rl_<TS>.log` (symlink: `install_rl_latest.log`).

Pinning notes (deliberate, baked into the script):

- `torch>=2.7,<2.9` cu126 channel
- `vllm>=0.8.3,<0.11` — verl was developed against this range; vllm 0.13+
  removed `vllm.worker.worker_base`, `vllm.lora.models`, etc.
- `transformers<5`, `swebench==3.0.2`, `setuptools<81` — all required by
  one or another downstream import surface.

The script also runs `apply_patches_rl.sh` at the end (see Step 3).

---

## Step 3 — Apply RL patches

`install_rl_env.sh` runs this for you. To re-apply manually (e.g. after
re-pulling `DeepSWE_RL/rllm/`):

```bash
bash apply_patches_rl.sh           # apply (idempotent)
bash apply_patches_rl.sh --dry-run # just report status
```

Patches currently in registry (see `apply_patches_rl.py`):

| ID | What it does | Why |
|---|---|---|
| `rllm_ray_init` | Drops the `pip:` clause from `ray.init(runtime_env=...)` in `train_agent_ppo.py` | Ray's auto-virtualenv has no pip; trying to install ~20 packages fails immediately |

All patches are sentinel-tagged and idempotent. To revert any single one:

```bash
cd DeepSWE_RL/rllm && git checkout <target-file>
```

An earlier `patch_rllm_wake_up_dispatch.py` was tried and dropped — see
"Resolved: rollout wake_up" below for why.

---

## Step 4 — Prepare RL data

```bash
source DeepSWE_RL/.venv/bin/activate
cd DeepSWE_RL/rllm
python examples/swe/prepare_swe_data.py
```

Downloads 6 HF datasets (~3 GB) and writes parquets under
`DeepSWE_RL/rllm/rllm/data/datasets/<NAME>/{train,test}_verl.parquet`.
These are the only data artifacts the RL trainer needs.

---

## Step 5 — Verify

```bash
python verify_rl_setup.py
```

Probes the venv, imports, parquets, patches, docker. Non-zero exit on any red.

---

## Step 6 — Dry-run smoke

```bash
bash run_rl_dryrun.sh
```

Adapts `deepswe_32b.sh` to 2 GPUs: Qwen2.5-Coder-1.5B base, `ulysses=1`, full
FSDP+optimizer offload, batch size 2, agent.max_steps=3, `rollout.mode=async`.
Total wall ~10–20 min *if* it completes.

Overridable via env vars:

```bash
MODEL=Qwen/Qwen2.5-Coder-0.5B-Instruct \
PROMPT_LEN=1024 RESP_LEN=1024 \
bash run_rl_dryrun.sh
```

Log: `rl_smoke/logs/rl_dryrun_<TS>.log` (symlink: `rl_dryrun_latest.log`).

---

## "Where am I" snapshot

For any debugging session, this one command captures the relevant state:

```bash
bash rl_status.sh
```

Output: combined status to stdout + a log file under `rl_smoke/logs/` you can
share verbatim.

---

## Resolved: rollout wake_up (was AttributeError on RayWorkerGroup)

Earlier dry-run runs failed at the first rollout batch with:

```
AttributeError: 'ActorHandle' object has no attribute 'wake_up'
```

Root cause was **not** a missing decorator or dispatch glitch. rllm's
`AgentPPOTrainer.init_workers()` picks the rollout engine based on a config
key:

```python
if self.config.actor_rollout_ref.rollout.mode == "async":
    rollout_engine = self.async_rollout_manager     # has working wake_up()
else:
    rollout_engine = agent_rollout_wg               # bare RayWorkerGroup, no wake_up
```

Our minimised dry-run config was missing `actor_rollout_ref.rollout.mode=async`,
so rllm took the broken branch. The fix is two extra args in `run_rl_dryrun.sh`:

```bash
actor_rollout_ref.rollout.mode=async
actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler
```

(verl's own `deepswe_32b.sh` reference script has both; we'd dropped them when
minimising for 2 GPUs.) An earlier well-intentioned patch
(`patch_rllm_wake_up_dispatch.py`) tried to route the call through
`RayWorkerGroup.execute_all_sync(...)`. That patch is now wrong — with
`mode=async` it would *break* the working path. If you find it still applied:

```bash
cd DeepSWE_RL/rllm && git checkout rllm/engine/agent_execution_engine.py
```

---

## Cleanup

```bash
# kill any hung trainer / ray / vllm processes
pkill -9 -f "ray::WorkerDict|ray::DashboardAgent|ray::RuntimeEnvAgent|train_agent_ppo" 2>/dev/null
ray stop --force 2>/dev/null || true

# wipe the venv for a fresh install
rm -rf DeepSWE_RL/.venv
```

The HF cache, Triton cache, etc. live under `$SWE_MASTER_CACHE_ROOT` (default
`/data/$USER/cache`). Wiping that frees ~100+ GB once vllm + model weights
have been pulled.
