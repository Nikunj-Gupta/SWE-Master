# SETUP

Step-by-step setup for this SWE-Master pipeline fork.

For background on **what** this is and **what's different from upstream**, see [README.md](README.md). This document is just the operational run order.

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Linux x86_64 | tested on Ubuntu 22.04 | other distros likely fine |
| NVIDIA GPUs | 8× 48 GB (Ada or H100/H200) | minimum: 1× 48 GB to serve a 7B teacher and run small SFT; recommended: 8 GPUs for parallel rollouts + SFT |
| NVIDIA driver | ≥ 550 | CUDA 12.x compatible |
| `nvcc` | **12.6** | tested combo; matched against torch's cu126 wheels |
| `gcc` | 11.x | for flash-attn / ring-flash-attn source builds |
| Docker | ≥ 20.10 | running; user has access (`docker ps` works) |
| `uv` | latest | install: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Disk | ≥ 500 GB free where Docker stores | for SWE-Bench-Verified Docker images (each ~1–2 GB) |
| Disk for HF cache | ≥ 300 GB | for vLLM teacher checkpoints; reuse system-wide via `HF_HOME` |

Confirm `nvcc` resolves to 12.6:

```bash
which nvcc                    # should be /usr/local/cuda/bin/nvcc or /usr/local/cuda-12.6/bin/nvcc
nvcc --version | tail -2      # release 12.6, V12.6.x
```

If you have an older `nvcc` (e.g. Ubuntu's `/usr/bin/nvcc` from `nvidia-cuda-toolkit` package), add to your `~/.bashrc`:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

Confirm Docker root is NOT on the same partition as `/` if `/` is small:

```bash
docker info | grep "Docker Root Dir"
df -h /
```

---

## Step 1 — Clone and install

```bash
cd /your/workspace/
git clone <this-fork-url> SWE-Master
cd SWE-Master

# Build three separate venvs:
bash install_sft_env.sh         # OpenRLHF + flash-attn + deepspeed (~15 min, source-builds flash-attn)
bash install_rollout_env.sh     # R2E-Gym agent client + litellm + swebench (~3 min)
bash install_vllm_env.sh        # vLLM + cu126 torch (~5 min)
```

Each script logs to `sft_smoke/logs/install_<role>_<timestamp>.log` for debugging.

---

## Step 2 — Apply patches

```bash
bash apply_patches.sh           # full mode (Ada / 48GB GPUs)
# OR
bash apply_patches.sh h200      # H200 mode (skips 4 memory band-aids)
```

Idempotent — safe to re-run after any reinstall. The script edits files inside:
- `OpenRLHF_SFT/OpenRLHF/openrlhf/` (memory + bug patches)
- `OpenRLHF_SFT/.venv/.../deepspeed/runtime/engine.py` (memory patch — in venv, wiped on reinstall)
- `R2E-Gym/src/r2egym/agenthub/` (bug + env + context patches)

See [README.md § patches](README.md#patches-this-fork-makes-to-upstream-code) for the full list and reasoning.

---

## Step 3 — Verify

```bash
python verify_setup.py
```

This probes each venv (imports + versions), checks CUDA matches, confirms Docker is reachable, and prints a checklist. Resolve any red items before running the pipeline.

---

## Step 4 — Download / prepare data

Get the SWE-Bench-Verified dataset (one-time, ~500 MB):

```bash
bash data_preparation/download_swe_datasets.sh
# Then point the run_rollout_smoke.sh DATASET path at where this lands.
```

By default our scripts assume `/data/nikunj/SWE-Master-backup/datasets/SWE-Bench-Verified/` — edit this path in `run_rollout_smoke.sh` to match your filesystem.

---

## Step 5 — Run the pipeline

### Option A: orchestrator (one command)

```bash
export WANDB_API_KEY=<your_wandb_key>
bash run_full_pipeline.sh
```

Runs: vLLM serve → 1-instance rollout smoke → convert → SFT smoke. Good as a first-time sanity test (~30 min wall).

### Option B: piece by piece

**Serve a teacher:**
```bash
export HF_HOME=/where/you/want/the/model/cache    # optional, defaults to /data/nikunj/hf_cache
TEACHER_MODEL=RUC-AIBOX/SWE-Master-32B-SFT \
TEACHER_NAME=swe-master-32b-sft \
GPUS=0,1,2,3,4,5,6,7 \
TP_SIZE=8 \
GPU_MEM=0.92 \
MAX_MODEL_LEN=131072 \
bash serve_vllm.sh
```

Check it's up: `curl http://localhost:8000/v1/models`.

**Run rollouts:**
```bash
TEACHER_NAME=swe-master-32b-sft \
MAX_STEPS=30 \
MAX_WORKERS=16 \
START_IDX=0 \
K=100 \
USE_FN_CALLING=False \
USED_YAML=./src/r2egym/agenthub/config/openhands/openhands_sp_non_fn_calling.yaml \
bash run_rollout_smoke.sh
```

Trajectories land in `rollout_smoke/results/smoke-<teacher>-<TS>.jsonl`.

**Convert + filter to SFT format:**
```bash
source R2E-Gym/.venv/bin/activate
python rollout_smoke/convert_rollouts_to_sft.py \
    --src rollout_smoke/results/<your-trajectory-file>.jsonl \
    --dst-prefix rollout_smoke/sft_data/<a_name>
# Output: <prefix>.openrlhf.filtered.jsonl (reward==1 only)
```

If you have multiple rollout files, run the convert script on each then `cat *.openrlhf.filtered.jsonl > all_reward1.jsonl`.

**Train SFT:**
```bash
export WANDB_API_KEY=<your_wandb_key>
# Edit the script to point --data.dataset at your aggregated reward==1 corpus.
bash OpenRLHF_SFT/scripts_swe_master/qwen_25_coder_3B_final_sft.sh
```

Output: `sft_final/hf/` (the trained model, ~6 GB).

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `CUDA version 11.5 does not match 12.6` during vLLM install or SFT | `/usr/bin/nvcc` from `nvidia-cuda-toolkit` package shadows the real one | Already handled by our scripts. If you hit it elsewhere, prepend `/usr/local/cuda/bin` to PATH. |
| `cannot find -lcuda` during vLLM startup | Missing `libcuda.so` stub | Our `serve_vllm.sh` sets `LIBRARY_PATH=$CUDA_HOME/lib64/stubs:...`. If you run vLLM manually, do the same. |
| `cannot import name 'HfFolder' from 'huggingface_hub'` | R2E-Gym needs `huggingface_hub<1.0` | Already pinned in `install_rollout_env.sh`. Run `bash install_rollout_env.sh` again. |
| `cannot import name 'is_offline_mode'` | Old huggingface_hub with new transformers 5.x | Pin transformers<5 — already in our installs. |
| `Qwen2Tokenizer has no attribute all_special_tokens_extended` | vLLM 0.11 vs transformers 5.x | Already pinned transformers<5 in `install_vllm_env.sh`. |
| `UnboundLocalError: make_test_spec` | docker.py shadowing bug | `bash apply_patches.sh` |
| `Path to a certificate and key files must be provided` (Docker) | TLS env vars set without certs | `bash apply_patches.sh` |
| `pip install chardet` hangs forever | Trying to reach `pypi-mirror.weizhipin.com` | `bash apply_patches.sh` |
| Rollout trajectories all reward==0 with `<function=>` empty actions | Model doesn't follow fn_calling scaffold | Set `USE_FN_CALLING=False` + use `openhands_sp_non_fn_calling.yaml` |
| `ContextWindowExceededError 'max_tokens' too large: 16384` | Agent asking for 16K output with 16K context | `bash apply_patches.sh` (lowers `lite_llm_max_token`) OR bump `MAX_MODEL_LEN` to 32K+ |
| SFT OOM on backward in cross-entropy | 18 GB dlogits buffer | `bash apply_patches.sh` |
| SFT OOM on `output["logits"].to(torch.float32)` | fp32 upcast doubles memory | `bash apply_patches.sh` |
| SFT OOM on `grad / gradient_accumulation_steps()` | 9 GB pointless copy | `bash apply_patches.sh` |

---

## Cleanup

After a session:

```bash
# Stop vLLM
kill $(cat vllm_venv/vllm.pid) 2>/dev/null

# Clean stale Docker containers from killed rollouts
docker ps -q --filter ancestor=slimshetty/swebench-verified | xargs -r docker stop
docker ps -aq --filter status=exited | xargs -r docker rm
```
