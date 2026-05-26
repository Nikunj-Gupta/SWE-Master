#!/usr/bin/env bash
# Smoke-test the rLLM (DeepSWE) RL training pipeline on minimal scale.
#
# Verifies that ray + verl + vllm + FSDP actor + sweagent rollout machinery
# can all launch and complete one PPO step on a tiny model with tiny batches.
#
# This is NOT a real training run — it's the "does the wiring work" smoke,
# adapted from rllm/scripts/agent/swe/deepswe_32b.sh for 2 GPUs.
#
# Knobs (env overridable):
#   MODEL           — base model HF id  (default: Qwen/Qwen2.5-Coder-1.5B-Instruct)
#   N_GPUS          — GPUs to use       (default: all visible to nvidia-smi, else 2)
#   PROMPT_LEN      — max prompt tokens (default: 2048)
#   RESP_LEN        — max response tok  (default: 2048)
#   BATCH_SIZE      — train batch       (default: N_GPUS, must be ≥ N_GPUS)
#   AGENT_MAX_STEPS — agent steps/traj  (default: 3)
set -euo pipefail

# Auto-discover repo root: this script lives at <repo>/run_rl_dryrun.sh
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RLLM_PKG_DIR=$REPO/DeepSWE_RL/rllm
RLLM_VENV=$REPO/DeepSWE_RL/.venv

# Cache redirects + PATH come from env.sh (portable across machines).
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"

# Logging — under rl_smoke/ to keep separate from sft_smoke/.
LOG_DIR=$REPO/rl_smoke/logs
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
LOG=$LOG_DIR/rl_dryrun_${TS}.log
ln -sfn "$(basename "$LOG")" "$LOG_DIR/rl_dryrun_latest.log"

# Knobs
MODEL=${MODEL:-Qwen/Qwen2.5-Coder-1.5B-Instruct}
# Default to every GPU `nvidia-smi -L` reports. Hard fallback to 2 if
# nvidia-smi isn't on PATH or returns nothing (e.g., CPU-only host doing
# a parse-only smoke). Override via `N_GPUS=N bash run_rl_dryrun.sh`.
_DETECTED_GPUS=$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)
N_GPUS=${N_GPUS:-${_DETECTED_GPUS:-2}}
[ "${N_GPUS:-0}" -ge 1 ] || N_GPUS=2
# Context budgets sized to match Qwen2.5-Coder-1.5B's 32K window with room
# for multi-turn agent trajectories. The previous 2K/2K combo was breaking
# at compute_log_prob (real trajectories had max_seq_len=6619).
PROMPT_LEN=${PROMPT_LEN:-8192}
RESP_LEN=${RESP_LEN:-8192}
# ppo_max_token_len_per_gpu must be >= longest packed micro-batch sequence.
# We use 4× (PROMPT_LEN+RESP_LEN) so even the longest unrolled trajectory
# from agent.max_steps turns has headroom.
PPO_MAX_TOKEN_LEN=${PPO_MAX_TOKEN_LEN:-$((4 * (PROMPT_LEN + RESP_LEN)))}
# BATCH_SIZE feeds both data.train_batch_size and actor.ppo_mini_batch_size.
# verl normalizes ppo_mini_batch_size by world_size/(tp*ulysses); with
# integer division, BATCH_SIZE < N_GPUS collapses it to 0 and aborts with
# `AssertionError: ppo_mini_batch_size 0 should be larger than 0 after
# normalization`. Default to N_GPUS so the smoke scales correctly with
# whatever hardware nvidia-smi sees.
BATCH_SIZE=${BATCH_SIZE:-$N_GPUS}
if [ "$BATCH_SIZE" -lt "$N_GPUS" ]; then
    echo "FATAL: BATCH_SIZE=$BATCH_SIZE < N_GPUS=$N_GPUS; verl would normalize" >&2
    echo "       ppo_mini_batch_size to 0. Set BATCH_SIZE >= $N_GPUS." >&2
    exit 1
fi
AGENT_MAX_STEPS=${AGENT_MAX_STEPS:-3}
# 1 = "is the plumbing correct" smoke; 10+ = real loss curve.
TOTAL_STEPS=${TOTAL_STEPS:-1}
# 1 = also send metrics to wandb (uses WANDB_API_KEY from ~/.netrc).
ENABLE_WANDB=${ENABLE_WANDB:-0}
EXP_NAME=${EXP_NAME:-dryrun-${TS}}

# Resolve parquet paths via the installed rllm package (independent of cwd).
# train_agent_ppo.py expects to run from $REPO (it has relative working_dir
# "./DeepSWE_RL/rllm/verl" and PYTHONPATH "./DeepSWE_RL/rllm" hardcoded).
TRAIN_FILE=$("$RLLM_VENV/bin/python" -c "import os,rllm; print(os.path.join(os.path.dirname(rllm.__file__), 'data', 'datasets', 'R2E_Gym_Subset', 'train_verl.parquet'))")
VAL_FILE=$("$RLLM_VENV/bin/python" -c "import os,rllm; print(os.path.join(os.path.dirname(rllm.__file__), 'data', 'datasets', 'SWE_Bench_Verified', 'test_verl.parquet'))")

# Build the logger spec for the trainer. Default is console-only. If wandb
# is requested, also pull WANDB_API_KEY from ~/.netrc so worker processes
# can authenticate (the patched ray.init forwards env vars to workers).
LOGGER_SPEC="['console']"
if [ "$ENABLE_WANDB" = "1" ]; then
    if [ -z "${WANDB_API_KEY:-}" ] && [ -f ~/.netrc ]; then
        WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1} f && /password/{print $2; exit}' ~/.netrc)
    fi
    if [ -z "$WANDB_API_KEY" ]; then
        echo "==> ENABLE_WANDB=1 but no WANDB_API_KEY found in env or ~/.netrc"
        echo "==> falling back to console-only logging"
    else
        export WANDB_API_KEY
        LOGGER_SPEC="['console','wandb']"
        echo "==> wandb logging enabled (key from $( [ -n "${WANDB_API_KEY:-}" ] && echo netrc/env || echo none ))"
    fi
fi

# Inner script that sg docker can launch via bash (sg invokes /bin/sh which
# lacks 'source'). Writing to a temp file is cleaner than nested quoting.
INNER=$(mktemp -t rl_dryrun_inner_XXXXXX.sh)
trap 'rm -f "$INNER"' EXIT
cat > "$INNER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source "$REPO/env.sh"
source "$RLLM_VENV/bin/activate"
cd "$REPO"

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
# VLLM_USE_V1=1: required by rllm's async rollout path.
# With rollout.mode=async (set below), rllm picks self.async_rollout_manager
# (an AsyncLLMServerManager), which creates AsyncvLLMServer instances that
# use vllm's V1 AsyncLLM. V1 AsyncLLM asserts VLLM_USE_V1 == 1; with =0 it
# raises "Using V1 AsyncLLMEngine, but envs.VLLM_USE_V1=False".
# (We tried =0 earlier when targeting the legacy sync path, but that path's
# wake_up dispatch is broken; the async path is the one that actually works.)
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=1
$([ -n "${WANDB_API_KEY:-}" ] && echo "export WANDB_API_KEY='$WANDB_API_KEY'")

python -m rllm.trainer.verl.train_agent_ppo \\
    algorithm.adv_estimator=loop \\
    data.train_files=$TRAIN_FILE \\
    data.val_files=$VAL_FILE \\
    data.train_batch_size=$BATCH_SIZE \\
    data.val_batch_size=$BATCH_SIZE \\
    data.max_prompt_length=$PROMPT_LEN \\
    data.max_response_length=$RESP_LEN \\
    data.filter_overlong_prompts=True \\
    actor_rollout_ref.model.path=$MODEL \\
    actor_rollout_ref.hybrid_engine=True \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=$BATCH_SIZE \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$PPO_MAX_TOKEN_LEN \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.name=vllm \\
    actor_rollout_ref.rollout.mode=async \\
    actor_rollout_ref.rollout.chat_scheduler=verl.schedulers.completions_scheduler.CompletionsScheduler \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \\
    actor_rollout_ref.rollout.n=1 \\
    actor_rollout_ref.rollout.val_kwargs.n=1 \\
    actor_rollout_ref.rollout.val_kwargs.temperature=0 \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \\
    trainer.critic_warmup=0 \\
    trainer.logger=$LOGGER_SPEC \\
    trainer.project_name='rl-dryrun' \\
    trainer.experiment_name='$EXP_NAME' \\
    trainer.val_before_train=False \\
    trainer.n_gpus_per_node=$N_GPUS \\
    trainer.nnodes=1 \\
    trainer.save_freq=-1 \\
    trainer.test_freq=-1 \\
    trainer.total_epochs=1 \\
    trainer.total_training_steps=$TOTAL_STEPS \\
    env.name=swe_arsenal \\
    +env.env_args.backend=docker \\
    +env.env_args.delete_image=False \\
    +env.env_args.scaffold=openhands \\
    agent.name=sweagent \\
    +agent.agent_args.scaffold=openhands \\
    agent.max_steps=$AGENT_MAX_STEPS \\
    agent.overlong_filter=True \\
    agent.async_engine=True
EOF
chmod +x "$INNER"

echo "==> repo:           $REPO"
echo "==> rllm venv:      $RLLM_VENV"
echo "==> train parquet:  $TRAIN_FILE"
echo "==> val parquet:    $VAL_FILE"
echo "==> model:          $MODEL"
echo "==> n_gpus:         $N_GPUS"
echo "==> prompt_len:     $PROMPT_LEN"
echo "==> resp_len:       $RESP_LEN"
echo "==> ppo_max_tok_pg: $PPO_MAX_TOKEN_LEN"
echo "==> batch_size:     $BATCH_SIZE"
echo "==> agent.max_steps:$AGENT_MAX_STEPS"
echo "==> total_steps:    $TOTAL_STEPS"
echo "==> logger:         $LOGGER_SPEC"
echo "==> exp_name:       $EXP_NAME"
echo "==> inner script:   $INNER"
echo "==> log:            $LOG"
echo "==> started at $(date -Iseconds)"

# sg docker because env=swe spawns rollout containers
sg docker -c "bash $INNER" 2>&1 | tee "$LOG"

echo
echo "==> finished at $(date -Iseconds)"
echo "==> log: $LOG"
