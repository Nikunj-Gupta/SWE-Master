#!/usr/bin/env bash
# Join a running Ray cluster as a worker node.
#
# Run ray_start_head.sh on the head machine first; that command prints
# the exact `HEAD_IP=... HEAD_PORT=... bash ray_start_work.sh` invocation
# to use here.
#
# Knobs (env overridable):
#   HEAD_IP    — IP of the head node.        REQUIRED.
#   HEAD_PORT  — GCS port on the head node.  (default: 8266)
#   NUM_GPUS   — GPUs this node contributes  (default: count from nvidia-smi).
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"
[ -f "$REPO/DeepSWE_RL/.venv/bin/activate" ] && source "$REPO/DeepSWE_RL/.venv/bin/activate"

if [ -z "${HEAD_IP:-}" ]; then
    echo "FATAL: HEAD_IP not set." >&2
    echo "       Get it from ray_start_head.sh's banner on the head machine." >&2
    exit 1
fi
HEAD_PORT=${HEAD_PORT:-8266}
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)}
[ "${NUM_GPUS:-0}" -ge 1 ] || { echo "FATAL: NUM_GPUS=${NUM_GPUS:-0} (need ≥1)" >&2; exit 1; }

echo "==> Joining Ray cluster"
echo "       HEAD       = $HEAD_IP:$HEAD_PORT"
echo "       NUM_GPUS   = $NUM_GPUS"
echo

# Raise the open-files soft limit before starting the raylet. Ray + vLLM
# across nodes open thousands of sockets; the default soft limit of 1024
# makes the raylet SIGABRT during vLLM engine-core launch with
# "open: Too many open files". The raylet inherits this shell's limit, so
# we must set it here (a non-root user can raise soft up to the hard cap).
_hard_nofile=$(ulimit -Hn)
if [ "$_hard_nofile" = "unlimited" ]; then
    ulimit -n 1048576 2>/dev/null || true
elif [ "${_hard_nofile:-0}" -gt "$(ulimit -Sn)" ] 2>/dev/null; then
    ulimit -n "$_hard_nofile" 2>/dev/null || true
fi
echo "==> open-files limit: soft=$(ulimit -Sn) hard=$(ulimit -Hn)"
if [ "$(ulimit -Sn)" != "unlimited" ] && [ "$(ulimit -Sn)" -lt 65536 ]; then
    echo "    WARNING: soft limit < 65536 and can't be raised (hard cap too low)." >&2
    echo "    The raylet may SIGABRT under multi-node vLLM load. To fix, a root" >&2
    echo "    user must raise the hard limit, e.g. in /etc/security/limits.conf:" >&2
    echo "        $USER  hard  nofile  1048576" >&2
fi

# Match the head's health-check tolerance — see ray_start_head.sh for
# the rationale. Without this, the runtime_env zip unpack from the first
# trainer launch starves this raylet's heartbeat loop and GCS kills it
# at ~5 s with "GCS failed to check the health of this node for 5 times".
export RAY_health_check_failure_threshold=${RAY_health_check_failure_threshold:-60}
export RAY_health_check_initial_delay_ms=${RAY_health_check_initial_delay_ms:-30000}

ray start --address="$HEAD_IP:$HEAD_PORT" --num-gpus="$NUM_GPUS"

echo
echo "==> Joined. Verify on the head node with: ray status"
echo "==> To leave the cluster: ray stop --force"
