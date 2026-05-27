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

ray start --address="$HEAD_IP:$HEAD_PORT" --num-gpus="$NUM_GPUS"

echo
echo "==> Joined. Verify on the head node with: ray status"
echo "==> To leave the cluster: ray stop --force"
