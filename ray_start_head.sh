#!/usr/bin/env bash
# Start the Ray head node for multi-node RL training.
#
# Run this on the *head* machine first (e.g. rio); then on each worker
# (e.g. sutlej) run ray_start_work.sh with HEAD_IP set to this node's IP.
#
# After both `ray start` commands complete, kick off the trainer from the
# head node with:
#     NNODES=<total> N_GPUS=<per-node> bash run_rl_dryrun.sh
# Python's ray.init() picks up the running cluster automatically (no extra
# config needed — Ray writes a session file at /tmp/ray that ray.init() reads).
#
# Knobs (env overridable):
#   HEAD_IP         — IP this node binds to.  REQUIRED (no sensible default
#                     on multi-NIC boxes; `hostname -I` may pick a docker
#                     bridge IP that workers can't reach).
#   NUM_GPUS        — GPUs to expose to Ray on this node
#                     (default: count from `nvidia-smi -L`).
#   PORT            — GCS port workers will connect to  (default: 8266)
#   DASHBOARD_PORT  — Ray dashboard HTTP port           (default: 8267)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
[ -f "$REPO/env.sh" ] && source "$REPO/env.sh"
# Need `ray` on PATH — comes from the RL venv.
[ -f "$REPO/DeepSWE_RL/.venv/bin/activate" ] && source "$REPO/DeepSWE_RL/.venv/bin/activate"

if [ -z "${HEAD_IP:-}" ]; then
    echo "FATAL: HEAD_IP not set." >&2
    echo "       This is the IP workers will dial into. On a multi-NIC box," >&2
    echo "       hostname -I gives multiple candidates — pick the one workers" >&2
    echo "       can actually reach (often the 10.x or 192.168.x address)." >&2
    echo "       Candidates here:" >&2
    hostname -I | tr ' ' '\n' | sed 's/^/         /' >&2
    exit 1
fi

NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | grep -c "^GPU " || true)}
[ "${NUM_GPUS:-0}" -ge 1 ] || { echo "FATAL: NUM_GPUS=${NUM_GPUS:-0} (need ≥1)" >&2; exit 1; }
PORT=${PORT:-8266}
DASHBOARD_PORT=${DASHBOARD_PORT:-8267}

echo "==> Starting Ray head"
echo "       HEAD_IP        = $HEAD_IP"
echo "       NUM_GPUS       = $NUM_GPUS"
echo "       PORT           = $PORT"
echo "       DASHBOARD_PORT = $DASHBOARD_PORT"
echo

ray start --head \
    --node-ip-address "$HEAD_IP" \
    --num-gpus "$NUM_GPUS" \
    --port "$PORT" \
    --dashboard-port "$DASHBOARD_PORT"

echo
echo "==> On each worker node, run:"
echo "       HEAD_IP=$HEAD_IP HEAD_PORT=$PORT NUM_GPUS=<gpus> bash ray_start_work.sh"
echo
echo "==> When all workers have joined, on THIS node run:"
echo "       NNODES=<total> N_GPUS=$NUM_GPUS bash run_rl_dryrun.sh"
echo
echo "==> To stop: ray stop --force"
