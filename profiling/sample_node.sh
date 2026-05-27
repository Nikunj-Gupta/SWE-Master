#!/usr/bin/env bash
# Sample GPU utilization + a NIC's byte counters at a fixed interval.
#
# Dependency-free: only nvidia-smi + /proc/net/dev + /sys/class/net. No venv,
# no sar/ifstat. Run this on EACH node for the duration of a multi-node
# training run, then feed the CSVs to analyze_comms.py.
#
# Usage:
#   bash profiling/sample_node.sh [NIC] [INTERVAL_S] [OUT_CSV]
#     NIC         default: the default-route interface (eno1 on rio,
#                 enp2s0f0 on sutlej)
#     INTERVAL_S  default: 0.5
#     OUT_CSV     default: rl_smoke/logs/profile/sample_<host>_<ts>.csv
#
# Stop with Ctrl-C (or `kill`); the CSV is complete at every line, so a
# partial capture is still analyzable.
set -euo pipefail

NIC=${1:-$(ip route show default 2>/dev/null | awk '{print $5; exit}')}
INTERVAL=${2:-0.5}
[ -n "$NIC" ] || { echo "FATAL: could not auto-detect NIC; pass it explicitly" >&2; exit 1; }
[ -e "/sys/class/net/$NIC" ] || { echo "FATAL: NIC '$NIC' not found" >&2; exit 1; }

HOST=$(hostname -s)
TS=$(date +%Y%m%d_%H%M%S)
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTDIR=$REPO/rl_smoke/logs/profile
mkdir -p "$OUTDIR"
OUT=${3:-$OUTDIR/sample_${HOST}_${TS}.csv}

# Link speed (Mbit/s) recorded as a comment so the analyzer can compute
# saturation % without guessing.
SPEED_MBIT=$(cat "/sys/class/net/$NIC/speed" 2>/dev/null || echo 0)

{
    echo "# host=$HOST nic=$NIC interval_s=$INTERVAL link_speed_mbit=$SPEED_MBIT"
    echo "epoch_s,gpu_util_mean,gpu_mem_used_mb_sum,gpu_power_w_sum,net_rx_bytes,net_tx_bytes"
} > "$OUT"

echo "==> $HOST: sampling GPU + $NIC (${SPEED_MBIT} Mbit/s) every ${INTERVAL}s"
echo "==> writing $OUT  — Ctrl-C to stop"

while true; do
    now=$(date +%s.%N)
    # mean GPU util across all GPUs; summed mem (MB) and power (W)
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw \
            --format=csv,noheader,nounits 2>/dev/null \
          | awk -F', *' '{u+=$1; m+=$2; p+=$3; n++}
                         END{if(n)printf "%.1f,%.0f,%.1f",u/n,m,p; else printf "0,0,0"}')
    # cumulative NIC byte counters (rx=field1, tx=field9 after the colon)
    net=$(grep -w "$NIC" /proc/net/dev | sed 's/.*://' | awk '{print $1","$9}')
    echo "${now},${gpu},${net}" >> "$OUT"
    sleep "$INTERVAL"
done
