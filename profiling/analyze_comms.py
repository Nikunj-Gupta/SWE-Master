#!/usr/bin/env python3
"""Analyze cross-node comms cost from sample_node.sh CSVs + a dryrun log.

Stdlib only — runs under any python3, no venv needed.

What it computes, per node:
  - capture duration, sample count
  - network throughput (MB/s) from the diff of cumulative byte counters,
    reported as peak / p95 / mean for rx and tx, plus % of link line-rate
  - GPU utilization distribution (mean / p50 / fraction of time < 20%, the
    "GPU idle, probably waiting on comms" bucket)
  - correlation: mean GPU util during the busiest-network decile vs overall.
    If GPU util is much lower while the network is saturated, the step is
    comms-bound — that's the cross-node tax made visible.

And from the trainer log (optional): the timing_s/* phase durations, with the
update_actor time called out against the single-node FSDP baseline so you can
attribute the delta to gradient all-reduce.

Usage:
  python profiling/analyze_comms.py rl_smoke/logs/profile/sample_*.csv \
      [--log rl_smoke/logs/rl_dryrun_latest.log] \
      [--baseline-update-actor 5.7]
"""
import argparse
import glob
import re
import statistics as st
import sys
from pathlib import Path

LINE = "-" * 64


def pct(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def load_csv(path):
    """Returns (meta dict, list of sample dicts)."""
    meta = {}
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("#"):
                for kv in ln[1:].split():
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        meta[k] = v
                continue
            if ln.startswith("epoch_s") or not ln.strip():
                continue
            parts = ln.split(",")
            if len(parts) < 6:
                continue
            try:
                rows.append({
                    "t": float(parts[0]),
                    "gpu_util": float(parts[1]),
                    "gpu_mem_mb": float(parts[2]),
                    "gpu_pow_w": float(parts[3]),
                    "rx": int(parts[4]),
                    "tx": int(parts[5]),
                })
            except ValueError:
                continue
    return meta, rows


def throughput_series(rows):
    """MB/s between consecutive samples; aligned to the *later* sample's time."""
    out = []
    for a, b in zip(rows, rows[1:]):
        dt = b["t"] - a["t"]
        if dt <= 0:
            continue
        rx = (b["rx"] - a["rx"]) / dt / 1e6   # MB/s
        tx = (b["tx"] - a["tx"]) / dt / 1e6
        out.append({"t": b["t"], "rx": max(0, rx), "tx": max(0, tx),
                    "gpu_util": b["gpu_util"]})
    return out


def analyze_node(path):
    meta, rows = load_csv(path)
    host = meta.get("host", Path(path).stem)
    nic = meta.get("nic", "?")
    speed_mbit = float(meta.get("link_speed_mbit", 0) or 0)
    line_mbps = speed_mbit / 8.0  # Mbit/s -> MB/s line rate
    if len(rows) < 3:
        print(f"  {host}: only {len(rows)} samples — too short to analyze")
        return
    series = throughput_series(rows)
    dur = rows[-1]["t"] - rows[0]["t"]
    rx = [s["rx"] for s in series]
    tx = [s["tx"] for s in series]
    util = [s["gpu_util"] for s in series]

    # busiest-network decile: where (rx+tx) is in the top 10%
    busy_thresh = pct([s["rx"] + s["tx"] for s in series], 90)
    busy = [s for s in series if (s["rx"] + s["tx"]) >= busy_thresh and busy_thresh > 0]
    util_busy = st.mean([s["gpu_util"] for s in busy]) if busy else 0.0

    def sat(v):
        return f"{100*v/line_mbps:5.1f}% of line" if line_mbps else "n/a"

    print(f"\n  {host}  (nic={nic}, link={speed_mbit:.0f} Mbit/s ≈ {line_mbps:.0f} MB/s)")
    print(f"    duration {dur:6.1f}s   samples {len(rows)}")
    print(f"    net rx   peak {max(rx):7.1f}  p95 {pct(rx,95):7.1f}  mean {st.mean(rx):7.1f} MB/s"
          f"   (peak {sat(max(rx))})")
    print(f"    net tx   peak {max(tx):7.1f}  p95 {pct(tx,95):7.1f}  mean {st.mean(tx):7.1f} MB/s"
          f"   (peak {sat(max(tx))})")
    print(f"    gpu util mean {st.mean(util):5.1f}%   p50 {pct(util,50):5.1f}%"
          f"   idle(<20%) {100*sum(1 for u in util if u<20)/len(util):4.1f}% of time")
    print(f"    >> during busiest-network 10% of samples: GPU util {util_busy:5.1f}%"
          f"  (overall {st.mean(util):5.1f}%)")
    if busy and util_busy < st.mean(util) * 0.6:
        print(f"    >> COMMS-BOUND signal: GPU drops to {util_busy:.0f}% while network saturates")


PHASES = ["collect_trajectory", "old_log_prob", "adv", "update_actor",
          "transform_trajectory", "step"]


def analyze_log(path, baseline_ua):
    text = Path(path).read_text(errors="ignore")
    # last step's metrics line
    m = {}
    for ph in PHASES:
        hits = re.findall(rf"timing_s/{ph}:([0-9.]+)", text)
        if hits:
            m[ph] = float(hits[-1])
    if not m:
        print("  (no timing_s/* metrics found in log)")
        return
    print(f"\n  per-phase wall (last step) from {Path(path).name}:")
    total = m.get("step", sum(v for k, v in m.items() if k != "step"))
    for ph in PHASES:
        if ph in m and ph != "step":
            share = 100 * m[ph] / total if total else 0
            print(f"    {ph:22s} {m[ph]:7.2f}s  ({share:4.1f}%)")
    print(f"    {'step (total)':22s} {m.get('step', total):7.2f}s")
    if "update_actor" in m and baseline_ua > 0:
        delta = m["update_actor"] - baseline_ua
        print(f"\n    update_actor {m['update_actor']:.2f}s vs single-node baseline "
              f"{baseline_ua:.2f}s")
        if delta > 0:
            print(f"    >> +{delta:.2f}s attributable to cross-node gradient all-reduce "
                  f"({100*delta/m.get('step',1):.0f}% of step wall)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("csvs", nargs="+", help="sample_node.sh CSV files (one per node)")
    ap.add_argument("--log", help="dryrun log to pull timing_s/* phase durations from")
    ap.add_argument("--baseline-update-actor", type=float, default=5.7,
                    help="single-node FSDP update_actor seconds for delta attribution "
                         "(default 5.7, our 4-GPU rio baseline)")
    args = ap.parse_args()

    paths = []
    for c in args.csvs:
        paths.extend(sorted(glob.glob(c)))
    if not paths:
        print("no CSV files matched", file=sys.stderr)
        return 1

    print(LINE)
    print("CROSS-NODE COMMS PROFILE")
    print(LINE)
    print(f"nodes: {len(paths)} CSV(s)")
    for p in paths:
        analyze_node(p)

    if args.log and Path(args.log).exists():
        print("\n" + LINE)
        print("PHASE TIMING (trainer log)")
        print(LINE)
        analyze_log(args.log, args.baseline_update_actor)

    print("\n" + LINE)
    print("Reading the result: if a node's GPU util collapses during its")
    print("busiest-network window AND tx/rx peak near line rate, update_actor")
    print("is bottlenecked on cross-node all-reduce, not compute. On sub-10GbE")
    print("ethernet that's expected; the fix is faster interconnect, gradient")
    print("compression, or fewer/larger sync steps — not GPU/kernel tuning.")
    print(LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
