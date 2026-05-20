#!/usr/bin/env python3
"""Replicate the shipped 5-line demo trajectory file N times for the smoke run.

The OpenRLHF SFT loop needs more than 5 samples to step through several
optimizer updates. Copying the same lines is fine for a "does it train"
proof-of-life; we are not measuring generalization.
"""
import argparse
import pathlib


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        default="/data/nikunj/SWE-Master/data_examples/sft_data/openrlhf_sft_multi_turn_data_demo.jsonl",
    )
    ap.add_argument("--dst", default="/data/nikunj/SWE-Master/sft_smoke/data/demo_x20.jsonl")
    ap.add_argument("--copies", type=int, default=20)
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    dst = pathlib.Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    lines = src.read_text().splitlines(keepends=True)
    with dst.open("w") as f:
        for _ in range(args.copies):
            f.writelines(lines)
    print(f"wrote {dst} ({len(lines) * args.copies} samples)")


if __name__ == "__main__":
    main()
