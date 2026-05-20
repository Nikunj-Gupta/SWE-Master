#!/usr/bin/env python3
"""
Close the R2E → SFT data loop for our rollout sweep.

Reads a single jsonl file containing R2E-Gym rollout trajectories
(potentially many per file), converts each to OpenRLHF multi-turn chat
format, then applies the reward==1 filter.

Writes two files:
  <prefix>.openrlhf.jsonl          all converted (1 line per trajectory)
  <prefix>.openrlhf.filtered.jsonl reward==1 only — SFT-ready
"""
import argparse
import json
import sys
from pathlib import Path

# Re-use the FC_SP constant + conversion logic from the repo script
sys.path.insert(0, "/data/nikunj/SWE-Master/OpenRLHF_SFT/SFT_data_pre_process/r2e_to_openrlhf_format")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "convert_r2e",
    "/data/nikunj/SWE-Master/OpenRLHF_SFT/SFT_data_pre_process/r2e_to_openrlhf_format/0_covert_r2e_format_to_sft_foramt.py",
)
convert_r2e = importlib.util.module_from_spec(spec)
# The original script runs main code on import (under `if __name__ == "__main__"` so it's safe).
spec.loader.exec_module(convert_r2e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="R2E trajectory jsonl (multiple trajectories OK)")
    ap.add_argument("--dst-prefix", required=True, help="output prefix (no extension)")
    args = ap.parse_args()

    src = Path(args.src)
    prefix = Path(args.dst_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: convert all trajectories
    messages_all = convert_r2e.process_single_jsonl(str(src))
    out_all = f"{prefix}.openrlhf.jsonl"
    convert_r2e.save_messages_to_jsonl(messages_all, out_all)
    print(f"\nwrote {out_all} ({len(messages_all)} trajectories total)")

    # Step 2: apply the reward==1 filter (the rule in 1_init_format_filter.py:14)
    out_filtered = f"{prefix}.openrlhf.filtered.jsonl"
    kept = 0
    with open(out_all) as fin, open(out_filtered, "w") as fout:
        for line in fin:
            sample = json.loads(line)
            # The reward field is saved as part of save_messages_to_jsonl; verify field
            if sample.get("reward") in (1, 1.0, "1.0", "1"):
                fout.write(line)
                kept += 1
    print(f"wrote {out_filtered} ({kept} trajectories with reward==1)")


if __name__ == "__main__":
    main()
