# Archived SFT scripts

These scripts represent stepping-stones from the development process. They're kept for reference but are **not** part of the recommended workflow.

| Script | Why archived |
|---|---|
| `qwen_25_coder_7B_smoke.sh` | First SFT attempt on Qwen2.5-Coder-7B. OOMed on step 2 at 32K context on our 48GB cards. The path forward (3B + the memory patches + final 56-traj corpus) is captured in `../qwen_25_coder_3B_final_sft.sh`. Keep this script around as documentation of the 7B memory ceiling on Ada-class hardware. |
| `qwen_25_coder_3B_real_data_29_4gpu.sh` | 4-GPU variant (using GPUs 4-7 while vLLM serves on 0-3). OOMed because ZeRO-3 sharding factor halved (each GPU holds 1/4 instead of 1/8 of the model state). Demonstrates the limit of parallel SFT+rollout on this hardware. |

Both scripts are runnable if you want to reproduce the failure modes — but the *successful* working path is in the parent directory.

## Live scripts (parent directory)

| Script | Purpose |
|---|---|
| `qwen_25_coder_3B_smoke.sh` | First-time SFT smoke on shipped demo data (validates the trainer works) |
| `qwen_25_coder_3B_real_data.sh` | SFT on a small 6-trajectory real-rollout corpus (validates the full data→training loop end-to-end) |
| `qwen_25_coder_3B_final_sft.sh` | SFT on the combined 56-trajectory corpus from Phase 1 + Phase 2 teacher rollouts (the canonical "trained on real data" run) |
| `qwen_25_coder_32B_new_remove_01_not_dedep.sh` | Upstream's published 32B SFT recipe (reference; needs 16+ H200-class GPUs to actually run) |
