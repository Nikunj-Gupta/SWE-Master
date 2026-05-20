<h1 align="center">SWE-Master — Reproducible Fork</h1>

<p align="center">
A fork of <a href="https://github.com/RUCAIBox/SWE-Master">RUCAIBox/SWE-Master</a> with
end-to-end-reproducible install scripts, upstream-bug patches, and validated
8×48GB / 8×H200 pipelines for both teacher rollouts and SFT training.
</p>

> **Looking for the original paper / pretrained models?** See [README_UPSTREAM.md](README_UPSTREAM.md) for the full upstream documentation. Citation, model collection on HuggingFace, paper, and pre-trained checkpoints are all preserved there.

---

## What this fork adds over upstream

The upstream SWE-Master code documents a great recipe, but in its published form it was developed against the BOSS Zhipin internal cluster (hardcoded pip mirror, TCP-2375 Docker, internal cache paths). Running it on a fresh machine reveals:

- **3 latent bugs** in `R2E-Gym/.../docker.py` (Python local-variable shadowing of `make_test_spec` in 3 branches)
- **3 environment-coupling defects** (hardcoded `pypi-mirror.weizhipin.com`, hardcoded `tcp://...:2375` for Docker, mandatory TLS env vars that fail without a cert)
- **A teacher-rollout context cliff** (default agent `max_tokens` ≥ vLLM `max-model-len`, blocking any conversation)
- **Three memory chokes** under modern transformers + 32K context that OOM on <80GB GPUs (CE backward, fp32 logits upcast, gradient-accumulation no-op divide)

This fork ships:

| Addition | What it does |
|---|---|
| `install_sft_env.sh` | One-shot install of OpenRLHF + flash-attn + DeepSpeed, with all version pins that took us a day of debugging to find |
| `install_rollout_env.sh` | R2E-Gym agent client + litellm + 3 SWE wheels + transformers/hf_hub pinning |
| `install_vllm_env.sh` | vLLM with cu126 torch + transformers<5 pin |
| `apply_patches.sh` | Idempotent application of 10 patches (full mode) or 6 patches (h200 mode — skips memory band-aids) |
| `serve_vllm.sh` | Parameterized vLLM launcher (any model, any TP, any context) |
| `run_rollout_smoke.sh` | Parameterized R2E-Gym sweep (any teacher, any K, any worker count) |
| `rollout_smoke/convert_rollouts_to_sft.py` | R2E trajectory → OpenRLHF SFT-format converter + reward==1 filter |
| `OpenRLHF_SFT/scripts_swe_master/qwen_25_coder_3B_smoke.sh` | Working 3B SFT smoke (demo data) |
| `OpenRLHF_SFT/scripts_swe_master/qwen_25_coder_3B_final_sft.sh` | Working 3B SFT on real rollout-derived data |
| `verify_setup.py` | Sanity-check probe for all 3 venvs + Docker + vLLM reachability |
| `run_full_pipeline.sh` | End-to-end orchestrator (envs → patches → serve → rollout → convert → train) |
| `SETUP.md` | Step-by-step setup guide (this fork) |
| `README.md` | This file |

**See [SETUP.md](SETUP.md) for the run order.**

---

## Hardware matrix

| Hardware | What works | What doesn't fit | Recommended mode |
|---|---|---|---|
| **8× RTX 6000 Ada (48GB, no NVLink, PCIe)** | 3B/7B SFT, all teacher inference (TP=2 for 32B, TP=8 for 30B-A3B) | 32B SFT (paper-scale; needs ring-attn + 16+ GPUs) | `apply_patches.sh` (full) |
| **8× H100 80GB** | Same + 7B SFT at 80K context, 14B SFT at 32K | 32B SFT at 80K (still tight) | `apply_patches.sh` (lean — drop memory band-aids) |
| **8× H200 141GB** | Everything in the paper, including 32B SFT at 32K and probably 80K | n/a | `apply_patches.sh h200` |
| **1× 80GB single-GPU** | Only inference; SFT possible with LoRA / smaller model | data-parallel SFT, multi-GPU vLLM serving | full patches but reduced TP / micro-batch |

---

## Patches this fork makes to upstream code

`apply_patches.sh` applies these. Idempotent. Two modes:

| # | ID | File | Category | full | h200 |
|---|---|---|---|---|---|
| 1 | `openrlhf_ce_inplace` | `OpenRLHF/openrlhf/models/utils.py` | memory band-aid | ✅ | ⊘ |
| 2 | `openrlhf_skip_fp32_upcast` | `OpenRLHF/openrlhf/models/actor.py` | memory band-aid | ✅ | ⊘ |
| 3 | `deepspeed_gas1_shortcircuit` | `.venv/.../deepspeed/runtime/engine.py` | memory band-aid | ✅ | ⊘ |
| 4–6 | `r2egym_make_test_spec_shadow_1` (covers 3 sites) | `R2E-Gym/.../runtime/docker.py` | **real bug** | ✅ | ✅ |
| 7 | `r2egym_unix_socket` | `R2E-Gym/.../runtime/docker.py` | env (Docker daemon) | ✅ | ✅* |
| 8 | `r2egym_skip_tls_local` | `R2E-Gym/.../runtime/docker.py` | env (Docker TLS) | ✅ | ✅* |
| 9 | `r2egym_pypi_mirror` (3 sites) | `R2E-Gym/.../runtime/docker.py` | env (network) | ✅ | ✅ |
| 10 | `r2egym_max_tokens_4096` (2 sites) | `R2E-Gym/.../agent/agent.py` | context-size | ✅ | ⊘** |

\* Only needed if your H200 box's Docker daemon listens on the Unix socket (most do).
\** Skipped on H200 if you run vLLM with `MAX_MODEL_LEN=131072`. Required if any teacher's context is ≤ 16K.

**Reasoning behind each patch** is documented as comments in `apply_patches.py`. Re-running `bash apply_patches.sh --dry-run` shows current state without modifying files.

---

## Findings from our experiments

Quick summary of what we tested (full session log in `notes/findings.md` if present):

| Teacher | scaffold | context | reward==1 rate on Verified | notes |
|---|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct | fn_calling | 16K | 0% | model emits empty `<function=>` templates — can't follow fn-call schema |
| Qwen2.5-Coder-14B-Instruct | fn_calling | 16K | 0% | same failure mode as 7B |
| Qwen2.5-Coder-14B-Instruct | **non_fn_calling** | 16K | 0% | real tool calls + sensible reproduction test; context overflow before fix |
| Qwen2.5-Coder-14B-Instruct | non_fn_calling | 32K | 0% | environment-debugging rabbit hole, never read problem statement |
| Qwen2.5-Coder-32B-Instruct | non_fn_calling | 32K (TP=2) | 0% | same env-debug failure mode |
| **SWE-Master-32B-SFT** (paper's published student) | **non_fn_calling** | **128K** (TP=8) | **23%** (K=100) | real engineering: explore → reproduce → diagnose → edit → test |
| **Qwen3-Coder-30B-A3B-Instruct** (MoE) | **non_fn_calling** | **128K** (TP=4) | **38%** (K=71) | strong open-source teacher, better than the paper's own SFT'd model on our sample |

**Key takeaways:**
- **Use `non_fn_calling` scaffold** for any model not specifically SFT'd against the openhands `fn_calling` schema. The fn-call format requires fine-tuning to emit reliably; raw coder models can't do it.
- **128K context is needed for sustained agent loops** — 32K hits context overflow during diagnostic phases of harder bugs.
- **Qwen3-Coder-30B-A3B-Instruct is a great free teacher** for this scaffold, outperforming Qwen2.5-Coder-32B-Instruct (dense) on our sample.
- **The paper's SWE-Master-32B-SFT model can be used as a teacher for further self-distillation** — meta but works.

---

## Citation & credit

This fork builds on the upstream SWE-Master work. If you use this code, **please cite the original paper**:

```bibtex
@misc{song2026swemasterunleashingpotentialsoftware,
      title={SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training}, 
      author={Huatong Song and Lisheng Huang and Shuang Sun and Jinhao Jiang and Ran Le and Daixuan Cheng and Guoxin Chen and Yiwen Hu and Zongchao Chen and Wayne Xin Zhao and Yang Song and Tao Zhang and Ji-Rong Wen},
      year={2026},
      eprint={2602.03411},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2602.03411}, 
}
```

This fork is **maintained by the user, not the original authors.** Issues with the upstream paper/methods → file at `RUCAIBox/SWE-Master`. Issues with the fork-specific scripts / patches → file here.

---

## License

[MIT](LICENSE) (matches upstream).
