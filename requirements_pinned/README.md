# Pinned requirements

These files are `uv pip freeze` snapshots of each working venv at the time of pipeline validation. They are **reference / lockfile substitute**, not the primary install path.

## Files

| File | venv | What it pins |
|---|---|---|
| `sft.txt` | `OpenRLHF_SFT/.venv` | OpenRLHF + flash-attn + DeepSpeed + Liger + transformers<5 |
| `rollout.txt` | `R2E-Gym/.venv` | R2E-Gym client + litellm + swebench wheels + transformers<5 + hf_hub<1 |
| `vllm.txt` | `vllm_venv` | vLLM 0.11 + cu126 torch + transformers<5 |

## Primary install path: use the install scripts

The fast and supported way to set up each venv is still:

```bash
bash install_sft_env.sh
bash install_rollout_env.sh
bash install_vllm_env.sh
```

These scripts apply the version constraints that took us a day of debugging to find (see `README.md` and `SETUP.md`). They don't use these freeze files.

## When to use the freeze files

**Only as a fallback** if a future version drift breaks the install scripts. To restore a working state from a frozen snapshot:

```bash
# Example for the SFT venv
cd OpenRLHF_SFT
uv venv --python 3.11 --clear
source .venv/bin/activate
uv pip install -r ../requirements_pinned/sft.txt
# Then re-apply the editable installs that the freeze file omits:
git clone https://github.com/OpenRLHF/OpenRLHF.git
uv pip install -e ./OpenRLHF --no-build-isolation
```

(Equivalent dances exist for the other two venvs.)

## Caveats

- These freeze files **omit `-e .` (editable) packages** like `r2egym` and `openrlhf` — you still have to install those manually from source.
- The PyPI versions in these pins may eventually be **yanked** by upstream. If that happens, the install scripts (which use looser constraints) are more likely to still work.
- These snapshots were taken on Python 3.11 + CUDA 12.6. Different Python or CUDA versions will need different wheel filenames.

## Regenerating

If you upgrade something and want to refresh the snapshots:

```bash
for venv_label in "OpenRLHF_SFT/.venv:sft" "R2E-Gym/.venv:rollout" "vllm_venv:vllm"; do
    vdir="${venv_label%:*}"
    label="${venv_label##*:}"
    VIRTUAL_ENV="$vdir" uv pip freeze | grep -vE "^-e " \
        > "requirements_pinned/${label}.txt"
done
```
