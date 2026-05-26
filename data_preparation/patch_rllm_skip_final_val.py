"""Patch DeepSWE_RL/rllm's `agent_ppo_trainer.py` so the post-training
"final validation" honours `test_freq > 0` like the periodic validation
already does.

Why: at the bottom of `fit_agent` (line ~449) rllm runs

    if self.global_steps >= self.total_training_steps:
        if self.val_reward_fn is not None:
            val_metrics = self._validate_agent()

The conditional is missing the `test_freq > 0` guard that line 430 has
for the *periodic* validation. Net effect: setting `trainer.test_freq=-1`
disables MID-training validation but the final post-training validation
fires unconditionally as soon as you supply `data.val_files`. On a tiny
smoke run (`total_training_steps=1`, val set = 500-instance
SWE-Bench-Verified) that's a 5-hour val pass after a 6-minute training
step, with no signal because nothing was learned.

After this patch:
    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0:
        val_metrics = self._validate_agent()

Same gating as the periodic check on line 430. `test_freq=-1` now cleanly
means "no validation, ever". Real training runs (e.g. deepswe_32b.sh)
set `test_freq=N>0`, so they're unaffected.

Idempotent: safe to re-run.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET = os.path.join(REPO, "DeepSWE_RL", "rllm", "rllm", "trainer", "verl", "agent_ppo_trainer.py")
SENTINEL = "# patched-by: patch_rllm_skip_final_val.py"

# The exact line we want to amend. Anchored to enough surrounding context
# (the `>= self.total_training_steps:` line plus the `# perform validation
# after training` comment) that we can't accidentally match the periodic
# check on line 430.
OLD = (
    "                if self.global_steps >= self.total_training_steps:\n"
    "                    # perform validation after training\n"
    "                    if self.val_reward_fn is not None:\n"
)
NEW = (
    "                if self.global_steps >= self.total_training_steps:\n"
    "                    # perform validation after training\n"
    "                    " + SENTINEL + "\n"
    "                    # Gate on test_freq>0 (matches periodic check on line 430).\n"
    "                    # Without this, test_freq=-1 still triggers a full final\n"
    "                    # validation pass after the last training step.\n"
    "                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0:\n"
)


def main() -> int:
    if not os.path.exists(TARGET):
        print(f"ERROR: target not found: {TARGET}")
        return 1
    with open(TARGET) as f:
        src = f.read()
    if SENTINEL in src:
        print(f"already patched: {TARGET}")
        return 0
    if OLD not in src:
        print(f"ERROR: anchor block not found in {TARGET}")
        print("       upstream may have rewritten fit_agent; review by hand.")
        return 2
    new_src = src.replace(OLD, NEW, 1)
    with open(TARGET, "w") as f:
        f.write(new_src)
    print(f"patched: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
