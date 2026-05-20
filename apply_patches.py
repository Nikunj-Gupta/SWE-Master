#!/usr/bin/env python3
"""
Apply the 10 local patches we discovered are needed to make the upstream
SWE-Master pipeline run on RTX 6000 Ada-class hardware (48 GB cards).

Each patch is encoded as a (file, old_string, new_string, reason) tuple.
The script is idempotent: if `new_string` already exists in the file, the
patch is skipped. If neither `old_string` nor `new_string` is found, we
print a warning (probably means the upstream code changed shape).

Usage:
    python apply_patches.py [--mode full|h200]

Modes:
    full  (default)  Apply all 10 patches. Required for <80GB GPUs.
    h200             Apply only the 6 real-bug / environment patches.
                     Skip the 4 memory band-aids (#1, #2, #3, #10).
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent


# ====================================================================
# Patch definitions
# ====================================================================

# Each entry: dict with keys
#   id            short tag (used by --mode h200 to filter)
#   category      "memory" (band-aid) | "bug" | "env" | "context"
#   file          absolute path
#   old           string to find (must be unique in file)
#   new           replacement
#   reason        human description
#   required      if True, missing old_string is an error not a warning

PATCHES = [
    # ----------------------------------------------------------------
    # Memory band-aids (skipped on H200)
    # ----------------------------------------------------------------
    {
        "id": "openrlhf_ce_inplace",
        "category": "memory",
        "file": REPO / "OpenRLHF_SFT/OpenRLHF/openrlhf/models/utils.py",
        "old": (
            '            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss\n'
            '\n'
            '            output = cross_entropy_loss(logits.reshape(-1, last_dim), labels.reshape(-1))'
        ),
        "new": (
            '            from flash_attn.ops.triton.cross_entropy import cross_entropy_loss\n'
            '\n'
            '            # inplace_backward=True reuses the logits buffer for dlogits, saving\n'
            '            # ~seq_len*vocab*4 bytes (≈18GB at 32K context, Qwen2.5 vocab=151,936).\n'
            '            # Safe here because `logits` is not used after this call.\n'
            '            output = cross_entropy_loss(\n'
            '                logits.reshape(-1, last_dim), labels.reshape(-1), inplace_backward=True\n'
            '            )'
        ),
        "reason": "Avoid 18GB dlogits buffer alloc on backward at 32K context",
    },
    {
        "id": "openrlhf_skip_fp32_upcast",
        "category": "memory",
        "file": REPO / "OpenRLHF_SFT/OpenRLHF/openrlhf/models/actor.py",
        "old": (
            '        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, **mm_inputs)\n'
            '        # https://github.com/OpenRLHF/OpenRLHF/pull/634\n'
            '        output["logits"] = output["logits"].to(torch.float32)'
        ),
        "new": (
            '        output = self.model(sequences, attention_mask=foward_attention_mask, position_ids=position_ids, **mm_inputs)\n'
            '        # https://github.com/OpenRLHF/OpenRLHF/pull/634\n'
            '        # The fp32 upcast doubles logits memory (seq×vocab×2 → seq×vocab×4).\n'
            '        # At 32K seq × Qwen2.5 vocab=151,936 that\'s a 9.7→19.4GB jump per GPU\n'
            '        # which won\'t fit on 48GB cards. Keep bf16 and let log_probs_from_logits\n'
            '        # fall into its else branch (looping F.log_softmax, still in bf16).\n'
            '        # Trade-off: bf16 log_softmax is less numerically stable on a 152K vocab.\n'
            '        # For smoke this is acceptable; revisit for production runs.\n'
            '        # output["logits"] = output["logits"].to(torch.float32)'
        ),
        "reason": "Avoid 10GB fp32 logits buffer at 32K context",
    },
    {
        "id": "deepspeed_gas1_shortcircuit",
        "category": "memory",
        "file": REPO / "OpenRLHF_SFT/.venv/lib/python3.11/site-packages/deepspeed/runtime/engine.py",
        "old": (
            '    def _backward_prologue_per_tensor(self, grad):\n'
            '        # Only scale gradients if scale_wrt_gas is True, consistent with backward() parameter\n'
            '        if grad is not None and self._scale_wrt_gas:\n'
            '            return grad / self.gradient_accumulation_steps()\n'
            '        return grad'
        ),
        "new": (
            '    def _backward_prologue_per_tensor(self, grad):\n'
            '        # Only scale gradients if scale_wrt_gas is True, consistent with backward() parameter\n'
            '        if grad is not None and self._scale_wrt_gas:\n'
            '            gas = self.gradient_accumulation_steps()\n'
            '            # `grad / 1` would still allocate a fresh tensor (~9GB for the LM head\n'
            '            # gradient at seq=32K, vocab=151,936), which blows past 48GB cards.\n'
            '            # Skip the no-op divide when gas == 1.\n'
            '            if gas == 1:\n'
            '                return grad\n'
            '            return grad / gas\n'
            '        return grad'
        ),
        "reason": "Avoid 9GB pointless tensor copy when gradient_accumulation_steps==1",
        "required": False,  # lives in .venv, may not be present yet
    },

    # ----------------------------------------------------------------
    # Real upstream bugs (always apply)
    # ----------------------------------------------------------------
    # Sites 4-6: make_test_spec local-var shadowing in 3 branches (swegym, swebench-verified, swerebench)
    # The fix is identical for all three branches; we use replace_all to hit all three.
    {
        "id": "r2egym_make_test_spec_shadow_1",
        "category": "bug",
        "file": REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
        "old": (
            "                self.logger.info(\"self.ds has make_test_spec, read directly\")\n"
            "                make_test_spec = json.loads(self.ds['make_test_spec'])\n"
            "                self.test_spec = TestSpec(**make_test_spec)"
        ),
        "new": (
            "                self.logger.info(\"self.ds has make_test_spec, read directly\")\n"
            "                # Renamed local var so it doesn't shadow the module-level\n"
            "                # `make_test_spec` function (Python marks the name local for\n"
            "                # the whole function and breaks the else branch).\n"
            "                spec_data = json.loads(self.ds['make_test_spec'])\n"
            "                self.test_spec = TestSpec(**spec_data)"
        ),
        "reason": "Fix UnboundLocalError: local var shadows module-level make_test_spec function",
        "replace_all": True,  # hit all 3 branches in one shot
    },

    # ----------------------------------------------------------------
    # Environment patches (Docker local socket + PyPI mirror)
    # ----------------------------------------------------------------
    {
        "id": "r2egym_unix_socket",
        "category": "env",
        "file": REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
        "old": (
            "        self.ip = ip\n"
            "        self.docker_host = r\"tcp://\" + self.ip + r\":2375\""
        ),
        "new": (
            "        self.ip = ip\n"
            "        # Use the Unix socket for local Docker — the default daemon listens\n"
            "        # only on /var/run/docker.sock and not TCP 2375. Override only for\n"
            "        # remote IPs where TCP 2375 is the right endpoint.\n"
            "        if ip in (\"127.0.0.1\", \"localhost\"):\n"
            "            self.docker_host = \"unix:///var/run/docker.sock\"\n"
            "        else:\n"
            "            self.docker_host = \"tcp://\" + self.ip + \":2375\""
        ),
        "reason": "Use Unix socket for local Docker (TCP 2375 not enabled by default)",
    },
    {
        "id": "r2egym_skip_tls_local",
        "category": "env",
        "file": REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
        "old": (
            "        custom_env = {\n"
            "            'DOCKER_HOST': self.docker_host, \n"
            "            'DOCKER_TLS_VERIFY': DOCKER_TLS_VERIFY, \n"
            "            'DOCKER_CERT_PATH': DOCKER_CERT_PATH, \n"
            "            # 'DOCKER_API_VERSION': '1.40' \n"
            "        }"
        ),
        "new": (
            "        custom_env = {\n"
            "            'DOCKER_HOST': self.docker_host,\n"
            "            # 'DOCKER_API_VERSION': '1.40'\n"
            "        }\n"
            "        # Only set TLS env vars for remote TCP daemons. With the Unix socket,\n"
            "        # DOCKER_TLS_VERIFY=1 makes docker-py demand a cert path that doesn't\n"
            "        # exist on local boxes.\n"
            "        if self.ip not in (\"127.0.0.1\", \"localhost\"):\n"
            "            custom_env['DOCKER_TLS_VERIFY'] = DOCKER_TLS_VERIFY\n"
            "            custom_env['DOCKER_CERT_PATH'] = DOCKER_CERT_PATH"
        ),
        "reason": "Skip TLS env vars when on Unix socket (docker-py demands cert files otherwise)",
    },
    {
        "id": "r2egym_pypi_mirror",
        "category": "env",
        "file": REPO / "R2E-Gym/src/r2egym/agenthub/runtime/docker.py",
        "old": 'self.run("pip install chardet --trusted-host pypi-mirror.weizhipin.com -i http://pypi-mirror.weizhipin.com/bzl-aliyun-pypi/simple")',
        "new": 'self.run("pip install chardet")',
        "reason": "Replace BOSS Zhipin internal PyPI mirror with public PyPI (3 sites)",
        "replace_all": True,
    },

    # ----------------------------------------------------------------
    # Context-size dependent (skipped on H200 if running native 128K)
    # ----------------------------------------------------------------
    {
        "id": "r2egym_max_tokens_4096",
        "category": "context",
        "file": REPO / "R2E-Gym/src/r2egym/agenthub/agent/agent.py",
        "old": (
            "        else:\n"
            "            lite_llm_max_token = 16384"
        ),
        "new": (
            "        else:\n"
            "            lite_llm_max_token = 4096  # Lowered from 16384 to fit with 16K total context"
        ),
        "reason": "Lower max_tokens so it fits within a 16K context window (2 sites)",
        "replace_all": True,
    },
]


def apply_patch(p: dict, dry_run: bool = False) -> str:
    """Apply one patch. Returns one of: 'applied', 'already', 'missing', 'error'."""
    fp = p["file"]
    if not fp.exists():
        if p.get("required", True):
            print(f"  [error] file not found: {fp}")
            return "error"
        else:
            print(f"  [skip] file not present yet (may need a fresh install first): {fp.name}")
            return "missing"

    text = fp.read_text()
    if p["new"] in text:
        # Already applied — idempotent skip
        print(f"  [skip] already applied")
        return "already"
    if p["old"] not in text:
        print(f"  [warn] old string not found in {fp.name} (upstream may have changed)")
        return "missing"

    # Count occurrences for safety
    n = text.count(p["old"])
    if not p.get("replace_all") and n > 1:
        print(f"  [error] old string is ambiguous ({n} occurrences) without replace_all")
        return "error"

    new_text = text.replace(p["old"], p["new"])
    if not dry_run:
        fp.write_text(new_text)
    print(f"  [ok] applied{' (DRY-RUN)' if dry_run else ''} ({n} replacement{'s' if n != 1 else ''})")
    return "applied"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["full", "h200"], default="full",
                    help="full: apply all 10 patches (Ada-class). h200: skip 4 memory band-aids.")
    ap.add_argument("--dry-run", action="store_true", help="show what would change without writing files")
    args = ap.parse_args()

    skip_categories = {"memory", "context"} if args.mode == "h200" else set()

    counts = {"applied": 0, "already": 0, "missing": 0, "error": 0}
    for p in PATCHES:
        if p["category"] in skip_categories:
            print(f"== [{p['category']}] {p['id']} — SKIPPED for --mode h200")
            continue
        print(f"== [{p['category']}] {p['id']}")
        print(f"     {p['reason']}")
        result = apply_patch(p, dry_run=args.dry_run)
        counts[result] += 1

    print()
    print(f"summary: {counts['applied']} applied, {counts['already']} already-applied, "
          f"{counts['missing']} missing, {counts['error']} error")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    sys.exit(main())
