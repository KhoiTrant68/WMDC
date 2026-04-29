"""
configs/ablation_variants.py
=============================
Defines the 6 component-removal ablation variants used in Table 4 of the
paper. Each variant is a dict mapping CLI flag names to values; the
training launcher (scripts/run_ablation.sh) reads this file and builds
training commands.

The variants ablate ONE component at a time relative to the full WMDC,
chosen to test the highest-leverage architectural claims of the paper.

Variant list
------------
  full              — reference (no removal); all components active.
  no_ueot           — UEOT routing → softmax routing.
  no_fdm            — FDM block   → standard SS2D Mamba block (no wavelet).
  no_stateful_mem   — stateful memory → dense channel-autoregressive concat.
  no_bootstrap_M1   — M_1 bootstrapped from Φ → M_1 = 0 (zero init).
  no_disp_bonus     — remove the −α·H(̄m) entropy bonus from the loss.
  no_dict_penalty   — remove the +β·P_dict diversity penalty from the loss.

Notes for the trainer
---------------------
* The full ablation costs ~6 variants × 3 λ values × ~400 GPU-hours per
  run. Plan the schedule in scripts/run_ablation.sh accordingly.
* The flag names below assume the trainer accepts them. If your trainer
  doesn't yet, the flags must be wired into train.py and models/WMDC.py
  (see "REQUIRED FLAG SUPPORT" footer below).
"""

from __future__ import annotations

# Lambda subset for the ablation (low / mid / high). Full RD curve uses
# 6 lambdas; ablation uses 3 to keep total compute under control.
ABLATION_LAMBDAS = [0.0035, 0.013, 0.0483]

VARIANTS: dict[str, dict] = {
    "full": {
        # No flags overridden — use defaults.
    },
    "no_ueot": {
        # Replace UEOT with temperature-scaled softmax.
        "--routing-mode": "softmax",
    },
    "no_fdm": {
        # Replace FDM block with vanilla SS2D Mamba.
        "--backbone": "ss2d",
    },
    "no_stateful_mem": {
        # Disable stateful memory; use dense channel-autoregressive concat.
        "--use-dense-concat": True,
    },
    "no_bootstrap_M1": {
        # Bootstrap M_1 from zeros instead of from hyper-prior.
        "--memory-init": "zero",
    },
    "no_disp_bonus": {
        # Set EMA-adaptive disp weight base to zero → no entropy bonus.
        "--disp-weight": 0.0,
    },
    "no_dict_penalty": {
        # Set fixed dictionary-penalty weight to zero.
        "--dict-penalty-weight": 0.0,
    },
}

# Order in which to run the variants (full first as the reference).
RUN_ORDER = [
    "full",
    "no_ueot",
    "no_fdm",
    "no_stateful_mem",
    "no_bootstrap_M1",
    "no_disp_bonus",
    "no_dict_penalty",
]

# ──────────────────────────────────────────────────────────────────────────
# REQUIRED FLAG SUPPORT  (must be implemented in the project)
# ──────────────────────────────────────────────────────────────────────────
# train.py / models/WMDC.py must accept the following CLI flags:
#
#   --routing-mode {softmax, balanced_eot, unbalanced_eot}
#   --backbone {fdm, cnn, swin, ss2d, fdm_reversed}    (NEW; see ablation_models/backbone_variants.py)
#   --use-dense-concat                                  (NEW; see analyze/measure_vram.py footer)
#   --memory-init {bootstrap, zero}                     (NEW; trivial — see WMDC.init_memory)
#   --disp-weight <float>                               (existing)
#   --dict-penalty-weight <float>                       (existing — see train.py update)
#
# When in doubt about a flag's effect, search train.py for the existing
# disp-weight/dict-penalty-weight wiring and follow the same pattern.
