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
  no_disp_bonus     — remove the routing-entropy regularisers from the loss
                      (both column-entropy bonus AND row-entropy penalty
                       AND the anti-leakage alignment hinge).
  no_dict_penalty   — remove the dictionary-token diversity penalty.

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
        # Ablate the ENTIRE dictionary-routing regularisation block:
        # column-entropy bonus, row-entropy sparsity penalty, AND the
        # anti-leakage alignment hinge.  This mirrors the original
        # ablation intent ("remove the −α·H entropy bonus") but is
        # adapted to the refactored MI-decomposition loss.
        "--column-entropy-weight": 0.0,
        "--row-entropy-weight": 0.0,
        "--alignment-weight": 0.0,
    },
    "no_dict_penalty": {
        # Set fixed dictionary-penalty weight to zero.
        "--dict-penalty-weight": 0.0,
    },
    # ── CMIC-derived components (added Phase 1–3) ─────────────────────
    "no_wls": {
        # Disable WLS/iWLS multi-scale wavelet shortcuts.  In config.sh
        # this is realised by stripping --use-wls-shortcut via the
        # __SKIP_WLS__ sentinel token rather than by adding a flag.
    },
    "no_olp": {
        # Disable the OLP orthogonality regulariser (still keeps the OLP
        # module as a drop-in nn.Linear; only the ‖W Wᵀ−I‖² term goes).
        "--ortho-weight": 0.0,
    },
    "no_ste_y": {
        # Train without the STE-on-y schedule (matches pre-Phase-1 behaviour).
        "--last-epochs-with-ste": 0,
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
    "no_wls",
    "no_olp",
    "no_ste_y",
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
#   --column-entropy-weight <float>                     (β_col: −H_col bonus)
#   --row-entropy-weight    <float>                     (β_row: H_row penalty)
#   --alignment-weight      <float>                     (γ: anti-leakage hinge)
#   --dict-penalty-weight   <float>                     (δ: token diversity)
#
# When in doubt about a flag's effect, search train.py for the existing
# wiring in RateDistortionLoss and follow the same pattern.
