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
        # column-entropy bonus + row-entropy sparsity penalty + anti-leakage
        # alignment hinge.  KEPT for backwards compatibility with the
        # original Table 4; use the three split variants below
        # (no_col_entropy / no_row_entropy / no_alignment) for the diagnostic
        # breakdown — reviewers consistently ask "which of the three matters?"
        "--column-entropy-weight": 0.0,
        "--row-entropy-weight": 0.0,
        "--alignment-weight": 0.0,
    },
    # ── Split version of no_disp_bonus — Phase-2 diagnostic ──────────
    "no_col_entropy": {
        # Drop only the −H_col bonus.  Tests the "every atom is used"
        # claim in isolation.
        "--column-entropy-weight": 0.0,
    },
    "no_row_entropy": {
        # Drop only the +H_row penalty.  Tests the "per-pixel sparsity"
        # claim in isolation.
        "--row-entropy-weight": 0.0,
    },
    "no_alignment": {
        # Drop only the complexity-alignment hinge.  Tests the "anti-
        # leakage" claim in isolation.
        "--alignment-weight": 0.0,
    },
    "no_alignment_margin": {
        # Keep the alignment loss but set margin=0 (legacy sign-only hinge).
        # Tests whether the strictly-positive correlation requirement
        # (κ = 0.2) actually buys anything over the sign hinge.
        "--alignment-margin": 0.0,
    },
    "no_dict_penalty": {
        # Set fixed dictionary-penalty weight to zero.
        "--dict-penalty-weight": 0.0,
    },
    # ── CMIC-derived components (added Phase 1–3) ─────────────────────
    "no_wls": {
        # Disable WLS/iWLS multi-scale wavelet shortcuts.  In config.sh
        # this is realised by STRIPPING `--use-wls-shortcut` from the
        # baseline command line via the `__SKIP_WLS__` sentinel that the
        # bash helper recognises (see config.sh::common_arch_flags).
        # The Python config has no flag to add because the baseline
        # passes `--use-wls-shortcut` and we *remove* it — there is no
        # corresponding "no-wls-shortcut" flag to add.
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
    # ── Multi-marginal OT ablation (Phase B8) ─────────────────────────
    # The full model defaults to use_conditional_marginals=False; the
    # *positive* ablation toggles it ON to measure the cross-slice
    # dictionary-specialisation gain.
    "full_cond_marg": {
        "--use-conditional-marginals": True,
        "--cond-alpha": 0.5,
    },
    "full_cond_marg_strong": {
        # Higher cross-slice coupling (α = 0.9, near-maximum).
        "--use-conditional-marginals": True,
        "--cond-alpha": 0.9,
    },
    # ── OT-specific ablations (THEORY.md §5) ──────────────────────────
    # Each variant isolates ONE OT property and maps to a numbered
    # proposition / table row in THEORY.md §5.
    "balanced_eot_only": {
        # §5.3 — disable the unbalanced row-marginal relaxation while
        # keeping balanced column constraint.  Tests Proposition 5.2:
        # does spatially-varying ρ actually help?  Reviewer-favourite
        # ablation since it isolates "unbalanced" from "balanced + ρ=∞".
        "--routing-mode": "balanced_eot",
    },
    "marg_div_tv": {
        # §5.6 — switch the unbalanced prox from KL{ρ} (smooth) to
        # TV{ρ} (hard gating).  Empirical, no theoretical winner.
        "--marginal-div": "tv",
    },
    "low_eps": {
        # §5.5 — pin ε near the floor (initial value, then sigmoid still
        # adapts upward).  Tests the "small ε = sharp routing but stiff
        # gradients" hypothesis.  Expected: noisier training, possible
        # rate gain if it converges.
        "--ot-eps": 0.06,
    },
    "high_eps": {
        # §5.5 — start ε in the upper range.  Smooth routing, P closer
        # to a ⊗ b, side-info MI drops.  Expected: rate loss.
        "--ot-eps": 0.5,
    },
    "sinkhorn_5iter": {
        # §5.7 — reduce Sinkhorn iterations from 20 to 5.  Tests
        # geometric-convergence claim (Eq. 18): at q ≈ 0.5 we should
        # still be at 3 % of the optimum, so rate impact ≈ 0.
        "--sinkhorn-iters": 5,
    },
    "cond_alpha_0p1": {
        # §5.4 — light cross-slice coupling (α = 0.1).  Together with
        # 0.3 / 0.5 / 0.9 forms a 4-point sweep of Proposition 5.3.
        "--use-conditional-marginals": True,
        "--cond-alpha": 0.1,
    },
    "cond_alpha_0p3": {
        "--use-conditional-marginals": True,
        "--cond-alpha": 0.3,
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
    "no_col_entropy",
    "no_row_entropy",
    "no_alignment",
    "no_alignment_margin",
    "no_dict_penalty",
    "no_wls",
    "no_olp",
    "no_ste_y",
    "full_cond_marg",
    "full_cond_marg_strong",
    # OT-specific (THEORY.md §5)
    "balanced_eot_only",
    "marg_div_tv",
    "low_eps",
    "high_eps",
    "sinkhorn_5iter",
    "cond_alpha_0p1",
    "cond_alpha_0p3",
]

# ──────────────────────────────────────────────────────────────────────────
# REQUIRED FLAG SUPPORT  (must be implemented in the project)
# ──────────────────────────────────────────────────────────────────────────
# train.py / models/WMDC.py must accept the following CLI flags:
#
#   --routing-mode {softmax, balanced_eot, unbalanced_eot}
#   --backbone {fdm, cnn, swin, ss2d, fdm_reversed}    (ablation_models/backbone_variants.py)
#   --use-dense-concat                                  (analyze/measure_vram.py footer)
#   --memory-init {bootstrap, zero}                     (see WMDC.init_memory)
#   --column-entropy-weight <float>                     (β_col: −H_col bonus)
#   --row-entropy-weight    <float>                     (β_row: H_row penalty)
#   --alignment-weight      <float>                     (γ: alignment hinge)
#   --alignment-margin      <float>                     (κ: margin inside the hinge)
#   --dict-penalty-weight   <float>                     (δ: token diversity)
#   --use-conditional-marginals                         (B8: multi-marginal OT)
#   --cond-alpha            <float>                     (α: cross-slice coupling)
#   --marginal-div          {kl, tv}                    (unbalanced prox choice)
#   --ot-eps                <float>                     (initial Sinkhorn ε)
#   --sinkhorn-iters        <int>                       (number of Sinkhorn iters)
#
# When in doubt about a flag's effect, search train.py for the existing
# wiring in RateDistortionLoss and follow the same pattern.
