"""
analyze/plot_prop_a_convergence.py
==================================
Empirical contraction-rate check for Proposition A.

Prop. A claim
-------------
For unbalanced entropic OT with KL prox and per-pixel ε in M but scalar ε in
the prox operator, the Sinkhorn iteration is a Banach contraction in the
Hilbert projective metric with rate

    κ_theo  =  tanh( osc(M) / 4 )  ·  ( ρ / (ρ + ε) )²        (KL prox)
    κ_theo  =  tanh( osc(M) / 4 )                              (TV prox)

Empirical rate
--------------
For the per-iter trace `δ_k = ‖log_f_{k+1} − log_f_k‖_∞`,

    κ_emp(k)  =  δ_{k+1} / δ_k.

The figure stacks κ_emp(k) over the grad-bearing Sinkhorn iterations against
the horizontal κ_theo line.  Prop. A passes iff every empirical point lies
≤ κ_theo (with a small slack — round-off near the fixed point inflates κ_emp
once δ_k is below ~1e-6).

Usage
-----
    python analyze/plot_prop_a_convergence.py \\
        --checkpoint runs/full/checkpoint_best.pth.tar \\
        --image      /data/kodak/kodim01.png \\
        --slices     0 2 4 \\
        --eps-sweep  0.05 0.1 0.2 \\
        -o prop_a_convergence.pdf \\
        --cuda
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib.pyplot as plt
import torch

from _common import load_image, load_model


@torch.no_grad()
def _capture_trace(model, x_padded: torch.Tensor, slice_idx: int) -> dict:
    """
    Enable `_audit_convergence` on the target slice's UnifiedDictionaryAttention,
    run one forward pass, then read back the trace.
    """
    attn = model.eot_attentions[slice_idx]
    prev_flag = attn._audit_convergence
    attn._audit_convergence = True
    try:
        _ = model(x_padded)
    finally:
        attn._audit_convergence = prev_flag
    trace = attn.convergence_trace()
    if trace is None:
        raise RuntimeError(
            f"Slice {slice_idx}: convergence_trace() returned None. "
            "Routing may have skipped the unbalanced_eot branch."
        )
    return trace


def _empirical_kappa(diffs: list[float]) -> list[float]:
    """κ_emp(k) = δ_{k+1} / δ_k.  Returns NaN for ratios where δ_k ≤ floor."""
    floor = 1e-12
    out = []
    for k in range(len(diffs) - 1):
        denom = diffs[k]
        if denom <= floor:
            out.append(float("nan"))
        else:
            out.append(diffs[k + 1] / denom)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--slices", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument(
        "--eps-sweep",
        type=float,
        nargs="+",
        default=None,
        help=(
            "If provided, override the model's learned ε at each value and "
            "produce one panel per ε.  Otherwise uses the model's learned ε."
        ),
    )
    ap.add_argument("-o", "--output", default="prop_a_convergence.pdf")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    model = load_model(args.checkpoint, device=device, strict_load=False)
    model.eval()
    _, x_padded, _, _ = load_image(args.image, device=device)

    eps_values = args.eps_sweep if args.eps_sweep else [None]
    n_panels = len(eps_values)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 4), sharey=True, squeeze=False
    )

    for col, eps_v in enumerate(eps_values):
        ax = axes[0, col]
        # Optional ε override across all slices.
        if eps_v is not None:
            for attn in model.eot_attentions:
                attn.log_eps.data.fill_(math.log(eps_v))
        kappa_theos: list[tuple[int, float]] = []
        for s in args.slices:
            tr = _capture_trace(model, x_padded, slice_idx=s)
            diffs = tr["log_f_diff_per_iter"]
            k_emp = _empirical_kappa(diffs)
            ax.semilogy(
                range(1, len(k_emp) + 1),
                k_emp,
                marker="o",
                color=f"C{args.slices.index(s)}",
                label=f"slice {s} (ε={tr['eps_target']:.3g}, "
                f"ρ={tr['rho_col']:.2f})",
            )
            kappa_theos.append((s, tr["theoretical_kappa"]))
        # Draw κ_theo per slice — colour-matched to the corresponding curve.
        for s, k in kappa_theos:
            ax.axhline(
                k,
                ls="--",
                alpha=0.4,
                color=f"C{args.slices.index(s)}",
            )
        ax.set_xlabel("Sinkhorn iter k")
        ax.set_ylabel(r"$\kappa_{\mathrm{emp}}(k) = \delta_{k+1}/\delta_k$")
        ax.set_title(
            "learned ε" if eps_v is None else f"ε override = {eps_v:.3g}"
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        r"Prop. A — empirical contraction rate $\leq$ "
        r"$\tanh(\mathrm{osc}(M)/4)\cdot(\rho/(\rho+\varepsilon))^2$ "
        "(dashed)"
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"saved → {args.output}")


if __name__ == "__main__":
    main()
