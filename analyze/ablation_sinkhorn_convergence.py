"""
analyze/ablation_sinkhorn_convergence.py  (FIXED)
=================================================
Plots the unbalanced Sinkhorn primal objective vs. iteration count, to
empirically justify the choice of `iters=20` in the model.

Bugs fixed vs. previous version
-------------------------------
1. Uses `model._hyper_decode(z_hat)` instead of the now-removed
   `model.h_scale_s` / `model.h_mean_s` (which were merged into
   `h_trunk + h_scale_head + h_mean_head` after refactor).
2. Unpacks the tuple returned by `model.hyper_to_dict(z_hat)`
   (previously assumed it was a single tensor).
3. Adds a sweep over multiple ε values to give the paper's convergence
   study real teeth (Section 3.3 of the paper claims iters=20 is
   sufficient — this script generates the evidence).

Usage
-----
    python analyze/ablation_sinkhorn_convergence.py \
        --checkpoint checkpoints/full/best.pth.tar \
        --image      /data/kodak/kodim01.png \
        --max-iters  30 \
        --eps-sweep  0.05 0.1 0.2 0.5 \
        -o sinkhorn_convergence.pdf \
        --cuda
"""

import argparse
import json

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from _common import load_image, load_model


def _decode_hyper(model, z_hat: torch.Tensor) -> torch.Tensor:
    """
    Run the hyper-decoder. Tries the new shared-trunk path first and falls
    back to the legacy path so this script works on either checkpoint.
    """
    if hasattr(model, "_hyper_decode"):
        scales, means = model._hyper_decode(z_hat)
        return torch.cat([scales, means], dim=1)
    if hasattr(model, "h_trunk"):
        h = model.h_trunk(z_hat)
        scales = model.h_scale_head(h)
        means = model.h_mean_head(h)
        return torch.cat([scales, means], dim=1)
    # Legacy fallback (pre-refactor checkpoints).
    scales = model.h_scale_s(z_hat)
    means = model.h_mean_s(z_hat)
    return torch.cat([scales, means], dim=1)


def _get_dt(model, z_hat: torch.Tensor) -> torch.Tensor:
    """hyper_to_dict returns (dt, dict_penalty) — we only need dt here."""
    out = model.hyper_to_dict(z_hat)
    return out[0] if isinstance(out, tuple) else out


def measure_convergence(
    model,
    x_padded: torch.Tensor,
    max_iters: int,
    *,
    eps_value: float | None = None,
) -> dict:
    """
    For iter counts 1..max_iters, run the unbalanced Sinkhorn solver on
    slice 0 and record the primal objective:

        J(P) = <C, P> + <ρ, KL(P1 || a)> + ε · KL(P^T 1 || b)

    Setting `eps_value` overrides the model's learned ε for the duration
    of the measurement (useful for the ε-sweep figure).
    """
    model.eval()
    attn = model.eot_attentions[0]

    with torch.no_grad():
        # ── Forward to slice-0 inputs ─────────────────────────────────────
        y = model.g_a(x_padded)
        z = model.h_a(y)
        # Use real entropy-coded z_hat for fidelity; falls back to round if
        # the entropy bottleneck CDF tables are not populated.
        try:
            z_str = model.entropy_bottleneck.compress(z)
            z_hat = model.entropy_bottleneck.decompress(z_str, z.size()[-2:])
        except Exception:
            z_hat = torch.round(z)

        dt = _get_dt(model, z_hat)
        hyper_prior = _decode_hyper(model, z_hat)

        rho_spatial = model._compute_rho_spatial(
            slice_idx=0, hyper_prior=hyper_prior, decoded_slices=[]
        )

        k_dict = model.k_projs[0](dt)
        memory_state = model.init_memory(hyper_prior)
        query = torch.cat([hyper_prior, memory_state], dim=1)

        B, _, H, W = query.shape
        HW = H * W

        C_mat = attn._cost_matrix(query, k_dict, H, W)
        rho_flat = rho_spatial.view(B, HW).clamp_min(0.01)

    # ── Iter sweep ───────────────────────────────────────────────────────
    iter_counts = list(range(1, max_iters + 1))
    primal_costs: list[float] = []
    transport_costs: list[float] = []
    kl_row_costs: list[float] = []
    kl_col_costs: list[float] = []

    # Determine effective epsilon.
    if eps_value is None:
        eps_val = F.softplus(attn.log_eps).item() + 0.01
    else:
        eps_val = eps_value

    # Save and override solver state so we can sweep iter count.
    original = (attn.iters, attn.n_grad_iters, attn.n_nograd_iters)
    if eps_value is not None:
        # Temporarily override learned eps. We rebuild from log_eps so we
        # can restore exactly.
        original_log_eps = attn.log_eps.detach().clone()
        # log_eps + 0.01 → softplus^{-1}(eps - 0.01)
        target_softplus = max(eps_value - 0.01, 1e-6)
        new_log = torch.log(torch.exp(torch.tensor(target_softplus)) - 1.0)
        with torch.no_grad():
            attn.log_eps.copy_(new_log)

    try:
        for n in iter_counts:
            attn.iters = n
            attn.n_grad_iters = n
            attn.n_nograd_iters = 0

            with torch.no_grad():
                P = attn._route_unbalanced_eot(C_mat, rho_flat)

                transport = (C_mat * P).sum(dim=(1, 2))  # (B,)

                P_row = P.sum(dim=2).clamp_min(1e-12)
                P_col = P.sum(dim=1).clamp_min(1e-12)

                a_unif = 1.0 / HW
                b_unif = 1.0 / C_mat.shape[-1]

                # KL(P1 || a) per pixel: u·log(u/v) - u + v
                kl_row_pp = P_row * torch.log(P_row / a_unif) - P_row + a_unif
                kl_row = (rho_flat * kl_row_pp).sum(dim=1)  # (B,)

                kl_col_pp = P_col * torch.log(P_col / b_unif) - P_col + b_unif
                kl_col = eps_val * kl_col_pp.sum(dim=1)  # (B,)

                primal = (transport + kl_row + kl_col).mean().item()

                primal_costs.append(primal)
                transport_costs.append(transport.mean().item())
                kl_row_costs.append(kl_row.mean().item())
                kl_col_costs.append(kl_col.mean().item())
    finally:
        attn.iters, attn.n_grad_iters, attn.n_nograd_iters = original
        if eps_value is not None:
            with torch.no_grad():
                attn.log_eps.copy_(original_log_eps)

    final = primal_costs[-1] if primal_costs else float("nan")
    return {
        "iter_counts": iter_counts,
        "primal_cost": primal_costs,
        "transport_cost": transport_costs,
        "kl_row": kl_row_costs,
        "kl_col": kl_col_costs,
        "relative_primal": [c / (final + 1e-12) for c in primal_costs],
        "eps_value": eps_val,
        "final_cost": final,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--max-iters", type=int, default=30)
    p.add_argument(
        "--eps-sweep",
        type=float,
        nargs="+",
        default=[0.1],
        help="Sweep multiple ε values for the convergence plot.",
    )
    p.add_argument("-o", "--output", type=str, default="sinkhorn_convergence.pdf")
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--cuda", action="store_true")
    p.add_argument(
        "--content-adaptive",
        action="store_true",
        dest="use_content_adaptive",
        help="Match checkpoint trained with ContentAdaptiveVSSBlock.",
    )
    p.add_argument("--cluster-num", type=int, default=8, dest="cluster_num")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = load_model(
        args.checkpoint,
        device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.max_iters,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
        update_for_inference=True,
    )
    _x, x_padded, _H, _W = load_image(args.image, device=device)

    all_results = {}
    for eps in args.eps_sweep:
        print(f"\n== ε = {eps} ==")
        res = measure_convergence(model, x_padded, args.max_iters, eps_value=eps)
        all_results[f"{eps:.4f}"] = res
        costs = res["primal_cost"]

        def _cost_at(n: int) -> str:
            return f"{costs[n - 1]:.6f}" if n <= len(costs) else "n/a"

        print(
            f"  cost @ iter 3:  {_cost_at(3)}\n"
            f"  cost @ iter 10: {_cost_at(10)}\n"
            f"  cost @ iter 20: {_cost_at(20)}\n"
            f"  cost @ iter {args.max_iters}: {res['final_cost']:.6f}"
        )

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    cmap = plt.get_cmap("viridis")

    for idx, (eps_label, res) in enumerate(all_results.items()):
        col = cmap(idx / max(len(all_results) - 1, 1))
        axes[0].plot(
            res["iter_counts"],
            res["primal_cost"],
            "-o",
            markersize=3,
            color=col,
            label=f"ε = {eps_label}",
        )
        axes[1].plot(
            res["iter_counts"],
            res["relative_primal"],
            "-o",
            markersize=3,
            color=col,
            label=f"ε = {eps_label}",
        )

    for ax in axes:
        ax.axvline(20, color="green", ls="--", alpha=0.7, label="chosen (20)")
        ax.set_xlabel("Sinkhorn iterations")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_ylabel("Primal objective")
    axes[0].set_title("Convergence of unbalanced Sinkhorn primal cost")
    axes[1].set_ylabel("Cost / cost at max iters")
    axes[1].set_title("Relative convergence")
    axes[1].axhline(1.0, color="gray", ls=":", alpha=0.5)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure → {args.output}")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved JSON → {args.json_out}")


if __name__ == "__main__":
    main()
