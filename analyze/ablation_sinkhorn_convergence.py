import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def measure_convergence(model, x_padded, max_iters: int, device: str):
    """
    Run the EOT routing with iter counts [1, 2, ..., max_iters] and record
    the primal transport cost Σ_{ij} C_{ij} π_{ij} at each count.

    Returns:
        iter_counts  : list of int
        costs        : list of float  (mean over batch and spatial positions)
    """
    model.eval()

    # Extract the components needed to build a cost matrix.
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
        z_hat = model.entropy_bottleneck.decompress(
            model.entropy_bottleneck.compress(z), z.size()[-2:]
        )
        dt = model.hyper_to_dict(z_hat)
        latent_scales = model.h_scale_s(z_hat)
        latent_means = model.h_mean_s(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)
        rho_spatial = model._compute_rho_spatial(
            slice_idx=0, hyper_prior=hyper_prior, decoded_slices=[]
        )

        # Use slice-0 projection for the convergence test.
        k_dict = model.k_projs[0](dt)
        v_dict = model.v_projs[0](dt)

        memory_state = model.init_memory(hyper_prior)
        query = torch.cat([hyper_prior, memory_state], dim=1)

        B, _, H, W = query.shape
        HW = H * W

        # Build cost matrix once (independent of iters).
        C_mat = model.eot_attention._cost_matrix(query, k_dict, H, W)
        rho_flat = rho_spatial.view(B, HW).clamp(min=0.01)

    iter_counts = list(range(1, max_iters + 1))
    costs = []

    # Temporarily swap iters inside the module.
    original_iters = model.eot_attention.iters

    for n_iters in iter_counts:
        model.eot_attention.iters = n_iters
        with torch.no_grad():
            P = model.eot_attention._route_unbalanced_eot(C_mat, rho_flat)

            # 1. Linear Transport Cost
            transport_cost = (C_mat * P).sum(dim=(1, 2))

            # 2. Marginal KL Divergences (Penalty for creating/destroying mass)
            # P_row_sum is P1, P_col_sum is P^T1
            P_row_sum = P.sum(dim=2) + 1e-8  # (B, HW)
            P_col_sum = P.sum(dim=1) + 1e-8  # (B, N)

            alpha = 1.0 / HW
            beta = 1.0 / C_mat.size(-1)

            # D_KL(u || v) = u log(u/v) - u + v
            kl_row = (P_row_sum * torch.log(P_row_sum / alpha) - P_row_sum + alpha).sum(
                dim=1
            )
            kl_col = (P_col_sum * torch.log(P_col_sum / beta) - P_col_sum + beta).sum(
                dim=1
            )

            # Spatial Rho applies to rows, uniform epsilon to cols (as per your derivation)
            rho_penalty = (rho_flat * kl_row).sum(dim=1)  # (B)

            # Total Unbalanced Primal Objective
            total_cost = (
                (transport_cost + rho_penalty + model.eot_attention.ot_eps * kl_col)
                .mean()
                .item()
            )

        costs.append(total_cost)

    model.eot_attention.iters = original_iters
    return iter_counts, costs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("-o", "--output", type=str, default="sinkhorn_convergence.pdf")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.max_iters,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.update(force=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    H, W = x.size(2), x.size(3)
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    iter_counts, costs = measure_convergence(model, x_padded, args.max_iters, device)

    # Cost relative to max-iter value (convergence fraction).
    final_cost = costs[-1]
    rel_costs = [c / (final_cost + 1e-8) for c in costs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(iter_counts, costs, "b-o", markersize=3)
    ax1.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Original (3)")
    ax1.axvline(x=20, color="green", linestyle="--", alpha=0.7, label="Fixed (20)")
    ax1.set_xlabel("Sinkhorn iterations")
    ax1.set_ylabel("Primal transport cost")
    ax1.set_title("Convergence of transport cost")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(iter_counts, rel_costs, "b-o", markersize=3)
    ax2.axvline(x=3, color="red", linestyle="--", alpha=0.7, label="Original (3)")
    ax2.axvline(x=20, color="green", linestyle="--", alpha=0.7, label="Fixed (20)")
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax2.set_xlabel("Sinkhorn iterations")
    ax2.set_ylabel("Cost / cost at max iters")
    ax2.set_title("Relative convergence")
    ax2.legend()
    ax2.grid(alpha=0.3)

    cost_at_3 = costs[2] if len(costs) > 2 else float("nan")
    cost_at_20 = costs[19] if len(costs) > 19 else float("nan")
    print(f"Transport cost at  3 iters: {cost_at_3:.6f}")
    print(f"Transport cost at 20 iters: {cost_at_20:.6f}")
    print(
        f"Relative gap (3 vs 20):     {abs(cost_at_3 - cost_at_20) / (cost_at_20 + 1e-8) * 100:.2f}%"
    )

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
