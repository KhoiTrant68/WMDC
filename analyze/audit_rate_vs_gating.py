"""
analyze/audit_rate_vs_gating.py
================================
Audits whether UEOT row-mass gating correlates with image complexity
(legitimate behaviour) or with bpp_y leakage (the reviewer concern that
the model "drops mass" at hard regions while pushing real information
through the Gaussian channel).

Method
------
For every latent position (i, j) in (H/16, W/16):
    row_mass(i,j)    = mean over slices of last_row_mass[s][i,j]
    bpp_y(i,j)       = sum over M=320 channels of -log2(y_likelihoods[c,i,j])
                       / 16² (per-input-pixel bpp contribution)
    complexity(i,j)  = mean variance of input pixels in the 16×16 block

Output
------
- Scatter: row_mass vs complexity (gating tracks complexity?)
- Scatter: row_mass vs bpp_y    (low gating → low or high bitrate?)
- Quintile bar chart: bpp_y / complexity per row_mass quintile
- JSON summary with Pearson correlations

Defence narrative
-----------------
Expected (no leakage): row_mass ↑ ↔ complexity ↑ ↔ bpp_y ↑
Leakage signature:     row_mass ↓ but bpp_y ↑ (model gates dictionary off
                       at complex regions, pushing the bitrate to Gaussian).

Usage
-----
    python analyze/audit_rate_vs_gating.py \
        --checkpoint  ckpts/lam_0.013/best.pth.tar \
        --dataset     /data/kodak \
        --output-dir  results/audit_gating \
        --cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from _common import iter_dataset, load_model


def _local_variance(x: torch.Tensor, block: int = 16) -> torch.Tensor:
    """
    Mean per-channel variance of x within non-overlapping `block`×`block` blocks.

    x : (B, 3, H, W) in [0, 1]
    Returns : (B, H/block, W/block) — one scalar per latent position.
    """
    B, C, H, W = x.shape
    # E[x²] − (E[x])² inside each block, then average across channels.
    sq_mean = F.avg_pool2d(x.pow(2), kernel_size=block, stride=block)
    mean = F.avg_pool2d(x, kernel_size=block, stride=block)
    var = (sq_mean - mean.pow(2)).clamp(min=0.0).mean(dim=1)  # (B, H/16, W/16)
    return var


def _collect_per_image(
    model, x: torch.Tensor, x_padded: torch.Tensor, H: int, W: int
) -> dict:
    """
    Forward x_padded once, collect three (latent_h, latent_w) maps:
        row_mass    : mean slice-wise spatial gating  (None if not unbalanced_eot)
        bpp_y       : per-position bitrate (in bpp of the original image)
        complexity  : input-pixel local variance
    """
    with torch.no_grad():
        out = model(x_padded)

    y_lh = out["likelihoods"]["y"]  # (1, M, h_lat, w_lat)
    h_lat, w_lat = y_lh.shape[-2:]
    true_h_lat, true_w_lat = H // 16, W // 16
    # Crop to original (unpadded) latent footprint for fair stats.
    y_lh_crop = y_lh[..., :true_h_lat, :true_w_lat]

    # bpp_y per latent position, normalised by the 16² input pixels it covers
    # so the scalar represents this position's contribution to the *image* bpp.
    log2_inv = 1.0 / math.log(2.0)
    bits_per_pos = -y_lh_crop.clamp(min=1e-12).log().sum(dim=1) * log2_inv  # (1, h, w)
    bpp_per_pos = bits_per_pos.squeeze(0) / (16 * 16)  # bpp contribution

    # Row mass: only present for unbalanced_eot. Average across slices to get
    # a single scalar per latent position.
    row_masses = []
    for attn in model.eot_attentions:
        rm = getattr(attn, "last_row_mass", None)
        if rm is None:
            continue
        # rm : (1, HW_padded). Reshape to padded latent grid then crop.
        rm_grid = rm.view(1, h_lat, w_lat)[..., :true_h_lat, :true_w_lat]
        row_masses.append(rm_grid.squeeze(0))

    if row_masses:
        rm_mean = torch.stack(row_masses, dim=0).mean(dim=0)  # (h, w)
    else:
        rm_mean = None

    complexity = _local_variance(x, block=16).squeeze(0)  # (h, w) at true latent res

    # Sanity: shapes must match before flattening.
    assert (
        complexity.shape == bpp_per_pos.shape
    ), f"shape mismatch complexity={complexity.shape} bpp={bpp_per_pos.shape}"
    if rm_mean is not None:
        assert rm_mean.shape == bpp_per_pos.shape

    return {
        "row_mass": None if rm_mean is None else rm_mean.cpu().numpy().ravel(),
        "bpp_y": bpp_per_pos.cpu().numpy().ravel(),
        "complexity": complexity.cpu().numpy().ravel(),
    }


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size < 2:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    denom = math.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


def _quintile_stats(
    row_mass: np.ndarray, bpp_y: np.ndarray, complexity: np.ndarray, n_bins: int = 5
) -> list[dict]:
    """Bin positions by row_mass quintile; report mean bpp / complexity per bin."""
    edges = np.quantile(row_mass, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf  # extend to catch all
    stats = []
    for k in range(n_bins):
        mask = (row_mass >= edges[k]) & (row_mass < edges[k + 1])
        if not mask.any():
            stats.append(
                {
                    "bin": k,
                    "n": 0,
                    "row_mass_mean": 0.0,
                    "bpp_y_mean": 0.0,
                    "complexity_mean": 0.0,
                }
            )
            continue
        stats.append(
            {
                "bin": k,
                "n": int(mask.sum()),
                "row_mass_mean": float(row_mass[mask].mean()),
                "bpp_y_mean": float(bpp_y[mask].mean()),
                "complexity_mean": float(complexity[mask].mean()),
            }
        )
    return stats


def _plot(
    row_mass: np.ndarray,
    bpp_y: np.ndarray,
    complexity: np.ndarray,
    quint: list[dict],
    correlations: dict,
    out_path: str,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Subsample for scatter readability (large N otherwise).
    n = row_mass.size
    if n > 20000:
        idx = np.random.default_rng(0).choice(n, 20000, replace=False)
        rm_s, bp_s, cp_s = row_mass[idx], bpp_y[idx], complexity[idx]
    else:
        rm_s, bp_s, cp_s = row_mass, bpp_y, complexity

    # 1. row_mass vs complexity
    ax = axes[0]
    ax.scatter(rm_s, cp_s, s=2, alpha=0.25, color="#1f77b4")
    ax.set_xlabel("row_mass (UEOT gating)")
    ax.set_ylabel("local variance (image complexity)")
    ax.set_title(
        f"Gating vs Complexity\nρ = {correlations['row_mass_vs_complexity']:.3f}"
    )
    ax.grid(alpha=0.3)

    # 2. row_mass vs bpp_y
    ax = axes[1]
    ax.scatter(rm_s, bp_s, s=2, alpha=0.25, color="#d62728")
    ax.set_xlabel("row_mass (UEOT gating)")
    ax.set_ylabel("bpp_y (per position)")
    ax.set_title(f"Gating vs Bitrate\nρ = {correlations['row_mass_vs_bpp_y']:.3f}")
    ax.grid(alpha=0.3)

    # 3. Quintile bar chart
    ax = axes[2]
    bins = np.arange(len(quint))
    width = 0.35
    bpp_means = [q["bpp_y_mean"] for q in quint]
    cpx_means = [q["complexity_mean"] for q in quint]

    # Normalise both to [0, 1] for shared axis.
    def _norm(arr):
        a = np.asarray(arr, dtype=np.float64)
        return (a - a.min()) / (a.max() - a.min() + 1e-12)

    ax.bar(
        bins - width / 2, _norm(bpp_means), width, label="bpp_y (norm)", color="#d62728"
    )
    ax.bar(
        bins + width / 2,
        _norm(cpx_means),
        width,
        label="complexity (norm)",
        color="#1f77b4",
    )
    ax.set_xticks(bins)
    ax.set_xticklabels(
        [
            f"Q{i+1}\n(low→high\nrow_mass)" if i in (0, len(bins) - 1) else f"Q{i+1}"
            for i in bins
        ]
    )
    ax.set_ylabel("normalised mean")
    ax.set_title("Quintiles by row_mass")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        "UEOT gating audit — does row_mass track complexity (legit) or invert bpp_y (leakage)?",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--output-dir", type=str, default="results/audit_gating")
    p.add_argument("--cuda", action="store_true")
    p.add_argument(
        "--content-adaptive",
        action="store_true",
        default=False,
        dest="use_content_adaptive",
        help="Use content-adaptive K-means token permutation in Mamba blocks.",
    )
    p.add_argument(
        "--cluster-num",
        type=int,
        default=8,
        dest="cluster_num",
        help="Number of clusters for content-adaptive K-means tokenization.",
    )
    p.add_argument(
        "--use-wls-shortcut",
        action="store_true",
        default=False,
        dest="use_wls_shortcut",
        help="Enable WLS/iWLS multi-scale shortcuts (must match checkpoint).",
    )
    args = p.parse_args()

    if args.routing_mode != "unbalanced_eot":
        raise SystemExit(
            f"This audit only applies to routing_mode=unbalanced_eot "
            f"(got '{args.routing_mode}'). The whole concept of row-mass "
            "gating exists only in that mode."
        )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    model = load_model(
        args.checkpoint,
        device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
        use_wls_shortcut=args.use_wls_shortcut,
        update_for_inference=False,  # forward() only
    )

    # Required so eot_attentions populate last_row_mass during eval forward().
    for a in model.eot_attentions:
        a.store_attn_probs = True

    rm_all, bpp_all, cpx_all, per_image = [], [], [], []
    for path, x, x_padded, H, W in iter_dataset(args.dataset, device=device):
        rec = _collect_per_image(model, x, x_padded, H, W)
        if rec["row_mass"] is None:
            print(f"[skip] {os.path.basename(path)}: last_row_mass not populated")
            continue
        rm_all.append(rec["row_mass"])
        bpp_all.append(rec["bpp_y"])
        cpx_all.append(rec["complexity"])
        per_image.append(
            {
                "image": os.path.basename(path),
                "n_positions": int(rec["row_mass"].size),
                "row_mass_vs_complexity": _pearson(rec["row_mass"], rec["complexity"]),
                "row_mass_vs_bpp_y": _pearson(rec["row_mass"], rec["bpp_y"]),
            }
        )
        print(
            f"  {os.path.basename(path)}: "
            f"n={rec['row_mass'].size}  "
            f"ρ(rm,cpx)={per_image[-1]['row_mass_vs_complexity']:+.3f}  "
            f"ρ(rm,bpp)={per_image[-1]['row_mass_vs_bpp_y']:+.3f}"
        )

    if not rm_all:
        raise SystemExit("No usable images — aborting.")

    rm = np.concatenate(rm_all)
    bp = np.concatenate(bpp_all)
    cp = np.concatenate(cpx_all)

    correlations = {
        "row_mass_vs_complexity": _pearson(rm, cp),
        "row_mass_vs_bpp_y": _pearson(rm, bp),
        "complexity_vs_bpp_y": _pearson(cp, bp),
    }
    quintiles = _quintile_stats(rm, bp, cp, n_bins=5)

    summary = {
        "checkpoint": args.checkpoint,
        "dataset": args.dataset,
        "n_images": len(rm_all),
        "n_latent_positions_total": int(rm.size),
        "global_correlations": correlations,
        "quintiles_by_row_mass": quintiles,
        "per_image": per_image,
        "interpretation": {
            "no_leakage_signature": (
                "row_mass_vs_complexity > 0 AND row_mass_vs_bpp_y has same sign as "
                "complexity_vs_bpp_y. Gating tracks scene complexity; bitrate "
                "follows complexity, not adversarial gating."
            ),
            "leakage_signature": (
                "row_mass_vs_complexity < 0 AND row_mass_vs_bpp_y < 0. The model "
                "drops dictionary mass at complex regions, pushing the bitrate "
                "into the Gaussian channel — i.e. UEOT is being abused to launder "
                "rate."
            ),
        },
    }

    json_path = os.path.join(args.output_dir, "audit_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    pdf_path = os.path.join(args.output_dir, "audit_gating.pdf")
    _plot(rm, bp, cp, quintiles, correlations, pdf_path)

    print()
    print("=== Global correlations (Pearson) ===")
    for k, v in correlations.items():
        print(f"  {k:30s} {v:+.4f}")
    print()
    print("=== Quintiles by row_mass (low→high) ===")
    print(f"{'bin':>4} {'n':>8} {'row_mass':>10} {'bpp_y':>10} {'complexity':>12}")
    for q in quintiles:
        print(
            f"{q['bin']:>4} {q['n']:>8} {q['row_mass_mean']:>10.4f} "
            f"{q['bpp_y_mean']:>10.4f} {q['complexity_mean']:>12.6f}"
        )
    print()
    print(f"[done] summary → {json_path}")
    print(f"[done] figure  → {pdf_path}")


if __name__ == "__main__":
    main()
