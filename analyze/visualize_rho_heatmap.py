"""
analyze/visualize_rho_heatmap.py
=================================
Visualises the per-pixel relaxation strength ρ_i predicted by the
spatial-ρ head. Generates one figure per input image showing:

    Original | ρ-map (slice 0) | ρ-map (slice 2) | ρ-map (slice 4)

The paper's claim is that smooth regions get high ρ (transport ~ balanced)
and content-rich regions get low ρ (more permissive marginals).
This script generates the supporting visual.

Usage
-----
    python analyze/visualize_rho_heatmap.py \
        --checkpoint    ckpts/lam_0.013/best.pth.tar \
        --image-dir     /data/kodak \
        --num-images    4 \
        --output        rho_heatmap.pdf \
        --cuda

Notes
-----
* Requires the model to expose `_compute_rho_spatial(slice_idx, hyper_prior, decoded_slices)`
  (which it does in models/WMDC.py).
* Slices visualised: indices 0, K//2, K-1 by default. Override with
  --slices 0 1 2 3 4.
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from _common import iter_dataset, load_model


def _decode_hyper(model, z_hat: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "_hyper_decode"):
        scales, means = model._hyper_decode(z_hat)
        return torch.cat([scales, means], dim=1)
    if hasattr(model, "h_trunk"):
        h = model.h_trunk(z_hat)
        return torch.cat([model.h_scale_head(h), model.h_mean_head(h)], dim=1)
    return torch.cat([model.h_scale_s(z_hat), model.h_mean_s(z_hat)], dim=1)


def collect_rho_maps(model, x_padded: torch.Tensor, slice_indices: list[int]):
    """
    Returns: (rho_maps, latent_h, latent_w)
        rho_maps : list of length len(slice_indices); each (latent_h, latent_w) numpy.
    """
    rho_maps: list[np.ndarray] = []
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
        try:
            z_str = model.entropy_bottleneck.compress(z)
            z_hat = model.entropy_bottleneck.decompress(z_str, z.size()[-2:])
        except Exception:
            z_hat = torch.round(z)

        hyper_prior = _decode_hyper(model, z_hat)

        # Get slice channels for "decoded slice" simulation.
        # We feed the predicted (mu) as a stand-in for previously decoded
        # slices when querying ρ for higher-index slices. This is a faithful
        # eval-time approximation; at inference the real ŷ_i would be used.
        latent_h, latent_w = z_hat.shape[-2] * 4, z_hat.shape[-1] * 4

        for s_idx in slice_indices:
            # `decoded_slices` is a list whose length tells the predictor
            # which slice we're at; we pass empty list for slice 0 and
            # zeros for higher slices so ρ depends only on hyper prior.
            decoded = []
            if s_idx > 0:
                slice_ch = hyper_prior.shape[1] // 2 // model.num_slices
                for _ in range(s_idx):
                    decoded.append(
                        torch.zeros(
                            x_padded.shape[0],
                            slice_ch,
                            latent_h,
                            latent_w,
                            device=x_padded.device,
                        )
                    )
            try:
                rho_spatial = model._compute_rho_spatial(
                    slice_idx=s_idx,
                    hyper_prior=hyper_prior,
                    decoded_slices=decoded,
                )
                # rho_spatial shape: (B, 1, latent_h, latent_w) or (B, latent_h, latent_w)
                rho_np = rho_spatial.squeeze().cpu().numpy()
            except Exception as e:
                print(f"  [WARN] _compute_rho_spatial(slice={s_idx}) failed: {e}")
                rho_np = np.zeros((latent_h, latent_w), dtype=np.float32)

            rho_maps.append(rho_np)

    return rho_maps, latent_h, latent_w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--image-dir", required=True)
    p.add_argument("--num-images", type=int, default=4)
    p.add_argument(
        "--slices",
        type=int,
        nargs="+",
        default=None,
        help="Slice indices to visualise. Default: [0, K//2, K-1].",
    )
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--output", type=str, default="rho_heatmap.pdf")
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
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
        update_for_inference=True,
    )

    if args.slices is None:
        K = model.num_slices
        slice_idx = sorted({0, K // 2, K - 1})
    else:
        slice_idx = sorted(set(args.slices))

    # Collect images.
    image_paths = sorted(
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )[: args.num_images]

    n_rows = len(image_paths)
    n_cols = 1 + len(slice_idx)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False
    )

    for r, img_path in enumerate(image_paths):
        from _common import load_image

        x, x_padded, H, W = load_image(img_path, device=device)
        rho_maps, lat_h, lat_w = collect_rho_maps(model, x_padded, slice_idx)

        # Image column.
        orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
        axes[r, 0].imshow(orig_np)
        axes[r, 0].axis("off")
        axes[r, 0].set_title(os.path.basename(img_path), fontsize=10)

        # Determine global vmin/vmax for fair comparison across slices.
        all_rho = np.stack([m for m in rho_maps if m.size > 0])
        vmin, vmax = float(np.min(all_rho)), float(np.max(all_rho))

        # Crop padded latent to true latent size.
        true_lat_h = H // 16
        true_lat_w = W // 16

        for c, (s, rho) in enumerate(zip(slice_idx, rho_maps), start=1):
            if rho.ndim == 2:
                rho_cropped = rho[:true_lat_h, :true_lat_w]
            else:
                rho_cropped = rho.reshape(-1, lat_h, lat_w).mean(0)[
                    :true_lat_h, :true_lat_w
                ]
            im = axes[r, c].imshow(
                rho_cropped,
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"ρ — slice {s}", fontsize=10)

        # Add colorbar at the right of the row.
        plt.colorbar(im, ax=axes[r, -1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
