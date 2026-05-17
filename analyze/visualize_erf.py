"""
Per-image Effective Receptive Field (ERF) visualization for WMDC.

Reproduces Figure-16-style panels: top row = input image, bottom row = ERF
heatmap overlay, one column per image. ERF is computed by backpropagating
from the center patch of the encoder output y = g_a(x) to the input image,
then taking |grad| averaged over channels and log-normalized.

Example
-------
    python analyze/visualize_erf.py \
        -d /path/to/kodak \
        -c ckpt_wmdc.pth.tar \
        --image_names kodim17.png kodim21.png kodim19.png kodim18.png kodim06.png \
        -o erf_wmdc.pdf --cuda
"""

import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def build_model(args, device):
    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
        use_wls_shortcut=args.use_wls_shortcut,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt), strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def compute_erf(model, x, center_radius=1):
    """|grad of center-patch sum w.r.t. x|, averaged over channels → (H, W)."""
    x = x.clone().detach().requires_grad_(True)
    y = model.g_a(x)
    _, _, hy, wy = y.shape
    cy, cx = hy // 2, wy // 2
    r = center_radius
    target = y[:, :, cy - r : cy + r + 1, cx - r : cx + r + 1].sum()
    grad = torch.autograd.grad(target, x)[0]
    return grad.abs().mean(dim=1).squeeze(0).detach().cpu().numpy()


def normalize_erf(erf, log=True, lo_pct=1.0, hi_pct=99.5):
    if log:
        erf = np.log1p(erf)
    lo, hi = np.percentile(erf, lo_pct), np.percentile(erf, hi_pct)
    return np.clip((erf - lo) / max(hi - lo, 1e-12), 0.0, 1.0)


def pad_to_multiple(x, m=64):
    H, W = x.shape[2:]
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), H, W


def main():
    parser = argparse.ArgumentParser(description="Per-image ERF — WMDC")
    parser.add_argument("-d", "--img_dir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="erf_wmdc.pdf")
    parser.add_argument(
        "--num_images", type=int, default=5, help="Columns (ignored if --image_names)."
    )
    parser.add_argument(
        "--image_names",
        nargs="*",
        default=None,
        help="Specific filenames to plot, in order.",
    )
    parser.add_argument(
        "--center_radius",
        type=int,
        default=1,
        help="Half-size of the latent center patch to backprop from.",
    )
    parser.add_argument("--cmap", type=str, default="viridis")
    parser.add_argument("--alpha", type=float, default=0.6)
    parser.add_argument(
        "--layout",
        type=str,
        default="two_rows",
        choices=["two_rows", "overlay_only", "heatmap_only"],
        help=(
            "two_rows: row1=image, row2=ERF overlay. "
            "overlay_only: single row, ERF overlaid on image. "
            "heatmap_only: single row, ERF heatmap with no image."
        ),
    )
    parser.add_argument(
        "--no_log", action="store_true", help="Disable log1p normalization."
    )
    parser.add_argument(
        "--show_titles", action="store_true", help="Print image filename as title."
    )

    # Model config — must match training of the checkpoint
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--content-adaptive",
        action="store_true",
        default=False,
        dest="use_content_adaptive",
    )
    parser.add_argument("--cluster-num", type=int, default=8, dest="cluster_num")
    parser.add_argument(
        "--use-wls-shortcut",
        action="store_true",
        default=False,
        dest="use_wls_shortcut",
    )
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if args.image_names:
        img_files = args.image_names
    else:
        img_files = sorted(
            f
            for f in os.listdir(args.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )[: args.num_images]
    if not img_files:
        raise RuntimeError("No images found")

    print(f"Loading WMDC from {args.checkpoint}")
    model = build_model(args, device)
    to_tensor = transforms.ToTensor()

    num_cols = len(img_files)
    num_rows = 2 if args.layout == "two_rows" else 1
    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(2.6 * num_cols, 2.6 * num_rows),
        squeeze=False,
    )
    plt.subplots_adjust(wspace=0.04, hspace=0.06)

    for col, img_name in enumerate(img_files):
        print(f"  - {img_name}")
        img = Image.open(os.path.join(args.img_dir, img_name)).convert("RGB")
        x = to_tensor(img).unsqueeze(0).to(device)
        x_pad, H, W = pad_to_multiple(x, m=64)

        erf = compute_erf(model, x_pad, center_radius=args.center_radius)[:H, :W]
        erf_norm = normalize_erf(erf, log=not args.no_log)
        orig_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()

        if args.layout == "two_rows":
            ax_top, ax_bot = axes[0][col], axes[1][col]
            ax_top.imshow(orig_np)
            ax_bot.imshow(orig_np)
            ax_bot.imshow(erf_norm, cmap=args.cmap, alpha=args.alpha, vmin=0, vmax=1)
            cell_axes = [ax_top, ax_bot]
            if args.show_titles:
                ax_top.set_title(os.path.splitext(img_name)[0], fontsize=10, pad=2)
            if col == 0:
                ax_top.set_ylabel("Input", fontsize=11, labelpad=6)
                ax_bot.set_ylabel("ERF", fontsize=11, labelpad=6)
        elif args.layout == "overlay_only":
            ax = axes[0][col]
            ax.imshow(orig_np)
            ax.imshow(erf_norm, cmap=args.cmap, alpha=args.alpha, vmin=0, vmax=1)
            cell_axes = [ax]
            if args.show_titles:
                ax.set_title(os.path.splitext(img_name)[0], fontsize=10, pad=2)
        else:  # heatmap_only
            ax = axes[0][col]
            ax.imshow(erf_norm, cmap=args.cmap, vmin=0, vmax=1)
            cell_axes = [ax]
            if args.show_titles:
                ax.set_title(os.path.splitext(img_name)[0], fontsize=10, pad=2)

        for ax in cell_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
