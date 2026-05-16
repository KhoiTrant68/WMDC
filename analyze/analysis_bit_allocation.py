"""
analyze/analysis_bit_allocation.py — fixed version
=====================================================
Changes: --sinkhorn-iters arg added; model construction consistent with eval.py.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

from models.WMDC import WMDC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="bit_allocation_plot.pdf")
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--content-adaptive",
        action="store_true",
        default=False,
        dest="use_content_adaptive",
        help="Use content-adaptive K-means token permutation in Mamba blocks.",
    )
    parser.add_argument(
        "--cluster-num",
        type=int,
        default=8,
        dest="cluster_num",
        help="Number of clusters for content-adaptive K-means tokenization.",
    )
    parser.add_argument(
        "--use-wls-shortcut",
        action="store_true",
        default=False,
        dest="use_wls_shortcut",
        help="Enable WLS/iWLS multi-scale shortcuts (must match checkpoint).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    model.update(force=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    H, W = x.size(2), x.size(3)
    num_pixels = H * W

    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out_enc = model.compress(x_padded)

    y_strings = out_enc["strings"][0]
    z_strings = out_enc["strings"][1]

    slice_bpps = [(len(s[0]) * 8) / num_pixels for s in y_strings]
    z_bpp = (len(z_strings[0]) * 8) / num_pixels
    total_bpp = sum(slice_bpps) + z_bpp

    print(f"Total BPP:           {total_bpp:.4f}")
    print(f"Hyperprior (Z) BPP:  {z_bpp:.4f} ({z_bpp / total_bpp * 100:.1f}%)")
    for i, bpp in enumerate(slice_bpps):
        print(f"Slice {i + 1} BPP:        {bpp:.4f} ({bpp / total_bpp * 100:.1f}%)")

    labels = ["Z (Prior)", "Slice 1", "Slice 2", "Slice 3", "Slice 4", "Slice 5"]
    bpps = [z_bpp] + slice_bpps

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        labels,
        bpps,
        color=["gray", "darkred", "red", "orange", "lightcoral", "mistyrose"],
    )
    ax.set_ylabel("Bits Per Pixel (BPP)")
    ax.set_title("Bit Allocation Across Autoregressive Slices")

    for bar, bpp in zip(bars, bpps):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.001 * total_bpp,
            f"{bpp / total_bpp * 100:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
