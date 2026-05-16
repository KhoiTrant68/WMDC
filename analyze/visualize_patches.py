import argparse
import math
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate visual comparison patches for WMDC"
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("-i", "--image", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="visual_comparison")
    parser.add_argument(
        "--crop",
        type=int,
        nargs=4,
        default=[300, 450, 250, 400],
        metavar=("Y1", "Y2", "X1", "X2"),
        help="Crop region [y1 y2 x1 x2] in pixel coordinates",
    )
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--use-actual-bpp",
        action="store_true",
        help="Use actual entropy-coded BPP (slower). Recommended for paper figures.",
    )
    parser.add_argument("--cuda", action="store_true")
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = WMDC(
        N=args.N,
        M=args.M,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
        use_wls_shortcut=args.use_wls_shortcut,
    ).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    model.update(force=True)

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    H, W = x.size(2), x.size(3)
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    num_pixels = H * W

    with torch.no_grad():
        if args.use_actual_bpp:
            out_enc = model.compress(x_padded)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            x_hat_padded = out_dec["x_hat"]

            def _get_size(obj):
                if isinstance(obj, bytes):
                    return len(obj)
                if isinstance(obj, (list, tuple)):
                    return sum(_get_size(s) for s in obj)
                return 0

            bpp_value = (_get_size(out_enc["strings"]) * 8) / num_pixels
            bpp_label = "BPP (actual)"
        else:
            out_net = model(x_padded)
            x_hat_padded = out_net["x_hat"]
            bpp_value = sum(
                torch.log(lh).sum() for lh in out_net["likelihoods"].values()
            ).item() / (-math.log(2) * num_pixels)
            bpp_label = "BPP (est.)"

        x_hat = x_hat_padded[:, :, :H, :W].clamp(0, 1)
        mse = F.mse_loss(x, x_hat)
        psnr = -10 * math.log10(mse.item()) if mse.item() > 0 else 100.0

    orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    rec_np = x_hat.squeeze().permute(1, 2, 0).cpu().numpy()

    y1, y2, x1, x2 = args.crop
    if y2 > orig_np.shape[0] or x2 > orig_np.shape[1] or y1 >= y2 or x1 >= x2:
        print(
            f"Warning: crop [{y1}:{y2}, {x1}:{x2}] invalid for {orig_np.shape[:2]}. Using full image."
        )
        y1, y2, x1, x2 = 0, orig_np.shape[0], 0, orig_np.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(orig_np[y1:y2, x1:x2])
    axes[0].set_title("Original Image (Crop)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(rec_np[y1:y2, x1:x2])
    axes[1].set_title(
        f"WMDC (Ours)\n{bpp_label}: {bpp_value:.3f} | PSNR: {psnr:.2f} dB",
        fontsize=14,
    )
    axes[1].axis("off")

    fig.tight_layout()
    pdf_path = f"{args.output}.pdf"
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
