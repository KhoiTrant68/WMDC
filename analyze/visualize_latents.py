import argparse
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def main():
    parser = argparse.ArgumentParser(
        description="Visualise per-slice latent activation heatmaps"
    )
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "-o", "--output", type=str, default="latent_sparsity_visualization.pdf"
    )
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
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
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

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

    with torch.no_grad():
        out_enc = model.compress(x_padded)

        latents = []
        original_decompress = model.gaussian_conditional.decompress

        def patched_decompress(*args, **kwargs):
            y_hat_slice = original_decompress(*args, **kwargs)
            spatial_heatmap = (
                torch.mean(torch.abs(y_hat_slice), dim=1).squeeze(0).detach().cpu()
            )
            latents.append(spatial_heatmap)
            return y_hat_slice

        model.gaussian_conditional.decompress = patched_decompress
        try:
            _ = model.decompress(out_enc["strings"], out_enc["shape"])
        finally:
            model.gaussian_conditional.decompress = original_decompress

    assert len(latents) == 5, f"Expected 5 latent heatmaps, got {len(latents)}"

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    orig_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    latent_h = H // 16
    latent_w = W // 16

    for i in range(5):
        hm = latents[i].numpy()
        hm_cropped = hm[:latent_h, :latent_w]
        axes[i + 1].imshow(hm_cropped, cmap="magma", interpolation="nearest")
        axes[i + 1].set_title(f"Slice {i + 1} Latent")
        axes[i + 1].axis("off")

    fig.tight_layout()
    ext = os.path.splitext(args.output)[1].lstrip(".").lower()
    fmt = "pdf" if ext == "pdf" else ext if ext else "pdf"
    fig.savefig(args.output, format=fmt, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved latent visualisation to {args.output}")


if __name__ == "__main__":
    main()
