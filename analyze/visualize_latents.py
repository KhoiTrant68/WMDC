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
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="latent_sparsity_visualization.pdf",
        help="Output path.  Extension controls format (pdf recommended).",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = WMDC(N=192, M=320, num_slices=5, routing_mode=args.routing_mode).to(device)
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
            """
            Intercepts each gaussian_conditional.decompress() call inside
            model.decompress() to capture the mean-abs activation heatmap.

            y_hat_slice shape: (B, slice_ch, latent_H, latent_W)
            = (1, 64, padded_H//16, padded_W//16)

            We take channel-mean of |y_hat_slice| to get a single spatial
            heatmap per slice, then move it to CPU immediately to free VRAM.
            """
            y_hat_slice = original_decompress(*args, **kwargs)
            spatial_heatmap = (
                torch.mean(torch.abs(y_hat_slice), dim=1)  # (1, lH, lW)
                .squeeze(0)  # (lH, lW)
                .detach()
                .cpu()
            )
            latents.append(spatial_heatmap)
            return y_hat_slice

        model.gaussian_conditional.decompress = patched_decompress
        try:
            _ = model.decompress(out_enc["strings"], out_enc["shape"])
        finally:
            model.gaussian_conditional.decompress = original_decompress

    assert (
        len(latents) == 5
    ), f"Expected 5 latent heatmaps (one per slice), got {len(latents)}"

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    # Column 0: original unpadded image
    orig_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # The latent heatmap spatial size is (padded_H//16, padded_W//16).
    # We crop it to (H//16, W//16) to remove padding artefacts.
    # Since H % 64 == 0 (enforced by WMDC), H//16 == ceil(H/16). ✓
    latent_h = H // 16
    latent_w = W // 16

    for i in range(5):
        hm = latents[i].numpy()  # (padded_H//16, padded_W//16)
        hm_cropped = hm[:latent_h, :latent_w]  # (H//16, W//16) — remove pad

        axes[i + 1].imshow(hm_cropped, cmap="magma", interpolation="nearest")
        axes[i + 1].set_title(f"Slice {i + 1} Latent")
        axes[i + 1].axis("off")

    fig.tight_layout()

    # Infer format from extension; always call plt.close to free memory.
    ext = os.path.splitext(args.output)[1].lstrip(".").lower()
    fmt = "pdf" if ext == "pdf" else ext if ext else "pdf"
    fig.savefig(args.output, format=fmt, bbox_inches="tight")
    plt.close(fig)

    print(
        "Saved latent sparsity visualisation "
        "(quantised decoder-side reconstructions) to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()
