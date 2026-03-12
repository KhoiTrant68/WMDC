import argparse
import os

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from models.WMDC import WMDC


def pad_image(x, p=64):
    """Padding fixed to exactly 64. g_a (16x) * h_a (4x) = 64x."""
    h, w = x.size(2), x.size(3)
    pad_h = (p - h % p) % p
    pad_w = (p - w % p) % p
    if pad_w == 0 and pad_h == 0:
        return x, (0, 0, 0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (0, pad_w, 0, pad_h)


def crop_image(x, padding):
    """Crop the padded regions for accurate visualization overlay."""
    _, pad_w, _, pad_h = padding
    if pad_w == 0 and pad_h == 0:
        return x
    return x[..., : x.size(2) - pad_h, : x.size(3) - pad_w]


def main():
    parser = argparse.ArgumentParser(description="Visualize HDDA Maps (CVPR Ready)")
    parser.add_argument("-d", "--img_dir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="hdda_attention_maps.pdf")
    parser.add_argument("--slice", type=int, default=0, help="Latent slice (0-4)")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Format: (Dictionary Entry Index[0-127], Head Index [0-19])
    # Pick a mix of early and late heads/tokens to show diversity
    target_maps = [
        (0, 0),
        (32, 2),
        (64, 5),
        (127, 19),
    ]

    # Initialize new FD-SSM + HDDA Model
    model = WMDC(N=192, M=320, num_slices=5).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    img_files = sorted(
        [f for f in os.listdir(args.img_dir) if f.lower().endswith((".png", ".jpg"))]
    )[
        :4
    ]  # Process top 4 images

    num_images = len(img_files)
    num_cols = len(target_maps) + 1

    fig, axes = plt.subplots(
        nrows=num_images, ncols=num_cols, figsize=(4 * num_cols, 3 * num_images)
    )
    if num_images == 1:
        axes = [axes]

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    for row_idx, img_name in enumerate(img_files):
        print(f"Processing {img_name}...")
        img_path = os.path.join(args.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)

        # Pad securely to 64x
        x_pad, padding = pad_image(x, p=64)

        with torch.no_grad():
            _ = model(x_pad)

        # 1. Grab attention from our class attribute (B, Head, HW, Dict_Num)
        # Assuming batch size 1, squeeze batch dim -> (Head, HW, 128)
        probs = model.dt_cross_attention[args.slice].attn_probs.squeeze(0).cpu()

        # 2. Reshape to spatial grid: y is strictly 1/16th of x_pad
        latent_h, latent_w = x_pad.size(2) // 16, x_pad.size(3) // 16
        probs = probs.view(probs.size(0), latent_h, latent_w, probs.size(-1))

        # Original image for overlay (cropped back to original size)
        orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
        h_orig, w_orig = orig_np.shape[0], orig_np.shape[1]

        # --- PLOT INPUT IMAGE ---
        ax_img = axes[row_idx][0]
        ax_img.imshow(orig_np)
        ax_img.axis("off")
        if row_idx == 0:
            ax_img.set_title("Input Image", fontsize=16, pad=10)

        # --- PLOT ATTENTION OVERLAYS ---
        for col_idx, (dict_idx, head_idx) in enumerate(target_maps):

            # Extract 2D heatmap: [latent_h, latent_w]
            attn_map = probs[head_idx, :, :, dict_idx].unsqueeze(0).unsqueeze(0)

            # Crop the padded area out of the attention map conceptually,
            # then interpolate to original resolution for smooth overlay.
            attn_cropped = crop_image(attn_map, [p // 16 for p in padding])
            attn_resized = F.interpolate(
                attn_cropped, size=(h_orig, w_orig), mode="bicubic", align_corners=False
            )
            attn_np = attn_resized.squeeze().numpy()

            # Normalize for visualization
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

            ax_map = axes[row_idx][col_idx + 1]

            # Alpha Blending
            ax_map.imshow(orig_np)  # Background
            im = ax_map.imshow(attn_np, cmap="jet", alpha=0.5)  # Heatmap overlay

            ax_map.axis("off")

            if row_idx == 0:
                ax_map.set_title(
                    f"Dict Token {dict_idx} | Head {head_idx}", fontsize=14, pad=10
                )

    plt.savefig(args.output, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved highly-interpretable attention maps to {args.output}")


if __name__ == "__main__":
    main()
