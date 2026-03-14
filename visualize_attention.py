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


def main():
    parser = argparse.ArgumentParser(description="Visualize HDDA Maps (CVPR Ready)")
    parser.add_argument("-d", "--img_dir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="hdda_attention_maps.pdf")
    parser.add_argument("--slice", type=int, default=0, help="Latent slice (0-4)")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Format: Target Dictionary Token Indices [0 to 127]
    # EOT has no "Heads" like standard Multi-head Attention
    target_maps = [0, 32, 64, 127]

    # Initialize new FD-SSM + HDDA Model
    model = WMDC(N=192, M=320, num_slices=5).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    img_files = sorted(
        [
            f
            for f in os.listdir(args.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
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

        with torch.no_grad():
            _ = model(x)

        # 1. Grab attention: Shape (1, HW, 128) -> Squeeze batch -> (HW, 128)
        probs = model.eot_attention[args.slice].attn_probs.squeeze(0).cpu()

        # 2. Reshape to spatial grid
        latent_h, latent_w = x.size(2) // 16, x.size(3) // 16
        probs = probs.view(latent_h, latent_w, probs.size(-1))  # (H, W, 128)

        # Original image for overlay
        orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()

        # --- PLOT INPUT IMAGE ---
        ax_img = axes[row_idx][0]
        ax_img.imshow(orig_np)
        ax_img.axis("off")
        if row_idx == 0:
            ax_img.set_title("Input Image", fontsize=16, pad=10)

        # --- PLOT ATTENTION OVERLAYS ---
        for col_idx, dict_idx in enumerate(target_maps):

            # Extract 2D heatmap:[1, 1, latent_h, latent_w]
            attn_map = probs[:, :, dict_idx].unsqueeze(0).unsqueeze(0)

            # This avoids sub-pixel misalignment caused by integer division (`p // 16`).
            attn_resized = F.interpolate(
                attn_map,
                size=(x.size(2), x.size(3)),
                mode="bicubic",
                align_corners=False,
            )
            attn_cropped = attn_resized
            attn_np = attn_cropped.squeeze().numpy()

            # Normalize for visualization [0, 1]
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

            ax_map = axes[row_idx][col_idx + 1]

            # Alpha Blending
            ax_map.imshow(orig_np)  # Background
            ax_map.imshow(attn_np, cmap="jet", alpha=0.55)  # Heatmap overlay

            ax_map.axis("off")

            if row_idx == 0:
                ax_map.set_title(f"Dict Token {dict_idx}", fontsize=14, pad=10)

    plt.savefig(args.output, format="pdf", bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved highly-interpretable attention maps to {args.output}")


if __name__ == "__main__":
    main()
