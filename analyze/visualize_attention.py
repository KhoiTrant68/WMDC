import argparse
import os
import sys

# Ensure the parent directory is in the path to import models
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


def main():
    parser = argparse.ArgumentParser(
        description="Visualize EOT Attention Maps (Scientifically Rigorous)"
    )
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("-d", "--img_dir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="hdda_attention_maps.pdf")

    # Visualization Modes
    parser.add_argument(
        "--mode",
        type=str,
        default="top_tokens",
        choices=["top_tokens", "slice_evolution"],
        help="'top_tokens' shows the 4 most active tokens for a single slice. "
        "'slice_evolution' tracks a single token across all 5 slices.",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=0,
        help="Which autoregressive slice to visualise (0-4). Used only in 'top_tokens' mode.",
    )
    parser.add_argument(
        "--target_token",
        type=int,
        default=0,
        help="Specific token to track. Used only in 'slice_evolution' mode.",
    )

    # Model Hparams
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # 1. Initialize Model
    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # 2. Setup Hook (Registered exactly once to prevent memory leaks)
    attn_maps = []

    def hook_fn(module, input, output):
        # We detach and move to CPU immediately to save VRAM
        attn_maps.append(module.attn_probs.detach().cpu())

    hook = model.eot_attentions.register_forward_hook(hook_fn)

    # 3. Discover Images
    img_files = sorted(
        f
        for f in os.listdir(args.img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )[
        :4
    ]  # Limit to 4 images for a clean figure

    if not img_files:
        raise RuntimeError(f"No images found in {args.img_dir}")

    num_images = len(img_files)

    # Determine columns based on mode
    if args.mode == "top_tokens":
        num_cols = 1 + 4  # Original + 4 Top Tokens
    else:
        num_cols = 1 + 5  # Original + 5 Slices

    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=num_cols,
        figsize=(4 * num_cols, 3 * num_images),
    )

    # Standardize axes array structure
    if num_images == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_images == 1:
        axes = [axes]
    elif num_cols == 1:
        axes = [[ax] for ax in axes]

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    try:
        for row_idx, img_name in enumerate(img_files):
            print(f"Processing {img_name}...")
            attn_maps.clear()  # CRITICAL: clear maps from previous image

            img_path = os.path.join(args.img_dir, img_name)
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)

            # Pad to multiple of 64
            H, W = x.size(2), x.size(3)
            pad_h = (64 - H % 64) % 64
            pad_w = (64 - W % 64) % 64
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

            # Forward pass (hook captures the maps)
            with torch.no_grad():
                _ = model(x_padded)

            padded_h, padded_w = x_padded.size(2), x_padded.size(3)
            latent_h, latent_w = (
                padded_h // 16,
                padded_w // 16,
            )  # 4 stride-2 downsamples

            # Plot original image in column 0
            orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
            ax_img = axes[row_idx][0]
            ax_img.imshow(orig_np)
            ax_img.axis("off")
            if row_idx == 0:
                ax_img.set_title("Input Image", fontsize=16, pad=10)

            # --- MODE 1: Top Tokens ---
            if args.mode == "top_tokens":
                probs = attn_maps[args.slice].squeeze(0)  # (HW, N)
                probs = probs.view(latent_h, latent_w, probs.size(-1))

                # Dynamically find the top 4 most utilized tokens
                token_usage = probs.sum(dim=(0, 1))
                top_tokens = torch.topk(token_usage, k=4).indices.tolist()
                print(
                    f"  Top 4 tokens for {img_name} (Slice {args.slice}): {top_tokens}"
                )

                # Find global maximum probability for these specific tokens
                # to normalize the colormap scientifically.
                global_max_prob = max([probs[:, :, t].max().item() for t in top_tokens])

                for col_idx, dict_idx in enumerate(top_tokens):
                    attn_map = (
                        probs[:, :, dict_idx].unsqueeze(0).unsqueeze(0)
                    )  # (1, 1, H, W)

                    # FIX: Use bilinear to prevent probability overshoot (<0 or >1)
                    attn_resized = F.interpolate(
                        attn_map,
                        size=(padded_h, padded_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    attn_np = attn_resized[:, :, :H, :W].squeeze().numpy()

                    ax_map = axes[row_idx][col_idx + 1]
                    ax_map.imshow(orig_np)
                    # FIX: Use fixed vmin/vmax so heatmaps reflect true relative magnitudes
                    ax_map.imshow(
                        attn_np, cmap="jet", alpha=0.55, vmin=0, vmax=global_max_prob
                    )
                    ax_map.axis("off")
                    if row_idx == 0:
                        ax_map.set_title(f"Dict Token {dict_idx}", fontsize=14, pad=10)

            # --- MODE 2: Slice Evolution ---
            elif args.mode == "slice_evolution":
                target = args.target_token

                # Collect the map for the target token across all 5 slices
                slice_maps = []
                for s in range(5):
                    probs = attn_maps[s].squeeze(0).view(latent_h, latent_w, -1)
                    slice_maps.append(probs[:, :, target])

                # Find global maximum across all slices for accurate comparison
                global_max_prob = max([sm.max().item() for sm in slice_maps])

                for s in range(5):
                    attn_map = slice_maps[s].unsqueeze(0).unsqueeze(0)

                    attn_resized = F.interpolate(
                        attn_map,
                        size=(padded_h, padded_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    attn_np = attn_resized[:, :, :H, :W].squeeze().numpy()

                    ax_map = axes[row_idx][s + 1]
                    ax_map.imshow(orig_np)
                    ax_map.imshow(
                        attn_np, cmap="jet", alpha=0.55, vmin=0, vmax=global_max_prob
                    )
                    ax_map.axis("off")
                    if row_idx == 0:
                        ax_map.set_title(
                            f"Slice {s} (Token {target})", fontsize=14, pad=10
                        )

    finally:
        # Guarantee hook is removed even if script crashes
        hook.remove()

    fig.savefig(args.output, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"\nSaved attention maps to {args.output}")


if __name__ == "__main__":
    main()
