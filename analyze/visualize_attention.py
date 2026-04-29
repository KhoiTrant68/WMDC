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
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )

    parser.add_argument("-d", "--img_dir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="attention_maps.pdf")

    parser.add_argument(
        "--mode",
        type=str,
        default="top_tokens",
        choices=["top_tokens", "slice_evolution", "spatial_gating", "entropy_map"],
    )

    parser.add_argument("--slice", type=int, default=0)
    parser.add_argument("--target_token", type=int, default=0)
    parser.add_argument("--top_k", type=int, default=4)

    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)

    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    if args.mode == "spatial_gating" and args.routing_mode != "unbalanced_eot":
        print(
            f"[WARN] spatial_gating requires routing_mode=unbalanced_eot "
            f"(got '{args.routing_mode}'). Cells will show N/A placeholders."
        )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Model
    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    # Load images
    img_files = sorted(
        [
            f
            for f in os.listdir(args.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )[:4]

    if not img_files:
        raise RuntimeError("No images found")

    num_images = len(img_files)

    # Determine columns
    if args.mode == "top_tokens":
        num_cols = 1 + args.top_k
    else:
        num_cols = 1 + model.num_slices

    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=num_cols,
        figsize=(4 * num_cols, 3 * num_images),
    )

    if num_images == 1:
        axes = [axes]

    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    for row_idx, img_name in enumerate(img_files):
        print(f"Processing {img_name}")

        img_path = os.path.join(args.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        x = transforms.ToTensor()(img).unsqueeze(0).to(device)

        H, W = x.shape[2:]
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64

        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        with torch.no_grad():
            _ = model(x_padded)

        attn_maps = [p.cpu() for p in model.slice_attn_probs]

        padded_h, padded_w = x_padded.shape[2:]
        latent_h, latent_w = padded_h // 16, padded_w // 16

        orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()

        ax_img = axes[row_idx][0]
        ax_img.imshow(orig_np)
        ax_img.axis("off")
        if row_idx == 0:
            ax_img.set_title("Input", fontsize=14)

        # =========================
        # MODE 1: TOP TOKENS
        # =========================
        if args.mode == "top_tokens":
            probs = attn_maps[args.slice].squeeze(0)
            probs = probs.view(latent_h, latent_w, -1)

            token_usage = probs.sum(dim=(0, 1))
            top_tokens = torch.topk(token_usage, k=args.top_k).indices.tolist()

            global_max = max([probs[:, :, t].max().item() for t in top_tokens])

            for col_idx, t in enumerate(top_tokens):
                attn = probs[:, :, t].unsqueeze(0).unsqueeze(0)

                attn_resized = F.interpolate(
                    attn,
                    size=(padded_h, padded_w),
                    mode="bilinear",
                    align_corners=False,
                )

                attn_np = attn_resized[:, :, :H, :W].squeeze().numpy()

                ax = axes[row_idx][col_idx + 1]
                ax.imshow(orig_np)
                ax.imshow(attn_np, cmap="viridis", alpha=0.6, vmin=0, vmax=global_max)
                ax.axis("off")

                if row_idx == 0:
                    ax.set_title(f"Token {t}")

        # =========================
        # MODE 2: SLICE EVOLUTION
        # =========================
        elif args.mode == "slice_evolution":
            target = args.target_token
            num_slices = len(attn_maps)

            slice_maps = []

            for s in range(num_slices):
                probs = attn_maps[s].squeeze(0).view(latent_h, latent_w, -1)
                slice_maps.append(probs[:, :, target])

            global_max = max([m.max().item() for m in slice_maps])

            for s in range(num_slices):
                attn = slice_maps[s].unsqueeze(0).unsqueeze(0)

                attn_resized = F.interpolate(
                    attn,
                    size=(padded_h, padded_w),
                    mode="bilinear",
                    align_corners=False,
                )

                attn_np = attn_resized[:, :, :H, :W].squeeze().numpy()

                ax = axes[row_idx][s + 1]
                ax.imshow(orig_np)
                ax.imshow(attn_np, cmap="viridis", alpha=0.6, vmin=0, vmax=global_max)
                ax.axis("off")

                if row_idx == 0:
                    ax.set_title(f"Slice {s}")

        # =========================
        # MODE 3: SPATIAL GATING
        # =========================
        elif args.mode == "spatial_gating":
            num_slices = len(attn_maps)

            for s in range(num_slices):
                ax = axes[row_idx][s + 1]
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"Slice {s}")

                raw = model.eot_attentions[s].last_row_mass
                if raw is None:
                    # last_row_mass is only populated for unbalanced_eot.
                    ax.text(
                        0.5, 0.5,
                        f"N/A\n(requires\nunbalanced_eot,\ngot {args.routing_mode})",
                        ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="gray",
                    )
                    continue

                row_mass = raw.cpu().view(1, 1, latent_h, latent_w)
                mass_resized = F.interpolate(
                    row_mass,
                    size=(padded_h, padded_w),
                    mode="bilinear",
                    align_corners=False,
                )
                mass_np = mass_resized[:, :, :H, :W].squeeze().numpy()
                ax.imshow(orig_np)
                ax.imshow(mass_np, cmap="inferno", alpha=0.7)

        # =========================
        # MODE 4: ENTROPY MAP 🔥
        # =========================
        elif args.mode == "entropy_map":
            num_slices = len(attn_maps)
            entropy_maps = []

            for s in range(num_slices):
                probs = attn_maps[s].squeeze(0)
                probs = probs.view(latent_h, latent_w, -1)

                eps = 1e-8
                entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)

                entropy_maps.append(entropy)

            global_max = max([e.max().item() for e in entropy_maps])

            for s in range(num_slices):
                attn = entropy_maps[s].unsqueeze(0).unsqueeze(0)

                attn_resized = F.interpolate(
                    attn,
                    size=(padded_h, padded_w),
                    mode="bilinear",
                    align_corners=False,
                )

                attn_np = attn_resized[:, :, :H, :W].squeeze().numpy()

                ax = axes[row_idx][s + 1]
                ax.imshow(orig_np)
                ax.imshow(attn_np, cmap="inferno", alpha=0.7, vmin=0, vmax=global_max)
                ax.axis("off")

                if row_idx == 0:
                    ax.set_title(f"Slice {s}")

    fig.savefig(args.output, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
