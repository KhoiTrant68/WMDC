import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Visualize EOT Attention Maps")
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("-d", "--img_dir", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="hdda_attention_maps.pdf")
    parser.add_argument(
        "--slice",
        type=int,
        default=0,
        help="Which autoregressive slice to visualise (0–4)",
    )
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    if args.slice < 0 or args.slice >= 5:
        raise ValueError(f"--slice must be 0–4, got {args.slice}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Dictionary token indices to visualise
    target_maps = [0, 32, 64, 127]

    model = WMDC(N=192, M=320, num_slices=5, routing_mode=args.routing_mode).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()

    img_files = sorted(
        f
        for f in os.listdir(args.img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )[:4]

    if not img_files:
        raise RuntimeError(f"No images found in {args.img_dir}")

    num_images = len(img_files)
    num_cols = len(target_maps) + 1  # 1 original + N attention maps

    fig, axes = plt.subplots(
        nrows=num_images,
        ncols=num_cols,
        figsize=(4 * num_cols, 3 * num_images),
    )

    # Normalise axes to always be a list of rows, each row a list of axes.
    # plt.subplots returns:
    #   num_images=1, num_cols=1  → a single Axes object
    #   num_images=1, num_cols>1  → 1-D array of Axes (length num_cols)
    #   num_images>1, num_cols=1  → 1-D array of Axes (length num_images)
    #   num_images>1, num_cols>1  → 2-D array (num_images, num_cols)
    if num_images == 1 and num_cols == 1:
        axes = [[axes]]
    elif num_images == 1:
        axes = [axes]  # list of one 1-D array
    elif num_cols == 1:
        axes = [[ax] for ax in axes]  # list of single-element lists
    # else: already (num_images, num_cols) 2-D — no change needed

    plt.subplots_adjust(wspace=0.02, hspace=0.02)

    for row_idx, img_name in enumerate(img_files):
        print(f"Processing {img_name}...")
        img_path = os.path.join(args.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)

        H, W = x.size(2), x.size(3)
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

        # Capture the attention probability tensor from the requested slice.
        # eot_attention is called once per autoregressive slice (5 times total).
        # We record all 5 outputs via the hook and index by args.slice.
        attn_maps = []

        def hook_fn(module, input, output):
            # module.attn_probs is set inside UnifiedDictionaryAttention.forward()
            # before the hook fires, so it always reflects the current slice call.
            attn_maps.append(module.attn_probs.detach().cpu())

        hook = model.eot_attention.register_forward_hook(hook_fn)

        try:
            with torch.no_grad():
                _ = model(x_padded)
        finally:
            # Always remove the hook even if forward() raises an exception,
            # to avoid accumulating hooks across loop iterations.
            hook.remove()

        # attn_maps[i] corresponds to autoregressive slice i.
        # Shape: (B=1, HW, N) where HW = (padded_H/16) * (padded_W/16).
        if args.slice >= len(attn_maps):
            raise RuntimeError(
                f"Hook only captured {len(attn_maps)} slice(s); "
                f"--slice {args.slice} is out of range."
            )

        probs = attn_maps[args.slice].squeeze(0)  # (HW, N)

        padded_h = H + pad_h
        padded_w = W + pad_w
        latent_h = padded_h // 16
        latent_w = padded_w // 16

        # Reshape flat HW dimension back to spatial grid
        probs = probs.view(latent_h, latent_w, probs.size(-1))  # (lh, lw, N)

        orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()  # (H, W, 3)

        # Column 0: original image
        ax_img = axes[row_idx][0]
        ax_img.imshow(orig_np)
        ax_img.axis("off")
        if row_idx == 0:
            ax_img.set_title("Input Image", fontsize=16, pad=10)

        # Columns 1+: one per target dictionary token
        for col_idx, dict_idx in enumerate(target_maps):
            # dict_idx=127 is valid for dict_num=128 (indices 0..127)
            attn_map = probs[:, :, dict_idx].unsqueeze(0).unsqueeze(0)  # (1,1,lh,lw)

            # Upsample to padded spatial resolution, then crop to original
            attn_resized = F.interpolate(
                attn_map,
                size=(padded_h, padded_w),
                mode="bicubic",
                align_corners=False,
            )
            attn_cropped = attn_resized[:, :, :H, :W]  # (1,1,H,W)
            attn_np = attn_cropped.squeeze().numpy()  # (H, W)

            # Min-max normalise for visualisation
            lo, hi = attn_np.min(), attn_np.max()
            attn_np = (attn_np - lo) / (hi - lo + 1e-8)

            ax_map = axes[row_idx][col_idx + 1]
            ax_map.imshow(orig_np)
            ax_map.imshow(attn_np, cmap="jet", alpha=0.55)
            ax_map.axis("off")

            if row_idx == 0:
                ax_map.set_title(f"Dict Token {dict_idx}", fontsize=14, pad=10)

    fig.savefig(args.output, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"Saved attention maps to {args.output}")


if __name__ == "__main__":
    main()
