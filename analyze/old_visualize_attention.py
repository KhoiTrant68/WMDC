import argparse
import os

import matplotlib
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from models.WMDC import WMDC


def main():
    parser = argparse.ArgumentParser(
        description="Visualize WMDC Dictionary Attention Maps"
    )
    parser.add_argument(
        "-d",
        "--img_dir",
        type=str,
        required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="attention_maps_grid.pdf",
        help="Output filename",
    )
    parser.add_argument(
        "--slice",
        type=int,
        default=0,
        help="Which slice to visualize (0 to num_slices-1)",
    )
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # Format: (Entry Index, Head Index)
    target_maps = [
        (0, 0),
        (10, 0),
        (50, 0),
        (99, 0),
        (120, 0),
        (0, 1),
        (10, 1),
        (50, 1),
        (99, 1),
        (120, 1),
    ]

    model = WMDC(N=args.N, M=args.M, num_slices=5).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    extracted_probs = []

    def get_attention_probs(module, input, output):
        extracted_probs.append(output.detach().cpu())

    target_module = model.fusion_lh[args.slice].softmax
    hook_handle = target_module.register_forward_hook(get_attention_probs)

    img_files = sorted(
        [
            f
            for f in os.listdir(args.img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )[:5]

    num_images = len(img_files)
    num_cols = len(target_maps) + 1

    fig, axes = plt.subplots(
        nrows=num_images, ncols=num_cols, figsize=(3 * num_cols, 2 * num_images)
    )
    if num_images == 1:
        axes = [axes]  # Ensure 2D indexing works

    plt.subplots_adjust(wspace=0.05, hspace=0.05)  # Tight layout like the paper

    for row_idx, img_name in enumerate(img_files):
        print(f"Processing {img_name}...")
        img_path = os.path.join(args.img_dir, img_name)
        img = Image.open(img_path).convert("RGB")
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)

        orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()

        # The downsampling factor in WMDC to the latent space is 32x (16x from g_a, 2x from DWT)
        latent_h = x.size(2) // 32
        latent_w = x.size(3) // 32
        max_dict_idx = (
            latent_h * latent_w
        ) - 1  # Maximum valid index in the dictionary

        extracted_probs.clear()
        with torch.no_grad():
            _ = model(x)

        # Grab the extracted attention tensor
        # Shape:[1, head_num, (latent_h * latent_w), dict_num]
        probs = extracted_probs[0].squeeze(0)

        # Reshape spatial dimension back to 2D
        # Shape:[head_num, latent_h, latent_w, dict_num]
        probs = probs.view(probs.size(0), latent_h, latent_w, probs.size(-1))

        ax_img = axes[row_idx][0]
        ax_img.imshow(orig_np)

        if row_idx == num_images - 1:
            ax_img.set_xlabel("Input\nImage", fontsize=16, labelpad=10)
            ax_img.xaxis.set_label_position("bottom")

        ax_img.set_xticks([])
        ax_img.set_yticks([])
        for spine in ax_img.spines.values():
            spine.set_edgecolor("gray")
            spine.set_linewidth(1.5)

        # --- PLOT ATTENTION MAPS (Columns 1 to N) ---
        for col_idx, (entry_idx, head_idx) in enumerate(target_maps):

            safe_entry_idx = min(entry_idx, max_dict_idx)

            # Extract 2D heatmap:[latent_h, latent_w]
            attn_map = probs[head_idx, :, :, safe_entry_idx].numpy()

            ax_map = axes[row_idx][col_idx + 1]

            # Use 'hot' colormap
            im = ax_map.imshow(
                attn_map, cmap="gist_heat", aspect="auto", interpolation="nearest"
            )

            # Format axes
            ax_map.tick_params(axis="both", which="major", labelsize=6)
            for spine in ax_map.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(0.5)

            if row_idx == num_images - 1:
                label_txt = f"Dictionary\nEntry {safe_entry_idx}, Head {head_idx}"
                if safe_entry_idx != entry_idx:
                    label_txt += "\n(Clamped)"
                ax_map.set_xlabel(label_txt, fontsize=14, labelpad=10)

    hook_handle.remove()
    plt.savefig(args.output, format="pdf", bbox_inches="tight", dpi=300)
    plt.savefig(
        args.output.replace(".pdf", ".jpg"), format="jpg", bbox_inches="tight", dpi=300
    )
    plt.close()


if __name__ == "__main__":
    main()
