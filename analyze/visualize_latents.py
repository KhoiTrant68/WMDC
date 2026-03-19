import argparse
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument(
        "--output", type=str, default="latent_sparsity_visualization.pdf"
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

        # Monkey-patch to capture heatmap
        def patched_decompress(*args, **kwargs):
            y_hat_slice = original_decompress(*args, **kwargs)
            spatial_heatmap = torch.mean(
                torch.abs(y_hat_slice.detach().cpu()), dim=1
            ).squeeze(0)
            latents.append(spatial_heatmap)
            return y_hat_slice

        model.gaussian_conditional.decompress = patched_decompress
        _ = model.decompress(out_enc["strings"], out_enc["shape"])
        model.gaussian_conditional.decompress = original_decompress  # Restore original

    fig, axes = plt.subplots(1, 6, figsize=(18, 3))

    # Show original unpadded image
    orig_np = x.squeeze(0).permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Use math.ceil to prevent truncation of fractional patches on the edge
    latent_h, latent_w = math.ceil(H / 16.0), math.ceil(W / 16.0)

    for i in range(5):
        hm = latents[i].numpy()
        hm_cropped = hm[:latent_h, :latent_w]

        axes[i + 1].imshow(hm_cropped, cmap="magma", interpolation="nearest")
        axes[i + 1].set_title(f"Slice {i+1} Latent")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches="tight")
    print(
        "Saved authentic sparsity visualization using the quantized bitstream variables."
    )


if __name__ == "__main__":
    main()
