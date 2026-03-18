import argparse
import math
import os

import matplotlib
import torch
import torch.nn.functional as F
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms

from models.WMDC import WMDC


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Visual Comparison Patches for WMDC"
    )
    parser.add_argument("-i", "--image", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", type=str, required=True)
    parser.add_argument("-o", "--output", type=str, default="visual_comparison")
    parser.add_argument(
        "--crop", type=int, nargs=4, default=[300, 450, 250, 400], help="y1 y2 x1 x2"
    )
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--cuda", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = WMDC(N=args.N, M=args.M).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    model.update()

    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    H, W = x.size(2), x.size(3)
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out_net = model(x_padded)
        x_hat_padded = out_net["x_hat"].clamp(0, 1)

        # Crop back to original dimensions
        x_hat = x_hat_padded[:, :, :H, :W]

        mse = F.mse_loss(x, x_hat)
        psnr = -10 * math.log10(mse.item()) if mse.item() > 0 else 100

        # Calculate BPP against original unpadded pixels
        num_pixels = H * W
        bpp = sum(torch.log(lh).sum() for lh in out_net["likelihoods"].values()) / (
            -math.log(2) * num_pixels
        )

    orig_np = x.squeeze().permute(1, 2, 0).cpu().numpy()
    rec_np = x_hat.squeeze().permute(1, 2, 0).cpu().numpy()

    y1, y2, x1, x2 = args.crop

    if y2 > orig_np.shape[0] or x2 > orig_np.shape[1] or y1 >= y2 or x1 >= x2:
        y1, y2, x1, x2 = 0, orig_np.shape[0], 0, orig_np.shape[1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(orig_np[y1:y2, x1:x2])
    axes[0].set_title("Original Image (Crop)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(rec_np[y1:y2, x1:x2])
    axes[1].set_title(
        f"WMDC (Ours)\nBPP: {bpp.item():.3f} | PSNR: {psnr:.2f}dB", fontsize=14
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{args.output}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    plt.savefig(f"{args.output}.jpg", format="jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Success!")


if __name__ == "__main__":
    main()
