import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a complex Kodak image (e.g., kodim23.png)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load Model
    model = WMDC(N=192, M=320, num_slices=5).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    model.update(force=True)

    # Process Image
    img = Image.open(args.image).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)

    H, W = x.size(2), x.size(3)
    num_pixels = H * W

    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")

    with torch.no_grad():
        out_enc = model.compress(x_padded)

    y_strings = out_enc["strings"][0]  # List of 5 slice strings
    z_strings = out_enc["strings"][1]  # Hyperprior string

    # Calculate BPP per slice
    slice_bpps = [(len(s[0]) * 8) / num_pixels for s in y_strings]
    z_bpp = (len(z_strings[0]) * 8) / num_pixels
    total_bpp = sum(slice_bpps) + z_bpp

    print(f"Total BPP: {total_bpp:.4f}")
    print(f"Hyperprior (Z) BPP: {z_bpp:.4f} ({z_bpp/total_bpp*100:.1f}%)")
    for i, bpp in enumerate(slice_bpps):
        print(f"Slice {i+1} BPP: {bpp:.4f} ({bpp/total_bpp*100:.1f}%)")

    # Generate Paper-Ready Plot
    labels = ["Z (Prior)", "Slice 1", "Slice 2", "Slice 3", "Slice 4", "Slice 5"]
    bpps = [z_bpp] + slice_bpps

    plt.figure(figsize=(8, 5))
    bars = plt.bar(
        labels,
        bpps,
        color=["gray", "darkred", "red", "orange", "lightcoral", "mistyrose"],
    )
    plt.ylabel("Bits Per Pixel (BPP)")
    plt.title("Bit Allocation Across Autoregressive Slices")

    # Add percentage labels on top
    for bar, bpp in zip(bars, bpps):
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{bpp/total_bpp*100:.1f}%",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig("bit_allocation_plot.pdf", dpi=300)
    print("Saved bit_allocation_plot.pdf for the paper.")


if __name__ == "__main__":
    main()
