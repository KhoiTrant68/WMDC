import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image

from models.WMDC import WMDC


def pad_image(x, p=128):
    """Pads image to be divisible by p"""
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    if (
        padding_left == 0
        and padding_right == 0
        and padding_top == 0
        and padding_bottom == 0
    ):
        return x, (0, 0, 0, 0)
    x_padded = F.pad(
        x, (padding_left, padding_right, padding_top, padding_bottom), mode="reflect"
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def crop_image(x, padding):
    """Crops back to original size"""
    if sum(padding) == 0:
        return x
    return F.pad(x, (-padding[0], -padding[1], -padding[2], -padding[3]))


def compute_actual_bpp(strings, num_pixels):
    """Recursively computes the size of the bitstream in bytes, converts to BPP"""
    size = 0
    for s in strings:
        if isinstance(s, (list, tuple)):
            size += compute_actual_bpp(s, 1) / 8
        elif isinstance(s, bytes):
            size += len(s)
    return (size * 8) / num_pixels


def main():
    parser = argparse.ArgumentParser(description="Evaluate WMDC performance")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to evaluation dataset (e.g., Kodak)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to trained model .pth.tar"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Directory to save images and RD_report.json",
    )
    parser.add_argument("--N", type=int, default=192)
    parser.add_argument("--M", type=int, default=320)
    parser.add_argument("--cuda", action="store_true", help="Use GPU")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    # --- SETUP OUTPUT DIRECTORIES ---
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Loading model from {args.checkpoint}...")
    model = WMDC(N=args.N, M=args.M, num_slices=5).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    print("Updating Entropy Coder CDFs...")
    model.update(force=True)

    image_paths = sorted(
        [
            os.path.join(args.dataset, f)
            for f in os.listdir(args.dataset)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    results = {"bpp": [], "psnr": [], "ms_ssim": [], "enc_time": [], "dec_time": []}

    print(f"Evaluating on {len(image_paths)} images from {args.dataset}")
    print(f"Outputs will be saved to: {args.output}/")

    print("Warming up CUDA context for precise timing...")
    with torch.no_grad():
        dummy_x = torch.zeros(1, 3, 256, 256).to(device)
        _ = model.compress(dummy_x)
        if args.cuda:
            torch.cuda.synchronize()
    print("Warmup complete. Starting benchmark...")

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)

            x_padded, padding = pad_image(x, p=128)
            num_pixels_padded = x_padded.size(2) * x_padded.size(3)

            if args.cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            out_enc = model.compress(x_padded)
            if args.cuda:
                torch.cuda.synchronize()
            enc_time = time.time() - t0

            if args.cuda:
                torch.cuda.synchronize()
            t1 = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            if args.cuda:
                torch.cuda.synchronize()
            dec_time = time.time() - t1

            x_hat = crop_image(out_dec["x_hat"], padding).clamp_(0, 1)

            img_filename = os.path.basename(img_path)
            save_image(x_hat, os.path.join(img_dir, img_filename))

            bpp = compute_actual_bpp(out_enc["strings"], num_pixels_padded)
            mse = F.mse_loss(x, x_hat)
            psnr = -10 * math.log10(mse.item()) if mse.item() > 0 else 100
            msssim = ms_ssim(x, x_hat, data_range=1.0).item()

            results["bpp"].append(bpp)
            results["psnr"].append(psnr)
            results["ms_ssim"].append(msssim)
            results["enc_time"].append(enc_time)
            results["dec_time"].append(dec_time)
            print(
                f"Image: {img_filename} | BPP: {bpp:.4f} | PSNR: {psnr:.2f} | MS-SSIM: {msssim:.4f} | Enc: {enc_time:.3f}s | Dec: {dec_time:.3f}s"
            )

    avg_results = {k: sum(v) / len(v) for k, v in results.items()}
    with open(os.path.join(args.output, "RD_report.json"), "w") as f:
        json.dump(avg_results, f, indent=4)


if __name__ == "__main__":
    main()
