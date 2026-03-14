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


def compute_actual_bpp(strings, num_pixels):
    def get_size(obj):
        if isinstance(obj, bytes):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(get_size(s) for s in obj)
        return 0

    return (get_size(strings) * 8) / num_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)

    model = WMDC(N=192, M=320, num_slices=5).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint.get("state_dict", checkpoint))
    model.eval()
    model.update(force=True)

    image_paths = sorted(
        [
            os.path.join(args.dataset, f)
            for f in os.listdir(args.dataset)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )
    results = {"bpp": [], "psnr": [], "ms_ssim": [], "enc_time": [], "dec_time": []}

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            num_pixels_original = x.size(2) * x.size(3)

            t0 = time.time()
            out_enc = model.compress(x)
            enc_time = time.time() - t0

            t1 = time.time()
            out_dec = model.decompress(
                out_enc["strings"],
                out_enc["shape"],
                original_shape=out_enc.get("original_shape"),
            )
            dec_time = time.time() - t1

            x_hat = out_dec["x_hat"].clamp(0, 1)
            save_image(x_hat, os.path.join(img_dir, os.path.basename(img_path)))

            bpp = compute_actual_bpp(out_enc["strings"], num_pixels_original)
            mse = F.mse_loss(x, x_hat)
            psnr = -10 * math.log10(mse.item()) if mse.item() > 0 else 100
            msssim = ms_ssim(x, x_hat, data_range=1.0).item()

            results["bpp"].append(bpp)
            results["psnr"].append(psnr)
            results["ms_ssim"].append(msssim)
            results["enc_time"].append(enc_time)
            results["dec_time"].append(dec_time)

            print(
                f"Image: {os.path.basename(img_path)} | BPP: {bpp:.4f} | "
                f"PSNR: {psnr:.2f} | Enc: {enc_time:.3f}s | Dec: {dec_time:.3f}s"
            )

    avg_results = {k: sum(v) / len(v) for k, v in results.items()}
    with open(os.path.join(args.output, "RD_report.json"), "w") as f:
        json.dump(avg_results, f, indent=4)


if __name__ == "__main__":
    main()
