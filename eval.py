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


def compute_actual_bpp(strings, num_pixels: int) -> float:
    """Compute BPP from actual compressed string byte lengths."""

    def get_size(obj):
        if isinstance(obj, bytes):
            return len(obj)
        elif isinstance(obj, (list, tuple)):
            return sum(get_size(s) for s in obj)
        return 0

    return (get_size(strings) * 8) / num_pixels


def pad_image(x: torch.Tensor, p: int = 64):
    H, W = x.size(2), x.size(3)
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    x_padded = (
        F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        if (pad_h > 0 or pad_w > 0)
        else x
    )
    return x_padded, pad_h, pad_w


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument(
        "--ot-eps",
        type=float,
        default=0.1,
        help="Must match the value used during training.",
    )
    parser.add_argument(
        "--sinkhorn-iters",
        type=int,
        default=20,
        help="Must match the value used during training.",
    )
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)

    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
    ).to(device)

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
    per_image = []

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            x = transforms.ToTensor()(img).unsqueeze(0).to(device)
            H, W = x.size(2), x.size(3)
            num_pixels_original = H * W

            x_padded, pad_h, pad_w = pad_image(x, p=64)

            t0 = time.time()
            out_enc = model.compress(x_padded)
            enc_time = time.time() - t0

            t1 = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            dec_time = time.time() - t1

            x_hat = out_dec["x_hat"]
            if pad_h > 0 or pad_w > 0:
                x_hat = x_hat[:, :, :H, :W]
            x_hat = x_hat.clamp(0, 1)

            save_image(x_hat, os.path.join(img_dir, os.path.basename(img_path)))

            # BPP always over original (unpadded) pixel count.
            bpp = compute_actual_bpp(out_enc["strings"], num_pixels_original)
            mse = F.mse_loss(x, x_hat)
            psnr = -10 * math.log10(mse.item()) if mse.item() > 0 else 100.0
            msssim = ms_ssim(x, x_hat, data_range=1.0).item()

            results["bpp"].append(bpp)
            results["psnr"].append(psnr)
            results["ms_ssim"].append(msssim)
            results["enc_time"].append(enc_time)
            results["dec_time"].append(dec_time)

            per_image.append(
                {
                    "file": os.path.basename(img_path),
                    "bpp": round(bpp, 4),
                    "psnr": round(psnr, 2),
                    "ms_ssim": round(msssim, 4),
                    "enc_time": round(enc_time, 3),
                    "dec_time": round(dec_time, 3),
                }
            )

            print(
                f"{os.path.basename(img_path)} | "
                f"BPP: {bpp:.4f} | PSNR: {psnr:.2f} dB | "
                f"MS-SSIM: {msssim:.4f} | "
                f"Enc: {enc_time:.3f}s | Dec: {dec_time:.3f}s"
            )

    avg_results = {k: sum(v) / len(v) for k, v in results.items()}
    report = {
        "average": avg_results,
        "per_image": per_image,
        "args": vars(args),
    }
    report_path = os.path.join(args.output, "RD_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\nAverage results saved to {report_path}")
    print(f"  BPP:     {avg_results['bpp']:.4f}")
    print(f"  PSNR:    {avg_results['psnr']:.2f} dB")
    print(f"  MS-SSIM: {avg_results['ms_ssim']:.4f}")
    print(f"  Enc:     {avg_results['enc_time']:.3f}s")
    print(f"  Dec:     {avg_results['dec_time']:.3f}s")


if __name__ == "__main__":
    main()
