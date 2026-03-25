import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms
from torchvision.utils import save_image

from models.WMDC import WMDC

# ---------------------------------------------------------------------------
# BPP helpers
# ---------------------------------------------------------------------------


def _byte_size(obj) -> int:
    """Recursively sum the byte-lengths of all byte-strings in a nested structure."""
    if isinstance(obj, bytes):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_byte_size(s) for s in obj)
    return 0


def compute_actual_bpp(strings, num_pixels: int) -> float:
    """Bits-per-pixel from actual compressed byte lengths."""
    return (_byte_size(strings) * 8) / num_pixels


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------


def pad_image(x: torch.Tensor, p: int = 64):
    H, W = x.size(2), x.size(3)
    pad_h = (p - H % p) % p
    pad_w = (p - W % p) % p
    if pad_h > 0 or pad_w > 0:
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    else:
        x_padded = x
    return x_padded, pad_h, pad_w


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _sync(device: str):
    """Synchronise CUDA stream so wall-clock timings are accurate."""
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Dictionary utilisation
# ---------------------------------------------------------------------------


def measure_dict_utilisation(model, x: torch.Tensor, device: str) -> dict:
    """
    Run a single forward pass and extract per-slice utilisation metrics.

    Returns a dict with:
      - 'marginal_entropy'  : column-marginal entropy H(m)  (nats)
      - 'max_entropy'       : log(dict_num)                  (nats)
      - 'utilisation_pct'   : H(m) / log(dict_num) * 100
      - 'slice_utilisation' : list[float] — one value per slice
                              (NOTE: eot_attention only stores the LAST slice's
                               attn_probs; full per-slice tracking requires a
                               small model change — see comment below)
    """
    x = x.to(device).float()
    x_pad, _, _ = pad_image(x, p=64)

    model.eval()
    with torch.no_grad():
        _ = model(x_pad)

    attn = model.eot_attention.attn_probs  # (B, HW, N) from last slice, or None
    result = {
        "marginal_entropy": float("nan"),
        "max_entropy": math.log(model.dict_num),
        "utilisation_pct": float("nan"),
        "slice_utilisation": [],
    }

    if attn is not None:
        marginal = attn.sum(dim=1)  # (B, N)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp(min=1e-8)
        H = -(marginal * marginal.clamp(min=1e-8).log()).sum(dim=1).mean()
        result["marginal_entropy"] = H.item()
        result["utilisation_pct"] = H.item() / result["max_entropy"] * 100.0

    # Clear buffer to prevent memory accumulation (32 MB per 4K image).
    model.eot_attention.attn_probs = None

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
        "--ot-eps", type=float, default=0.1, help="Must match training value."
    )
    parser.add_argument(
        "--sinkhorn-iters", type=int, default=20, help="Must match training value."
    )
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--measure-dict-util",
        action="store_true",
        help="Run an extra forward pass per image to measure "
        "dictionary utilisation (slower but informative).",
    )
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
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
    model.update(force=True)

    # ── Image list ───────────────────────────────────────────────────────────
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
    image_paths = sorted(
        os.path.join(args.dataset, f)
        for f in os.listdir(args.dataset)
        if os.path.splitext(f)[1].lower() in exts
    )
    if not image_paths:
        raise RuntimeError(f"No images found in {args.dataset}")

    # ── Per-image evaluation ──────────────────────────────────────────────────
    metrics = {
        k: []
        for k in (
            "bpp",
            "bpp_padded",
            "pad_overhead_bpp",
            "psnr",
            "ms_ssim",
            "enc_time",
            "dec_time",
        )
    }
    util_pcts: list[float] = []
    per_image: list[dict] = []

    to_tensor = transforms.ToTensor()

    with torch.no_grad():
        for img_path in image_paths:
            img = Image.open(img_path).convert("RGB")
            x = to_tensor(img).unsqueeze(0).to(device).float()

            H, W = x.size(2), x.size(3)
            num_pixels_orig = H * W

            x_pad, pad_h, pad_w = pad_image(x, p=64)
            H_pad = H + pad_h
            W_pad = W + pad_w
            num_pixels_padded = H_pad * W_pad

            # Encode
            _sync(device)
            t0 = time.time()
            out_enc = model.compress(x_pad)
            _sync(device)
            enc_time = time.time() - t0

            # Decode
            _sync(device)
            t1 = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            _sync(device)
            dec_time = time.time() - t1

            # Crop padding and clamp
            x_hat = out_dec["x_hat"]
            if pad_h > 0 or pad_w > 0:
                x_hat = x_hat[:, :, :H, :W]
            x_hat = x_hat.clamp(0, 1)

            save_image(x_hat, os.path.join(img_dir, os.path.basename(img_path)))

            # BPP — two variants
            # bpp_orig  : bits / original pixels  (standard Kodak/CLIC metric)
            # bpp_pad   : bits / padded pixels     (codec's actual cost domain)
            # pad_overhead = bpp_orig - bpp_pad    (should be < 0.001 for large images)
            bpp_orig = compute_actual_bpp(out_enc["strings"], num_pixels_orig)
            bpp_pad = compute_actual_bpp(out_enc["strings"], num_pixels_padded)
            pad_oh = bpp_orig - bpp_pad

            # Quality
            mse = F.mse_loss(x, x_hat)
            psnr = -10.0 * math.log10(mse.item()) if mse.item() > 0 else 100.0
            msssim = ms_ssim(x, x_hat, data_range=1.0).item()

            metrics["bpp"].append(bpp_orig)
            metrics["bpp_padded"].append(bpp_pad)
            metrics["pad_overhead_bpp"].append(pad_oh)
            metrics["psnr"].append(psnr)
            metrics["ms_ssim"].append(msssim)
            metrics["enc_time"].append(enc_time)
            metrics["dec_time"].append(dec_time)

            row = {
                "file": os.path.basename(img_path),
                "resolution": f"{W}x{H}",
                "bpp": round(bpp_orig, 4),
                "bpp_padded": round(bpp_pad, 4),
                "pad_overhead_bpp": round(pad_oh, 6),
                "psnr": round(psnr, 2),
                "ms_ssim": round(msssim, 4),
                "enc_time": round(enc_time, 3),
                "dec_time": round(dec_time, 3),
            }

            # Optional dictionary utilisation (extra forward pass)
            if args.measure_dict_util:
                util = measure_dict_utilisation(model, x, device)
                row["dict_utilisation_pct"] = round(util["utilisation_pct"], 2)
                util_pcts.append(util["utilisation_pct"])

            per_image.append(row)

            print(
                f"{os.path.basename(img_path):30s} | "
                f"BPP: {bpp_orig:.4f} (pad-domain: {bpp_pad:.4f}, "
                f"overhead: {pad_oh:+.5f}) | "
                f"PSNR: {psnr:.2f} dB | MS-SSIM: {msssim:.4f} | "
                f"Enc: {enc_time:.3f}s  Dec: {dec_time:.3f}s"
            )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def _stats(vals):
        a = float(np.mean(vals))
        s = float(np.std(vals))
        return {"mean": round(a, 4), "std": round(s, 4)}

    avg = {k: _stats(v) for k, v in metrics.items()}

    if util_pcts:
        avg["dict_utilisation_pct"] = _stats(util_pcts)

    print("\n── Average results ──────────────────────────────")
    print(
        f"  BPP (original pixels) : {avg['bpp']['mean']:.4f} ± {avg['bpp']['std']:.4f}"
    )
    print(
        f"  BPP (padded pixels)   : {avg['bpp_padded']['mean']:.4f} ± {avg['bpp_padded']['std']:.4f}"
    )
    print(
        f"  Pad overhead          : {avg['pad_overhead_bpp']['mean']:+.5f} bpp (mean)"
    )
    print(
        f"  PSNR                  : {avg['psnr']['mean']:.2f} ± {avg['psnr']['std']:.2f} dB"
    )
    print(
        f"  MS-SSIM               : {avg['ms_ssim']['mean']:.4f} ± {avg['ms_ssim']['std']:.4f}"
    )
    print(f"  Enc time              : {avg['enc_time']['mean']:.3f}s")
    print(f"  Dec time              : {avg['dec_time']['mean']:.3f}s")
    if util_pcts:
        print(
            f"  Dict utilisation      : {avg['dict_utilisation_pct']['mean']:.1f}% ± "
            f"{avg['dict_utilisation_pct']['std']:.1f}%"
        )

    report = {
        "average": avg,
        "per_image": per_image,
        "args": vars(args),
    }
    report_path = os.path.join(args.output, "RD_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nFull report saved → {report_path}")


if __name__ == "__main__":
    main()
