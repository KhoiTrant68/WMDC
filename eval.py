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
    """
    Recursively sum byte-lengths of all bytes objects in a nested structure.
        strings = [y_strings, z_strings]
        y_strings = [[bytes, bytes, ...], [bytes, bytes, ...], ...]   (5 slices × batch)
        z_strings = [bytes, bytes, ...]                                (batch)
    """
    if isinstance(obj, (bytes, bytearray)):
        return len(obj)
    if isinstance(obj, (list, tuple)):
        return sum(_byte_size(item) for item in obj)
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
    x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect") if (pad_h or pad_w) else x
    return x_padded, pad_h, pad_w


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------


def _sync(device: str):
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def probe_y_to_z_ratio(model: WMDC, device: str) -> tuple[int, int]:
    with torch.no_grad():
        dummy = torch.zeros(1, 3, 64, 64, device=device)
        dummy_y = model.g_a(dummy)
        dummy_z = model.h_a(dummy_y)
    rh = dummy_y.shape[2] // dummy_z.shape[2]
    rw = dummy_y.shape[3] // dummy_z.shape[3]
    return rh, rw


# ---------------------------------------------------------------------------
# Dictionary utilisation (all slices)
# ---------------------------------------------------------------------------


def compute_util_from_saved(
    all_probs: list,
    all_row_masses: list,
    shape: tuple,
    dict_num: int,
    y_to_z_ratio: tuple[int, int],
) -> dict:
    if not all_probs:
        return {
            "error": "No attention maps collected. Check routing_mode != 'softmax'."
        }

    H_lat_y = shape[0] * y_to_z_ratio[0]
    W_lat_y = shape[1] * y_to_z_ratio[1]
    max_ent = math.log2(dict_num)

    per_slice_util: list[float] = []
    assignment_maps: list = []
    entropy_maps: list = []
    row_mass_maps: list = []

    for s_idx, P in enumerate(all_probs):
        B, HW, N = P.shape
        marginal = P.sum(dim=1)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp(min=1e-8)
        H_bits = -(marginal * marginal.clamp(1e-8).log2()).sum(dim=1).mean()
        per_slice_util.append(float(H_bits.item() / max_ent * 100.0))

        try:
            P_sp = P[0].view(H_lat_y, W_lat_y, N)
        except RuntimeError:
            assignment_maps.append(None)
            entropy_maps.append(None)
            row_mass_maps.append(None)
            continue

        assignment_maps.append(P_sp.argmax(dim=-1).cpu())

        p_norm = P_sp / P_sp.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        H_pixel = -(p_norm * p_norm.clamp(1e-8).log2()).sum(dim=-1)
        entropy_maps.append(H_pixel.cpu())

        rm = all_row_masses[s_idx] if s_idx < len(all_row_masses) else None
        if rm is not None:
            try:
                row_mass_maps.append(rm[0].view(H_lat_y, W_lat_y).cpu())
            except RuntimeError:
                row_mass_maps.append(None)
        else:
            row_mass_maps.append(None)

    return {
        "per_slice_utilisation_pct": per_slice_util,
        "mean_utilisation_pct": float(np.mean(per_slice_util)),
        "max_entropy_bits": max_ent,
        "assignment_maps": assignment_maps,
        "entropy_maps": entropy_maps,
        "row_mass_maps": row_mass_maps,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="output")
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--marginal-div",
        type=str,
        default="kl",
        choices=["kl", "tv"],
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="fdm",
        choices=["fdm", "ss2d", "cnn", "swin", "fdm_reversed"],
    )
    parser.add_argument("--use-dense-concat", action="store_true")
    parser.add_argument(
        "--memory-init",
        type=str,
        default="bootstrap",
        choices=["bootstrap", "zero"],
    )
    parser.add_argument(
        "--content-adaptive",
        action="store_true",
        default=False,
        dest="use_content_adaptive",
    )
    parser.add_argument("--cluster-num", type=int, default=8, dest="cluster_num")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--measure-dict-util", action="store_true")
    parser.add_argument("--save-attn-maps", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output, exist_ok=True)
    img_dir = os.path.join(args.output, "images")
    os.makedirs(img_dir, exist_ok=True)
    if args.save_attn_maps:
        attn_dir = os.path.join(args.output, "attn_maps")
        os.makedirs(attn_dir, exist_ok=True)

    # ── Model ────────────────────────────────────────────────────────────────
    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=args.routing_mode,
        marginal_div=args.marginal_div,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        backbone=args.backbone,
        use_dense_concat=args.use_dense_concat,
        memory_init=args.memory_init,
        use_content_adaptive=args.use_content_adaptive,
        cluster_num=args.cluster_num,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    # strict=False: routing-mode-aware parameter cleanup
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[load] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    model.eval()
    model.update(force=True)

    if args.measure_dict_util:
        for a in model.eot_attentions:
            a.store_attn_probs = True

    y_to_z_ratio = probe_y_to_z_ratio(model, device)
    print(f"Probed y-to-z spatial ratio: {y_to_z_ratio}")

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
    nan_images: list[str] = []
    per_slice_util_all: list[list[float]] = []
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

            # Clone attention maps immediately
            saved_attn_probs = [p.detach().cpu() for p in model.slice_attn_probs]
            model.slice_attn_probs.clear()
            saved_row_masses = []
            for a in model.eot_attentions:
                rm = getattr(a, "last_row_mass", None)
                saved_row_masses.append(rm.detach().cpu() if rm is not None else None)

            _sync(device)
            enc_time = time.time() - t0

            # Decode
            _sync(device)
            t1 = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            _sync(device)
            dec_time = time.time() - t1

            # Crop and clamp
            x_hat = out_dec["x_hat"]
            if pad_h > 0 or pad_w > 0:
                x_hat = x_hat[:, :, :H, :W]
            x_hat = x_hat.clamp(0, 1)

            has_nan = torch.isnan(x_hat).any().item()
            if has_nan:
                nan_frac = torch.isnan(x_hat).float().mean().item()
                print(
                    f"[WARN] {os.path.basename(img_path)}: x_hat has "
                    f"{nan_frac:.1%} NaN values — metrics set to 0"
                )
                nan_images.append(os.path.basename(img_path))
                x_hat = torch.nan_to_num(x_hat, nan=0.0, posinf=1.0, neginf=0.0)
                psnr = 0.0
                msssim = 0.0
            else:
                save_image(x_hat, os.path.join(img_dir, os.path.basename(img_path)))
                mse = F.mse_loss(x, x_hat)
                psnr = -10.0 * math.log10(mse.item()) if mse.item() > 0 else 100.0
                try:
                    msssim = ms_ssim(x, x_hat, data_range=1.0).item()
                except Exception:
                    msssim = float("nan")

            # BPP — two variants
            bpp_orig = compute_actual_bpp(out_enc["strings"], num_pixels_orig)
            bpp_pad = compute_actual_bpp(out_enc["strings"], num_pixels_padded)
            pad_oh = bpp_orig - bpp_pad

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
                "ms_ssim": round(msssim, 4) if not math.isnan(msssim) else None,
                "enc_time": round(enc_time, 3),
                "dec_time": round(dec_time, 3),
                "has_nan": has_nan,
            }

            if args.measure_dict_util:
                util = compute_util_from_saved(
                    saved_attn_probs,
                    saved_row_masses,
                    out_enc["shape"],
                    model.dict_num,
                    y_to_z_ratio,
                )

                if "error" not in util:
                    row["per_slice_utilisation_pct"] = [
                        round(v, 2) for v in util["per_slice_utilisation_pct"]
                    ]
                    row["mean_utilisation_pct"] = round(util["mean_utilisation_pct"], 2)
                    per_slice_util_all.append(util["per_slice_utilisation_pct"])

                    if args.save_attn_maps:
                        stem = os.path.splitext(os.path.basename(img_path))[0]
                        for s_idx, (amap, emap, rmap) in enumerate(
                            zip(
                                util["assignment_maps"],
                                util["entropy_maps"],
                                util["row_mass_maps"],
                            )
                        ):
                            if amap is not None:
                                torch.save(
                                    amap,
                                    os.path.join(
                                        attn_dir, f"{stem}_slice{s_idx}_assign.pt"
                                    ),
                                )
                            if emap is not None:
                                torch.save(
                                    emap,
                                    os.path.join(
                                        attn_dir, f"{stem}_slice{s_idx}_entropy.pt"
                                    ),
                                )
                            if rmap is not None:
                                torch.save(
                                    rmap,
                                    os.path.join(
                                        attn_dir, f"{stem}_slice{s_idx}_rowmass.pt"
                                    ),
                                )
                else:
                    print(f"Warning: {util['error']}")

            per_image.append(row)

            nan_marker = " [NaN→0]" if has_nan else ""
            util_str = ""
            if args.measure_dict_util and "mean_utilisation_pct" in row:
                slices = row["per_slice_utilisation_pct"]
                util_str = (
                    " | Util: ["
                    + ", ".join(f"{v:.1f}%" for v in slices)
                    + f"] mean={row['mean_utilisation_pct']:.1f}%"
                )

            print(
                f"{os.path.basename(img_path):30s} | "
                f"BPP: {bpp_orig:.4f} | "
                f"PSNR: {psnr:.2f} dB | MS-SSIM: {msssim:.4f} | "
                f"Enc: {enc_time:.3f}s  Dec: {dec_time:.3f}s"
                f"{util_str}{nan_marker}"
            )

    # ── Aggregate statistics ──────────────────────────────────────────────────
    def _stats(vals):
        # Exclude NaN from statistics (NaN images report psnr=0 not nan)
        clean = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
        if not clean:
            return {"mean": float("nan"), "std": float("nan")}
        a = float(np.mean(clean))
        s = float(np.std(clean))
        return {"mean": round(a, 4), "std": round(s, 4)}

    avg = {k: _stats(v) for k, v in metrics.items()}

    if per_slice_util_all:
        per_slice_arr = np.array(per_slice_util_all)
        avg["per_slice_utilisation_pct"] = [
            {
                "slice": i,
                "mean": round(float(per_slice_arr[:, i].mean()), 2),
                "std": round(float(per_slice_arr[:, i].std()), 2),
            }
            for i in range(per_slice_arr.shape[1])
        ]
        avg["mean_utilisation_pct"] = _stats(list(per_slice_arr.mean(axis=1)))

    print("\n── Average results ──────────────────────────────")
    print(
        f"  BPP                   : {avg['bpp']['mean']:.4f} ± {avg['bpp']['std']:.4f}"
    )
    print(
        f"  PSNR                  : {avg['psnr']['mean']:.2f} ± {avg['psnr']['std']:.2f} dB"
    )
    print(
        f"  MS-SSIM               : {avg['ms_ssim']['mean']:.4f} ± {avg['ms_ssim']['std']:.4f}"
    )
    print(f"  Enc time              : {avg['enc_time']['mean']:.3f}s")
    print(f"  Dec time              : {avg['dec_time']['mean']:.3f}s")
    if nan_images:
        print(
            f"\n  [WARN] {len(nan_images)} images had NaN x_hat (PSNR set to 0, not 100):"
        )
        for n in nan_images:
            print(f"         - {n}")
    if per_slice_util_all:
        print("  Dict utilisation per slice:")
        for s in avg["per_slice_utilisation_pct"]:
            print(f"    Slice {s['slice']}: {s['mean']:.1f}% ± {s['std']:.1f}%")
        print(f"  Overall mean util     : {avg['mean_utilisation_pct']['mean']:.1f}%")

    report = {
        "average": avg,
        "per_image": per_image,
        "nan_images": nan_images,
        "args": vars(args),
        "y_to_z_ratio": list(y_to_z_ratio),
    }
    report_path = os.path.join(args.output, "RD_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\nFull report saved → {report_path}")


if __name__ == "__main__":
    main()
