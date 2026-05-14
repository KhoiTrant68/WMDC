from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Callable

import torch
import torch.nn as nn
from _common import load_model

try:
    from fvcore.nn import FlopCountAnalysis

    _HAS_FVCORE = True
except ImportError:
    _HAS_FVCORE = False
    try:
        from thop import profile as thop_profile

        _HAS_THOP = True
    except ImportError:
        _HAS_THOP = False


def _build_model_with_backbone(backbone: str, device: str) -> nn.Module:
    from models.WMDC import WMDC

    try:
        model = WMDC(N=192, M=320, num_slices=5, backbone=backbone).to(device)
    except TypeError:
        print(
            f"[INFO] WMDC.__init__ does not accept `backbone=`. "
            f"Falling back to in-place module swap for backbone={backbone}."
        )
        model = WMDC(N=192, M=320, num_slices=5).to(device)
        _swap_backbones(model, backbone)
    return model.eval()


def _swap_backbones(model: nn.Module, backbone: str) -> None:
    from ablation_models.backbone_variants import build_backbone
    from modules.wavelet_blocks import FrequencyDisentangledMamba

    replacements = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, FrequencyDisentangledMamba):
                dim = (
                    getattr(child, "dim", None)
                    or getattr(child, "hidden_dim", None)
                    or 192
                )
                new_block = build_backbone(backbone, dim=dim)
                replacements.append((parent, name, new_block))
    for parent, name, new_block in replacements:
        setattr(parent, name, new_block.to(next(model.parameters()).device))
    print(f"[INFO] Swapped {len(replacements)} FDM blocks for backbone={backbone}")


def _measure_latency(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int = 50,
    iters: int = 200,
    device: str = "cuda",
) -> dict:
    use_cuda = device == "cuda" and torch.cuda.is_available()
    for _ in range(warmup):
        _ = fn()
    if use_cuda:
        torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        if use_cuda:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = fn()
            torch.cuda.synchronize()
        else:
            t0 = time.perf_counter()
            _ = fn()
        samples_ms.append((time.perf_counter() - t0) * 1000.0)

    samples_ms.sort()
    n = len(samples_ms)
    return {
        "median_ms": statistics.median(samples_ms),
        "p25_ms": samples_ms[n // 4],
        "p75_ms": samples_ms[(3 * n) // 4],
        "mean_ms": statistics.mean(samples_ms),
        "std_ms": statistics.stdev(samples_ms) if n > 1 else 0.0,
        "n": n,
    }


def _count_flops(model: nn.Module, x: torch.Tensor) -> dict:
    """
    Count forward-pass FLOPs via fvcore or thop.
    """
    if _HAS_FVCORE:
        try:
            flops = FlopCountAnalysis(model, x)
            flops.unsupported_ops_warnings(True)
            flops.uncalled_modules_warnings(False)

            unsupported = flops.unsupported_ops()
            total = flops.total()

            # Detect Mamba selective_scan ops specifically
            mamba_missing = any(
                "selective_scan" in str(k).lower() or "mamba" in str(k).lower()
                for k in unsupported
            )

            if mamba_missing:
                missing_names = [
                    str(k) for k in unsupported if "selective_scan" in str(k).lower()
                ]
                print(
                    f"[WARN] fvcore: {len(unsupported)} unsupported op type(s), including "
                    f"Mamba selective_scan CUDA kernels: {missing_names[:3]}. "
                    f"Reported GFLOPs are a LOWER BOUND. "
                    f"Use latency for fair comparison in Table 2."
                )

            return {
                "gflops": total / 1e9,
                "library": "fvcore (lower bound)" if mamba_missing else "fvcore",
                "mamba_ops_missing": mamba_missing,
                "unsupported_op_count": len(unsupported),
                "note": (
                    "selective_scan CUDA ops excluded; true GFLOPs ~2-3x reported"
                    if mamba_missing
                    else None
                ),
            }
        except Exception as e:
            print(f"[WARN] fvcore FlopCountAnalysis failed: {e}")

    if _HAS_THOP:
        try:
            macs, _ = thop_profile(model, inputs=(x,), verbose=False)
            return {
                "gflops": macs * 2 / 1e9,
                "library": "thop (MAC*2, Mamba ops may be missing)",
                "mamba_ops_missing": True,  # thop also misses custom CUDA
                "note": "thop cannot count Mamba selective_scan CUDA ops",
            }
        except Exception as e:
            print(f"[WARN] thop.profile failed: {e}")

    return {
        "gflops": float("nan"),
        "library": "none (install fvcore: pip install fvcore)",
        "mamba_ops_missing": True,
        "note": "No FLOP counter available",
    }


def benchmark_one(backbone: str, args, device: str) -> dict:
    print(f"\n=== Backbone: {backbone} ===")
    model = _build_model_with_backbone(backbone, device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        miss, unexp = model.load_state_dict(sd, strict=False)
        if miss or unexp:
            print(f"  [load] {len(miss)} missing, {len(unexp)} unexpected")

    H, W = args.resolution
    x = torch.randn(1, 3, H, W, device=device)

    n_params = sum(p.numel() for p in model.parameters())

    print("  Measuring full-forward latency...")
    full_lat = _measure_latency(
        lambda: model(x), warmup=args.warmup, iters=args.iters, device=device
    )

    print("  Measuring encoder-only latency...")

    def enc_fn():
        with torch.no_grad():
            y = model.g_a(x)
            z = model.h_a(y)
            return z

    enc_lat = _measure_latency(
        enc_fn, warmup=args.warmup, iters=args.iters, device=device
    )

    print("  Counting GFLOPs (fvcore — lower bound for FDM due to Mamba CUDA ops)...")
    flops = _count_flops(model, x)

    return {
        "backbone": backbone,
        "params_M": n_params / 1e6,
        "resolution": [H, W],
        "full_latency": full_lat,
        "encoder_latency": enc_lat,
        "flops": flops,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backbones",
        nargs="+",
        default=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
        choices=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
    )
    p.add_argument("--resolution", type=int, nargs=2, default=[768, 512])
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default="backbone_benchmark.json")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if not _HAS_FVCORE:
        print(
            "[INFO] fvcore not installed → GFLOPs unavailable. Install: pip install fvcore"
        )

    print(
        "\n[NOTE] GFLOPs for FDM-based backbones (fdm, fdm_reversed) are LOWER BOUNDS.\n"
        "       Mamba selective_scan CUDA kernels are not counted by fvcore/thop.\n"
        "       Use 'full_latency.median_ms' as the primary metric in Table 2.\n"
    )

    results = []
    for bb in args.backbones:
        try:
            res = benchmark_one(bb, args, device)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] backbone={bb} failed: {e}")
            results.append({"backbone": bb, "error": str(e)})

    print("\n" + "=" * 90)
    print(
        f"{'Backbone':<14} {'Params(M)':>10} {'GFLOPs':>12} "
        f"{'Mamba?':>8} {'Full med(ms)':>14} {'Enc med(ms)':>14}"
    )
    print("-" * 90)
    for r in results:
        if "error" in r:
            print(f"{r['backbone']:<14}   ERROR: {r['error']}")
            continue
        flops_str = (
            f"{r['flops']['gflops']:.2f}†"
            if r["flops"].get("mamba_ops_missing")
            else f"{r['flops']['gflops']:.2f}"
        )
        print(
            f"{r['backbone']:<14} "
            f"{r['params_M']:>10.2f} "
            f"{flops_str:>12} "
            f"{'yes' if r['flops'].get('mamba_ops_missing') else 'no':>8} "
            f"{r['full_latency']['median_ms']:>14.2f} "
            f"{r['encoder_latency']['median_ms']:>14.2f}"
        )
    print("=" * 90)
    print("† GFLOPs lower bound (Mamba selective_scan CUDA ops excluded).")
    print("  Use latency column for fair cross-backbone comparison in Table 2.")

    with open(args.output, "w") as f:
        json.dump({"resolution": args.resolution, "results": results}, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
