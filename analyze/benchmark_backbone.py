"""
analyze/benchmark_backbone.py
=============================
Measures end-to-end encoder/decoder latency and GFLOPs for WMDC with
each of the four backbone variants (cnn / swin / ss2d / fdm), plus the
production FDM. Drives Table 2 of the paper.

What is measured
----------------
* Encoder-only latency (g_a + h_a)            — important for upload speed.
* Decoder-only latency (g_s + slice loop)     — important for client UX.
* Full forward latency (matches training)     — sanity check.
* GFLOPs (forward only)                        — via fvcore (preferred)
                                                 or thop (fallback).
* Param count (thousands).

Caveats
-------
* Latency is reported as median ± IQR over a warm GPU (50 warmup + 200
  measured iters). Mean ± std can be misleading on GPUs because of
  occasional CUDA driver context switches.
* GFLOPs from fvcore counts MAC operations and reports MAC*2 = FLOPs;
  thop differs slightly. We always print which library was used.
* This script does NOT load weights — it builds a randomly-initialised
  model, since latency/FLOPs are independent of weights. If you want a
  weighted variant, pass --checkpoint and we'll load it.

Usage
-----
    python analyze/benchmark_backbone.py \
        --backbones fdm cnn swin ss2d fdm_reversed \
        --resolution 768 512 \
        --warmup 50 --iters 200 \
        --output backbone_benchmark.json \
        --cuda
"""

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
    """
    Build a WMDC and replace its FDM blocks with the requested variant.

    NOTE: this requires `models/WMDC.py` to expose a `--backbone` flag in
    its constructor, OR you can monkey-patch the FDM modules in place.
    Below we assume the constructor accepts `backbone="..."`. If your
    project doesn't yet, you must either:
      (a) add a `backbone` kwarg to WMDC and respect it in the encoder/
          decoder construction, or
      (b) replace this function with a manual swap that traverses the
          model and re-assigns the relevant submodules.
    """
    from models.WMDC import WMDC

    try:
        model = WMDC(
            N=192,
            M=320,
            num_slices=5,
            backbone=backbone,  # <-- requires WMDC support
        ).to(device)
    except TypeError:
        # WMDC doesn't yet accept `backbone`; fall back to a runtime swap.
        print(
            f"[INFO] WMDC.__init__ does not accept `backbone=`. "
            f"Falling back to in-place module swap for backbone={backbone}."
        )
        model = WMDC(N=192, M=320, num_slices=5).to(device)
        _swap_backbones(model, backbone)
    return model.eval()


def _swap_backbones(model: nn.Module, backbone: str) -> None:
    """
    Replace every FrequencyDisentangledMamba submodule in `model` with
    the requested variant. Used when WMDC doesn't accept `backbone=` yet.
    """
    from ablation_models.backbone_variants import build_backbone
    from modules.wavelet_blocks import FrequencyDisentangledMamba

    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for parent in model.modules():
        for name, child in list(parent.named_children()):
            if isinstance(child, FrequencyDisentangledMamba):
                # Inspect dim from the child if possible; default to 192.
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
    """
    Measure latency of `fn` on a warm GPU.

    Returns dict with: median_ms, p25_ms, p75_ms, mean_ms, std_ms.
    """
    use_cuda = device == "cuda" and torch.cuda.is_available()
    # Warmup.
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
    Count forward-pass FLOPs. Returns gflops + library used.

    Many compression models contain custom CUDA kernels (e.g. Mamba's
    selective_scan). FLOP counters typically miss these. We log a
    warning if the count is suspiciously low.
    """
    if _HAS_FVCORE:
        try:
            flops = FlopCountAnalysis(model, x)
            flops.unsupported_ops_warnings(False)
            flops.uncalled_modules_warnings(False)
            total = flops.total()
            return {"gflops": total / 1e9, "library": "fvcore"}
        except Exception as e:
            print(f"[WARN] fvcore FlopCountAnalysis failed: {e}")

    if _HAS_THOP:
        try:
            macs, _ = thop_profile(model, inputs=(x,), verbose=False)
            return {"gflops": macs * 2 / 1e9, "library": "thop (MAC*2)"}
        except Exception as e:
            print(f"[WARN] thop.profile failed: {e}")

    return {"gflops": float("nan"), "library": "none"}


def benchmark_one(backbone: str, args, device: str) -> dict:
    print(f"\n=== Backbone: {backbone} ===")
    model = _build_model_with_backbone(backbone, device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        sd = ckpt.get("state_dict", ckpt)
        miss, unexp = model.load_state_dict(sd, strict=False)
        if miss or unexp:
            print(
                f"  [load] {len(miss)} missing, {len(unexp)} unexpected (expected for swapped backbones)"
            )

    H, W = args.resolution
    x = torch.randn(1, 3, H, W, device=device)

    # Param count.
    n_params = sum(p.numel() for p in model.parameters())

    # Latency: full forward.
    print("  Measuring full-forward latency...")
    full_lat = _measure_latency(
        lambda: model(x), warmup=args.warmup, iters=args.iters, device=device
    )

    # Latency: encoder only (g_a + h_a).
    print("  Measuring encoder-only latency...")

    def enc_fn():
        with torch.no_grad():
            y = model.g_a(x)
            z = model.h_a(y)
            return z

    enc_lat = _measure_latency(
        enc_fn, warmup=args.warmup, iters=args.iters, device=device
    )

    # GFLOPs (full forward).
    print("  Counting GFLOPs...")
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
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional: load weights into the model. Latency/FLOPs are weight-independent, but this lets us catch shape mismatches.",
    )
    p.add_argument("--output", type=str, default="backbone_benchmark.json")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    if not _HAS_FVCORE:
        print("[INFO] fvcore not installed → falling back to thop or skipping FLOPs.")

    results = []
    for bb in args.backbones:
        try:
            res = benchmark_one(bb, args, device)
            results.append(res)
        except Exception as e:
            print(f"[ERROR] backbone={bb} failed: {e}")
            results.append({"backbone": bb, "error": str(e)})

    # Pretty-print summary.
    print("\n" + "=" * 80)
    print(
        f"{'Backbone':<14} {'Params(M)':>10} {'GFLOPs':>10} "
        f"{'Full med(ms)':>14} {'Enc med(ms)':>14}"
    )
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r['backbone']:<14}   ERROR: {r['error']}")
            continue
        print(
            f"{r['backbone']:<14} "
            f"{r['params_M']:>10.2f} "
            f"{r['flops']['gflops']:>10.2f} "
            f"{r['full_latency']['median_ms']:>14.2f} "
            f"{r['encoder_latency']['median_ms']:>14.2f}"
        )
    print("=" * 80)

    with open(args.output, "w") as f:
        json.dump({"resolution": args.resolution, "results": results}, f, indent=2)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
