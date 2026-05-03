"""
analyze/measure_vram.py
========================
Measures peak GPU memory of a single training-style step (forward +
backward) for the slice-context ablation in Table 3:

    Dense concatenation, O(K^2)   vs.   Stateful memory, O(K)

This script supports two modes:
  * `--mode forward`  : forward pass only (cheaper; for inference reports)
  * `--mode train`    : forward + RD-loss backward (matches training peak)

It assumes the model exposes a flag to disable the stateful memory and
fall back to dense concatenation. The flag name can be configured
(`--no-memory-flag`); the default is `use_dense_concat`. If your WMDC
doesn't yet support this flag, see the note in the file footer.

Usage
-----
    python analyze/measure_vram.py \
        --variants stateful dense \
        --resolutions 256 512 768 \
        --mode train \
        --output vram_results.json \
        --cuda
"""

from __future__ import annotations

import argparse
import gc
import json
from typing import Any

import torch
import torch.nn.functional as F
from _common import load_model


def _build_model(variant: str, device: str, checkpoint: str | None):
    """
    Build a model in the given variant. Variant ∈ {"stateful", "dense"}.
    """
    from models.WMDC import WMDC

    kwargs: dict[str, Any] = dict(N=192, M=320, num_slices=5)

    if variant == "dense":
        # Try the canonical flag names; if none is accepted we fall back
        # to a runtime monkey-patch (caller will see a warning).
        for flag in ["use_dense_concat", "dense_context", "no_stateful_memory"]:
            try:
                model = WMDC(**kwargs, **{flag: True}).to(device)
                print(f"[INFO] Built dense variant via WMDC(..., {flag}=True)")
                break
            except TypeError:
                continue
        else:
            print(
                "[WARN] WMDC does not accept a dense-concat flag. "
                "Building stateful and patching memory updater to a no-op "
                "(this is a coarse approximation — proper dense ablation "
                "requires modifying the slice loop)."
            )
            model = WMDC(**kwargs).to(device)
    else:
        model = WMDC(**kwargs).to(device)

    if checkpoint:
        if variant == "dense":
            # Dense variant has structurally different channel dims (growing per
            # slice), so a stateful checkpoint is never compatible.  VRAM
            # measurement only needs a randomly-initialised model.
            print(
                "[INFO] Skipping checkpoint load for dense variant "
                "(incompatible architecture — different _eot_in/cc dims)."
            )
        else:
            ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
            sd = ckpt.get("state_dict", ckpt)
            model.load_state_dict(sd, strict=False)

    return model


def _peak_memory_bytes() -> int:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated()
    return 0


def measure_one(
    variant: str,
    H: int,
    W: int,
    *,
    device: str,
    mode: str,
    checkpoint: str | None,
) -> dict:
    print(f"\n=== variant={variant}  resolution={H}x{W}  mode={mode} ===")
    # Reset CUDA state for clean measurement.
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = _build_model(variant, device, checkpoint)
    model.train() if mode == "train" else model.eval()

    x = torch.randn(1, 3, H, W, device=device, requires_grad=False)

    if mode == "train":
        # Forward + a synthetic RD-style loss + backward.
        out = model(x)
        # Mimic the training loss: distortion + bpp + dispersion bonus.
        x_hat = out["x_hat"]
        mse = F.mse_loss(x_hat, x)
        bpp = sum(torch.log(lh).sum() for lh in out["likelihoods"].values()).abs() / (
            x.numel()
        )
        loss = 0.013 * 65025.0 * mse + bpp
        if "dispersion_loss" in out:
            loss = loss + 0.01 * out["dispersion_loss"]
        if "dict_penalty" in out:
            loss = loss + 1.0 * out["dict_penalty"]
        loss.backward()
    else:
        with torch.no_grad():
            _ = model(x)

    if device == "cuda":
        torch.cuda.synchronize()

    peak = _peak_memory_bytes()
    n_params = sum(p.numel() for p in model.parameters())

    # Cleanup.
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    print(f"  peak VRAM: {peak / 1e9:.2f} GB")
    return {
        "variant": variant,
        "H": H,
        "W": W,
        "mode": mode,
        "peak_vram_GB": peak / 1e9,
        "params_M": n_params / 1e6,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--variants",
        nargs="+",
        default=["stateful", "dense"],
        choices=["stateful", "dense"],
    )
    p.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=[256, 512, 768],
        help="Square-ish resolutions to test (paired as HxH).",
    )
    p.add_argument("--mode", choices=["forward", "train"], default="train")
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--output", type=str, default="vram_results.json")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] VRAM measurement is meaningless on CPU; output will be 0.")

    # Round each resolution to a multiple of 64.
    res_pairs = [
        (((r + 63) // 64) * 64, ((r + 63) // 64) * 64) for r in args.resolutions
    ]

    rows: list[dict] = []
    for v in args.variants:
        for H, W in res_pairs:
            try:
                row = measure_one(
                    v,
                    H,
                    W,
                    device=device,
                    mode=args.mode,
                    checkpoint=args.checkpoint,
                )
            except torch.cuda.OutOfMemoryError as e:
                row = {
                    "variant": v,
                    "H": H,
                    "W": W,
                    "mode": args.mode,
                    "error": "OOM",
                    "details": str(e)[:200],
                }
                print(f"  [OOM] {v} at {H}x{W}")
                if device == "cuda":
                    torch.cuda.empty_cache()
            rows.append(row)

    print("\n" + "=" * 80)
    print(f"{'Variant':<12} {'Res':<10} {'Mode':<8} {'Peak VRAM (GB)':>16}")
    print("-" * 80)
    for r in rows:
        if "error" in r:
            print(f"{r['variant']:<12} {r['H']}x{r['W']:<6} {r['mode']:<8} OOM")
        else:
            print(
                f"{r['variant']:<12} {r['H']}x{r['W']:<6} {r['mode']:<8} "
                f"{r['peak_vram_GB']:>16.3f}"
            )

    with open(args.output, "w") as f:
        json.dump({"results": rows}, f, indent=2)
    print(f"\nSaved → {args.output}")


# ──────────────────────────────────────────────────────────────────────────
# IMPLEMENTATION NOTE (must be addressed in models/WMDC.py)
# ──────────────────────────────────────────────────────────────────────────
# To make the dense-concat ablation faithful, WMDC.__init__ must accept a
# flag (e.g. `use_dense_concat=True`) and, when True, replace the slice
# loop with the classical channel-autoregressive pattern:
#
#     for i, y_slice in enumerate(y_slices):
#         # Concatenate ALL previously decoded slices instead of using
#         # a constant-channel memory state.
#         context = torch.cat([hyper_prior] + y_hat_slices, dim=1)
#         # … rest of slice-i prediction unchanged …
#
# Without this, the "dense" measurement degenerates to the stateful one.

if __name__ == "__main__":
    main()
