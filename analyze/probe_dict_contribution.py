"""
analyze/probe_dict_contribution.py
==================================
Quantify how much the dictionary attention output actually contributes to
the entropy model's conditioning.

Motivation
----------
The eval audit on the pre-fix checkpoint (Kodak, lambda=0.0036) found
column-marginal entropy stuck at ~93 % of log2(N) — routing was near-uniform
across all 5 slices.  A near-uniform routing produces a near-constant
dictionary side-information vector, which means the entropy model receives
no useful per-pixel conditioning from the dictionary branch.

This script verifies that suspicion by running every Kodak image twice:

  1. Normal forward    → BPP_normal,  PSNR_normal
  2. Dictionary-zeroed → BPP_zero,    PSNR_zero
     (a forward hook replaces every UnifiedDictionaryAttention output with
     a zero tensor of the same shape; the rest of the model is untouched.)

If the dictionary is doing meaningful work, zeroing it should INCREASE bpp
(entropy model loses useful context) and DECREASE PSNR (decoder loses
content-adaptive features).  If ΔBPP < 0.005 and |ΔPSNR| < 0.05 dB, the
branch is effectively dead — a strong signal that the routing collapsed
to uniform and the loss did not converge to a useful encoder.

Usage
-----
    python analyze/probe_dict_contribution.py \\
        --dataset /path/to/kodak \\
        --checkpoint /path/to/checkpoint_best.pth.tar \\
        --routing-mode unbalanced_eot \\
        --content-adaptive --cluster-num 8 \\
        --use-wls-shortcut \\
        --output results/dict_contribution.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze._common import (  # noqa: E402
    compute_actual_bpp,
    compute_psnr,
    iter_dataset,
    load_model,
)
from modules.dictionary_blocks import UnifiedDictionaryAttention  # noqa: E402


def _install_zero_hooks(model) -> list:
    """
    Register a forward hook on every UnifiedDictionaryAttention that
    replaces the (out, aux) return tuple's `out` tensor with zeros.

    Returns the list of handles so callers can remove them.
    """

    def make_zero_hook():
        def hook(_module, _inputs, output):
            out, aux = output
            return torch.zeros_like(out), aux

        return hook

    handles = []
    for m in model.modules():
        if isinstance(m, UnifiedDictionaryAttention):
            handles.append(m.register_forward_hook(make_zero_hook()))
    return handles


@torch.no_grad()
def _measure(model, x_padded: torch.Tensor, H: int, W: int) -> Tuple[float, float]:
    """Returns (real_bpp, psnr) on the un-padded HxW image."""
    out_enc = model.compress(x_padded)
    out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
    x_hat = out_dec["x_hat"][..., :H, :W].clamp(0, 1)
    x_orig = x_padded[..., :H, :W]
    return compute_actual_bpp(out_enc["strings"], H * W), compute_psnr(x_orig, x_hat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--routing-mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--backbone",
        type=str,
        default="fdm",
        choices=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
    )
    parser.add_argument("--use-dense-concat", action="store_true")
    parser.add_argument(
        "--memory-init", type=str, default="bootstrap", choices=["bootstrap", "zero"]
    )
    parser.add_argument("--content-adaptive", action="store_true")
    parser.add_argument("--cluster-num", type=int, default=8)
    parser.add_argument("--use-wls-shortcut", action="store_true")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = load_model(
        args.checkpoint,
        device=device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        backbone=args.backbone,
        use_dense_concat=args.use_dense_concat,
        memory_init=args.memory_init,
        use_content_adaptive=args.content_adaptive,
        cluster_num=args.cluster_num,
        use_wls_shortcut=args.use_wls_shortcut,
        strict_load=False,
    )

    per_image: list[dict] = []
    sum_bpp_norm = sum_psnr_norm = sum_bpp_zero = sum_psnr_zero = 0.0
    n = 0

    for path, _x, x_padded, H, W in iter_dataset(args.dataset, device=device):
        bpp_n, psnr_n = _measure(model, x_padded, H, W)
        handles = _install_zero_hooks(model)
        try:
            bpp_z, psnr_z = _measure(model, x_padded, H, W)
        finally:
            for h in handles:
                h.remove()

        per_image.append(
            {
                "file": os.path.basename(path),
                "resolution": f"{W}x{H}",
                "bpp_normal": round(bpp_n, 4),
                "psnr_normal": round(psnr_n, 2),
                "bpp_zero_dict": round(bpp_z, 4),
                "psnr_zero_dict": round(psnr_z, 2),
                "delta_bpp": round(bpp_z - bpp_n, 4),
                "delta_psnr": round(psnr_z - psnr_n, 2),
            }
        )
        sum_bpp_norm += bpp_n
        sum_psnr_norm += psnr_n
        sum_bpp_zero += bpp_z
        sum_psnr_zero += psnr_z
        n += 1
        print(
            f"{os.path.basename(path):>14s} "
            f"| normal: {bpp_n:.4f} bpp / {psnr_n:.2f} dB "
            f"| zeroed: {bpp_z:.4f} bpp / {psnr_z:.2f} dB "
            f"| Δbpp {bpp_z - bpp_n:+.4f} | ΔPSNR {psnr_z - psnr_n:+.2f}"
        )

    avg = {
        "bpp_normal": round(sum_bpp_norm / n, 4),
        "psnr_normal": round(sum_psnr_norm / n, 2),
        "bpp_zero_dict": round(sum_bpp_zero / n, 4),
        "psnr_zero_dict": round(sum_psnr_zero / n, 2),
        "delta_bpp": round((sum_bpp_zero - sum_bpp_norm) / n, 4),
        "delta_psnr": round((sum_psnr_zero - sum_psnr_norm) / n, 2),
    }

    # Heuristic verdict.  Thresholds calibrated against the typical
    # contribution a single side-info branch makes in CompressAI baselines:
    # >5 % bpp swing or >0.2 dB PSNR swing is "alive"; tiny swings indicate
    # the branch contributes nothing.
    bpp_swing = (avg["delta_bpp"] / max(avg["bpp_normal"], 1e-6)) * 100.0
    if abs(bpp_swing) < 1.0 and abs(avg["delta_psnr"]) < 0.05:
        verdict = "DEAD"
    elif abs(bpp_swing) < 5.0 and abs(avg["delta_psnr"]) < 0.2:
        verdict = "WEAK"
    else:
        verdict = "ALIVE"

    report = {
        "verdict": verdict,
        "bpp_swing_pct": round(bpp_swing, 2),
        "average": avg,
        "per_image": per_image,
        "args": vars(args),
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=4)

    print("\n── Verdict ──────────────────────────────────────")
    print(f"  bpp swing       : {bpp_swing:+.2f} %")
    print(f"  ΔPSNR           : {avg['delta_psnr']:+.2f} dB")
    print(f"  dictionary is   : {verdict}")
    print(f"  report saved to : {args.output}")


if __name__ == "__main__":
    main()
