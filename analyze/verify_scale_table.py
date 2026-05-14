"""
analyze/verify_scale_table.py
=============================

Audit the alignment between (a) the Gaussian scales emitted by the
per-slice cc_scale_transforms at training/eval time and (b) the
entropy coder's scale_table built by `model.update()`.

Why this matters
----------------
A scale that lies OUTSIDE the table's [scale_min, scale_max] interval
gets clipped to the nearest table index when arithmetic encoded.  Any
mass that "should" have been distributed across larger-σ Gaussian bins
gets concentrated into the table's terminal bin → real bpp is inflated
relative to the rate estimate the optimiser saw during training.

The fix shipped in `models/WMDC.py` clamps scales to [0.11, 256] (the
same interval the scale_table now spans) so that, by construction, no
index saturation can happen.  This script verifies the fix empirically
on real data and quantifies the head-room left at each end.

Reported per slice
------------------
  • percentiles of the RAW pre-clamp scale (5 / 50 / 95 / 99 / max)
  • fraction of scales that WOULD have hit the min/max clamp
  • bpp gap induced by the clamp (always 0 once the clamp is wide
    enough; non-zero means the scale_table needs to be widened)

Usage
-----
    python analyze/verify_scale_table.py \
        --checkpoint  ./checkpoints/lambda_0.0018_mse/checkpoint_best.pth.tar \
        --dataset     /kaggle/input/datasets/kodak-test \
        --routing-mode unbalanced_eot \
        --max-images  8 \
        --output      scale_audit.json

Notes
-----
This script does NOT mutate the checkpoint; it only attaches forward
hooks to capture the raw (pre-clamp) scale tensors.  Running it is
read-only and safe at any point in the training cycle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from _common import BD_RATE_LAMBDAS_FULL, compute_actual_bpp, iter_dataset, load_model

# ---------------------------------------------------------------------------
# Hook plumbing
# ---------------------------------------------------------------------------


def _attach_scale_hooks(model) -> List[List[torch.Tensor]]:
    """
    Register a forward-hook on every cc_scale_transforms[i] that records
    the RAW (pre-clamp) scale tensor.  Returns a list-of-lists `captures`
    where captures[i] is repopulated each forward; caller drains and
    clears after every image.
    """
    captures: List[List[torch.Tensor]] = [[] for _ in model.cc_scale_transforms]

    def make_hook(slice_idx):
        def hook(_module, _inp, out):
            # Detach + cpu — we only need statistics, not gradient.
            captures[slice_idx].append(out.detach().float().cpu())

        return hook

    handles = []
    for i, m in enumerate(model.cc_scale_transforms):
        handles.append(m.register_forward_hook(make_hook(i)))

    return captures, handles


# ---------------------------------------------------------------------------
# Statistics per slice
# ---------------------------------------------------------------------------


def _summarise(
    flat_scales: np.ndarray, clamp_min: float, clamp_max: float
) -> Dict[str, float]:
    """
    Returns percentile + saturation stats for one slice's RAW scales.
    """
    q = np.quantile(flat_scales, [0.05, 0.50, 0.95, 0.99, 1.00])
    sat_min = float(np.mean(flat_scales < clamp_min))
    sat_max = float(np.mean(flat_scales > clamp_max))
    return {
        "p05": float(q[0]),
        "p50": float(q[1]),
        "p95": float(q[2]),
        "p99": float(q[3]),
        "max": float(q[4]),
        "min": float(flat_scales.min()),
        "frac_below_min": sat_min,
        "frac_above_max": sat_max,
        "n_samples": int(flat_scales.size),
    }


def _scale_table_bin_distribution(
    flat_scales: np.ndarray, scale_table: np.ndarray
) -> Dict[str, float]:
    """
    Map each scale to its closest scale_table index (log space, mirroring
    GaussianConditional.build_indexes), then return how the mass concentrates.

    Specifically:
      • frac_in_terminal_bin : mass that ended up in the LARGEST table index
                                — if this is non-trivial, the table is too
                                narrow on the upper side.
      • frac_in_zeroth_bin   : same on the small-σ end.
    """
    # log-space distance to each table value
    log_scale = np.log(np.clip(flat_scales, 1e-9, None))
    log_table = np.log(scale_table)
    # For each sample, find the nearest bin
    idx = np.abs(log_scale[:, None] - log_table[None, :]).argmin(axis=1)
    n = len(scale_table)
    return {
        "frac_in_zeroth_bin": float(np.mean(idx == 0)),
        "frac_in_terminal_bin": float(np.mean(idx == n - 1)),
        "p99_idx": float(np.quantile(idx, 0.99)),
        "max_idx_used": int(idx.max()),
        "table_len": n,
    }


# ---------------------------------------------------------------------------
# Main pass
# ---------------------------------------------------------------------------


def audit(args) -> dict:
    device = "cuda" if (args.cuda and torch.cuda.is_available()) else "cpu"

    model = load_model(
        args.checkpoint,
        device=device,
        routing_mode=args.routing_mode,
        backbone=args.backbone,
        use_dense_concat=args.use_dense_concat,
        memory_init=args.memory_init,
        strict_load=False,
    )

    # ---- snapshot the scale_table built by update() ----------------------
    # GaussianConditional stores it as a buffer named 'scale_table'.
    if not hasattr(model.gaussian_conditional, "scale_table"):
        raise RuntimeError(
            "GaussianConditional has no scale_table buffer.  "
            "Did model.update(force=True) run?"
        )
    scale_table = model.gaussian_conditional.scale_table.detach().cpu().numpy()
    scale_min_table = float(scale_table.min())
    scale_max_table = float(scale_table.max())

    # ---- the clamp the model actually enforces ---------------------------
    # MUST match models/WMDC.py:forward/compress/decompress.
    clamp_min = 0.11
    clamp_max = 256.0
    if not (
        clamp_min >= scale_min_table - 1e-6 and clamp_max <= scale_max_table + 1e-6
    ):
        print(
            f"[warn] clamp [{clamp_min}, {clamp_max}] not contained in "
            f"scale_table [{scale_min_table:.4f}, {scale_max_table:.4f}]"
        )

    # ---- hook ------------------------------------------------------------
    captures, handles = _attach_scale_hooks(model)

    # Per-slice running pool of scales (capped to avoid OOM on big datasets)
    pool: Dict[int, List[np.ndarray]] = defaultdict(list)
    pool_cap_per_slice = args.max_samples_per_slice

    # Per-image real-bpp vs est-bpp gap (to surface miscalibration)
    rd_records: List[dict] = []

    n_done = 0
    for img_path, x, x_padded, H, W in iter_dataset(args.dataset, device=device):
        if args.max_images > 0 and n_done >= args.max_images:
            break
        n_done += 1

        # Clear captures from previous image
        for slc in captures:
            slc.clear()

        with torch.no_grad():
            out_net = model(x_padded)

        # ---- drain captured raw scales ----------------------------------
        for i, slc_caps in enumerate(captures):
            for t in slc_caps:
                arr = t.numpy().ravel()
                if pool_cap_per_slice > 0 and len(pool[i]) > 0:
                    cur = sum(a.size for a in pool[i])
                    room = pool_cap_per_slice - cur
                    if room <= 0:
                        continue
                    if arr.size > room:
                        arr = arr[:room]
                pool[i].append(arr)

        # ---- est vs real bpp on the same image --------------------------
        num_pixels = H * W
        est_bpp = sum(
            torch.log(lh).sum() for lh in out_net["likelihoods"].values()
        ).item() / (-math.log(2) * num_pixels)

        out_enc = model.compress(x_padded)
        real_bpp = compute_actual_bpp(out_enc["strings"], num_pixels)
        rd_records.append(
            {
                "file": os.path.basename(img_path),
                "est_bpp": round(est_bpp, 5),
                "real_bpp": round(real_bpp, 5),
                "gap": round(real_bpp - est_bpp, 5),
            }
        )

    for h in handles:
        h.remove()

    # ---- aggregate -------------------------------------------------------
    per_slice = []
    for i in sorted(pool.keys()):
        flat = np.concatenate(pool[i]) if pool[i] else np.array([])
        if flat.size == 0:
            continue
        stats = _summarise(flat, clamp_min, clamp_max)
        bins = _scale_table_bin_distribution(flat, scale_table)
        per_slice.append({"slice": i, **stats, **bins})

    # global est-vs-real gap stats
    if rd_records:
        gaps = np.array([r["gap"] for r in rd_records])
        global_gap = {
            "n_images": len(rd_records),
            "mean_gap_bpp": float(gaps.mean()),
            "std_gap_bpp": float(gaps.std()),
            "max_gap_bpp": float(gaps.max()),
            "min_gap_bpp": float(gaps.min()),
        }
    else:
        global_gap = {"n_images": 0}

    return {
        "scale_table": {
            "min": scale_min_table,
            "max": scale_max_table,
            "len": int(scale_table.size),
        },
        "model_clamp": {"min": clamp_min, "max": clamp_max},
        "per_slice": per_slice,
        "bpp_gap": global_gap,
        "per_image_bpp": rd_records,
        "args": {
            k: v
            for k, v in vars(args).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        },
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def _print_report(report: dict) -> None:
    st = report["scale_table"]
    clamp = report["model_clamp"]
    print("\n== Scale-table audit ============================================")
    print(f"  scale_table : [{st['min']:.4f}, {st['max']:.4f}]  ({st['len']} bins)")
    print(f"  model clamp : [{clamp['min']:.4f}, {clamp['max']:.4f}]")
    if st["max"] < clamp["max"] - 1e-6:
        print("  [warn] scale_table is NARROWER than the model clamp — widen update().")

    print("\n  Per-slice raw-scale percentiles + saturation:")
    print(
        f"    {'slc':>3} {'p05':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>10}"
        f"  {'<clip_min':>10} {'>clip_max':>10}  {'in_term_bin':>12}"
    )
    for row in report["per_slice"]:
        print(
            f"    {row['slice']:>3} "
            f"{row['p05']:>8.3f} {row['p50']:>8.3f} {row['p95']:>8.3f} "
            f"{row['p99']:>8.3f} {row['max']:>10.3f}  "
            f"{row['frac_below_min']*100:>9.3f}% "
            f"{row['frac_above_max']*100:>9.3f}%  "
            f"{row['frac_in_terminal_bin']*100:>11.3f}%"
        )

    gap = report["bpp_gap"]
    if gap.get("n_images", 0) > 0:
        print(
            f"\n  Estimated-vs-real bpp gap ({gap['n_images']} images):"
            f"  mean={gap['mean_gap_bpp']:+.5f}  "
            f"max={gap['max_gap_bpp']:+.5f}  "
            f"min={gap['min_gap_bpp']:+.5f}"
        )
        if abs(gap["mean_gap_bpp"]) > 0.005:
            print(
                "  [warn] non-trivial bpp gap — investigate (likelihood "
                "miscalibration, table saturation, or train/infer mismatch)."
            )
        else:
            print("  [OK] gap < 0.005 bpp — scale_table covers the operating range.")

    # Verdict
    any_top_saturated = any(
        row["frac_above_max"] > 0.0 or row["frac_in_terminal_bin"] > 0.001
        for row in report["per_slice"]
    )
    any_bot_saturated = any(
        row["frac_below_min"] > 0.0 or row["frac_in_zeroth_bin"] > 0.05
        for row in report["per_slice"]
    )
    print("\n  Verdict:")
    if not any_top_saturated and not any_bot_saturated:
        print("    [OK] No saturation at either end.  Scale-table fix is effective.")
    else:
        if any_top_saturated:
            print(
                "    [FAIL] Some slices hit the upper clamp/terminal bin — "
                "widen scale_table.max (currently 256)."
            )
        if any_bot_saturated:
            print(
                "    [FAIL] Some slices hit the lower clamp/zeroth bin — "
                "lower scale_table.min (currently 0.11)."
            )
    print("================================================================\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, help="Folder of test images.")
    p.add_argument("--output", default="scale_audit.json")
    p.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Cap number of images to audit (0 = all).",
    )
    p.add_argument(
        "--max-samples-per-slice",
        type=int,
        default=2_000_000,
        help="Cap pooled scale samples per slice to avoid OOM.  0 = unlimited.",
    )
    p.add_argument(
        "--routing-mode",
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    p.add_argument(
        "--backbone",
        default="fdm",
        choices=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
    )
    p.add_argument("--use-dense-concat", action="store_true")
    p.add_argument("--memory-init", default="bootstrap", choices=["bootstrap", "zero"])
    p.add_argument("--cuda", action="store_true", default=True)
    return p.parse_args()


def main():
    args = parse_args()
    report = audit(args)
    _print_report(report)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full JSON report → {args.output}")


if __name__ == "__main__":
    main()
