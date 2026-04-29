"""
analyze/compute_bd_rate.py
==========================
Evaluate a list of WMDC checkpoints (one per λ) on a dataset, then compute
BD-rate against an anchor RD curve (e.g. VTM, BPG, or another model
variant). Produces:

    * RD JSON per variant: list of (bpp, psnr) pairs over the dataset.
    * BD-rate vs. anchor (PSNR-bpp, akima interpolation).
    * Optional: a comparison PDF plot.

This is THE script that fills BD-rate columns in every paper table.

Required Python package
-----------------------
    pip install bjontegaard

Usage
-----
    # Evaluate one variant ("full" model) over six lambdas:
    python analyze/compute_bd_rate.py evaluate \
        --variant-name        full \
        --checkpoints         ckpts/full/lam_0.0018/best.pth.tar \
                              ckpts/full/lam_0.0035/best.pth.tar \
                              ckpts/full/lam_0.0067/best.pth.tar \
                              ckpts/full/lam_0.013/best.pth.tar \
                              ckpts/full/lam_0.025/best.pth.tar \
                              ckpts/full/lam_0.0483/best.pth.tar \
        --dataset             /data/kodak \
        --output              rd_results/full.json \
        --cuda

    # Compute BD-rate of "full" against VTM anchor JSON:
    python analyze/compute_bd_rate.py bd-rate \
        --anchor-json   anchors/vtm_kodak.json \
        --variant-jsons rd_results/full.json rd_results/no_disp.json \
        --output        bd_rate_table.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import numpy as np
import torch

from _common import evaluate_codec, iter_dataset, load_model

# bjontegaard is the standard library used in compression papers.
try:
    import bjontegaard as bd
except ImportError:
    bd = None
    print(
        "[WARN] `bjontegaard` not installed; BD-rate computation will fail. "
        "Install with: pip install bjontegaard",
        file=sys.stderr,
    )


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: evaluate
# ──────────────────────────────────────────────────────────────────────────
def cmd_evaluate(args):
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    if not args.checkpoints:
        raise SystemExit("No checkpoints provided.")

    rd_points: list[dict] = []
    per_image_records: list[dict] = []

    for ckpt_path in args.checkpoints:
        print(f"\n== Evaluating {ckpt_path} ==")
        model = load_model(
            ckpt_path,
            device,
            routing_mode=args.routing_mode,
            ot_eps=args.ot_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            update_for_inference=True,
            strict_load=False,
        )

        psnrs: list[float] = []
        bpps: list[float] = []
        for path, x, x_padded, H, W in iter_dataset(args.dataset, device=device):
            res = evaluate_codec(
                model, x, x_padded, H, W, use_actual_codec=not args.estimated_bpp
            )
            psnrs.append(res["psnr"])
            bpps.append(res["bpp"])
            per_image_records.append({
                "checkpoint": ckpt_path,
                "image": os.path.basename(path),
                "psnr": res["psnr"],
                "bpp": res["bpp"],
                "mode": res["mode"],
            })
            print(f"  {os.path.basename(path):>30s}: psnr={res['psnr']:.3f}  bpp={res['bpp']:.4f}")

        mean_psnr = float(np.mean(psnrs))
        mean_bpp = float(np.mean(bpps))
        rd_points.append({
            "checkpoint": ckpt_path,
            "psnr": mean_psnr,
            "bpp": mean_bpp,
            "n_images": len(psnrs),
        })
        print(f"  → dataset mean: psnr={mean_psnr:.3f}  bpp={mean_bpp:.4f}")

        del model
        torch.cuda.empty_cache() if device == "cuda" else None

    # Sort RD points by bpp ascending (required for BD-rate interpolation).
    rd_points.sort(key=lambda r: r["bpp"])

    out = {
        "variant": args.variant_name,
        "dataset": args.dataset,
        "rd_points": rd_points,
        "per_image": per_image_records if args.dump_per_image else None,
        "settings": {
            "routing_mode": args.routing_mode,
            "ot_eps": args.ot_eps,
            "sinkhorn_iters": args.sinkhorn_iters,
            "bpp_mode": "estimated" if args.estimated_bpp else "actual",
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved RD points → {args.output}")


# ──────────────────────────────────────────────────────────────────────────
# Subcommand: bd-rate
# ──────────────────────────────────────────────────────────────────────────
def _extract_curve(rd_dict: dict) -> tuple[list[float], list[float]]:
    """Returns (bpps, psnrs) sorted by bpp."""
    pts = rd_dict["rd_points"]
    pts_sorted = sorted(pts, key=lambda r: r["bpp"])
    bpps = [r["bpp"] for r in pts_sorted]
    psnrs = [r["psnr"] for r in pts_sorted]
    return bpps, psnrs


def cmd_bd_rate(args):
    if bd is None:
        raise SystemExit("`bjontegaard` package not available.")

    with open(args.anchor_json) as f:
        anchor = json.load(f)
    anchor_bpp, anchor_psnr = _extract_curve(anchor)
    anchor_name = anchor.get("variant", os.path.basename(args.anchor_json))

    print(f"=== BD-rate vs. anchor: {anchor_name} ===")
    print(f"Anchor curve: {len(anchor_bpp)} points, bpp range [{min(anchor_bpp):.3f}, {max(anchor_bpp):.3f}]")

    table_rows: list[dict] = []
    for var_path in args.variant_jsons:
        with open(var_path) as f:
            var = json.load(f)
        v_bpp, v_psnr = _extract_curve(var)
        v_name = var.get("variant", os.path.basename(var_path))

        try:
            bd_rate = bd.bd_rate(anchor_bpp, anchor_psnr, v_bpp, v_psnr, args.method)
            bd_psnr = bd.bd_psnr(anchor_bpp, anchor_psnr, v_bpp, v_psnr, args.method)
        except Exception as e:
            print(f"  [{v_name}] BD-rate failed: {e}")
            bd_rate = float("nan")
            bd_psnr = float("nan")

        row = {
            "variant": v_name,
            "anchor": anchor_name,
            "bd_rate_pct": bd_rate,
            "bd_psnr_db": bd_psnr,
            "n_points": len(v_bpp),
        }
        table_rows.append(row)
        print(f"  {v_name:>20s}: BD-rate = {bd_rate:+.3f}%   BD-PSNR = {bd_psnr:+.3f} dB")

    out = {"anchor": anchor_name, "method": args.method, "rows": table_rows}
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved BD-rate table → {args.output}")


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate checkpoints → RD JSON")
    p_eval.add_argument("--variant-name", type=str, required=True)
    p_eval.add_argument("--checkpoints", nargs="+", required=True)
    p_eval.add_argument("--dataset", type=str, required=True)
    p_eval.add_argument("--output", type=str, required=True)
    p_eval.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p_eval.add_argument("--ot-eps", type=float, default=0.1)
    p_eval.add_argument("--sinkhorn-iters", type=int, default=20)
    p_eval.add_argument(
        "--estimated-bpp",
        action="store_true",
        help="Use forward()-derived bpp instead of actual codec. NOT for paper tables.",
    )
    p_eval.add_argument("--dump-per-image", action="store_true")
    p_eval.add_argument("--cuda", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    # bd-rate
    p_bd = sub.add_parser("bd-rate", help="Compute BD-rate variant(s) vs. anchor")
    p_bd.add_argument("--anchor-json", type=str, required=True)
    p_bd.add_argument("--variant-jsons", nargs="+", required=True)
    p_bd.add_argument("--output", type=str, required=True)
    p_bd.add_argument(
        "--method",
        type=str,
        default="akima",
        choices=["akima", "pchip", "cubic"],
        help="Interpolation method for BD-rate computation.",
    )
    p_bd.set_defaults(func=cmd_bd_rate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
