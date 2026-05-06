"""
analyze/verify_train_inference_consistency.py
==============================================
Verifies that the model's train-time forward pass (`model(x)`) produces
the same PSNR and approximately the same BPP as the actual codec
round-trip (`model.compress` → `model.decompress`). The paper claims
"encoder/decoder remain bit-exact" (Section 3.4) — this is the script
that produces the supporting evidence.

What it measures
----------------
For each test image and each loaded checkpoint:
    psnr_train, bpp_train_estimate    ← model(x)
    psnr_codec, bpp_codec_actual      ← model.compress + decompress

    Δ_psnr = psnr_train − psnr_codec  (should be ≈ 0)
    Δ_bpp  = bpp_train  − bpp_codec   (small negative — header overhead)

Red flags
---------
* |Δ_psnr| > 0.05 dB             → encoder-decoder are not bit-exact
* Δ_bpp     > 0.005 bpp positive → bpp estimate is too optimistic
                                     (uniform-noise relaxation gap leaking
                                     into the reported number)

Usage
-----
    python analyze/verify_train_inference_consistency.py \
        --checkpoints ckpts/lam_0.013/best.pth.tar \
                      ckpts/lam_0.025/best.pth.tar \
        --dataset     /data/kodak \
        --output      consistency_report.json \
        --cuda
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import compute_psnr, evaluate_codec, iter_dataset, load_model


def evaluate_one(model, dataset_dir: str, device: str) -> List[dict]:
    rows: list[dict] = []
    for path, x, x_padded, H, W in iter_dataset(dataset_dir, device=device):
        # Train-style forward.
        train_res = evaluate_codec(model, x, x_padded, H, W, use_actual_codec=False)
        # Codec round-trip.
        codec_res = evaluate_codec(model, x, x_padded, H, W, use_actual_codec=True)
        rows.append(
            {
                "image": os.path.basename(path),
                "psnr_train": train_res["psnr"],
                "psnr_codec": codec_res["psnr"],
                "bpp_train": train_res["bpp"],
                "bpp_codec": codec_res["bpp"],
                "delta_psnr": train_res["psnr"] - codec_res["psnr"],
                "delta_bpp": train_res["bpp"] - codec_res["bpp"],
            }
        )
        print(
            f"  {os.path.basename(path):>30s}: "
            f"Δpsnr = {rows[-1]['delta_psnr']:+.4f} dB  "
            f"Δbpp = {rows[-1]['delta_bpp']:+.5f}"
        )
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--output", type=str, default="consistency_report.json")
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional PDF path for a Δ histogram per checkpoint.",
    )
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    all_results = []
    for ckpt in args.checkpoints:
        print(f"\n== Checkpoint: {ckpt} ==")
        model = load_model(
            ckpt,
            device,
            routing_mode=args.routing_mode,
            ot_eps=args.ot_eps,
            sinkhorn_iters=args.sinkhorn_iters,
            update_for_inference=True,
        )
        rows = evaluate_one(model, args.dataset, device)

        delta_psnr = [r["delta_psnr"] for r in rows]
        delta_bpp = [r["delta_bpp"] for r in rows]

        summary = {
            "checkpoint": ckpt,
            "n_images": len(rows),
            "delta_psnr": {
                "mean": float(np.mean(delta_psnr)),
                "std": float(np.std(delta_psnr)),
                "max_abs": float(np.max(np.abs(delta_psnr))),
            },
            "delta_bpp": {
                "mean": float(np.mean(delta_bpp)),
                "std": float(np.std(delta_bpp)),
                "max_abs": float(np.max(np.abs(delta_bpp))),
            },
            "rows": rows,
        }
        all_results.append(summary)

        # Verdict.
        red_flag = []
        if summary["delta_psnr"]["max_abs"] > 0.05:
            red_flag.append(
                f"|Δpsnr|_max = {summary['delta_psnr']['max_abs']:.4f} > 0.05 dB"
            )
        if summary["delta_bpp"]["mean"] > 0.005:
            red_flag.append(f"Δbpp mean = {summary['delta_bpp']['mean']:.5f} > 0.005")
        verdict = "🟢 OK" if not red_flag else "🔴 RED FLAG: " + "; ".join(red_flag)
        summary["verdict"] = verdict
        print(
            f"  Summary: Δpsnr={summary['delta_psnr']['mean']:+.4f}±{summary['delta_psnr']['std']:.4f} dB, "
            f"Δbpp={summary['delta_bpp']['mean']:+.5f}±{summary['delta_bpp']['std']:.5f}"
        )
        print(f"  {verdict}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    with open(args.output, "w") as f:
        json.dump({"checkpoints": all_results}, f, indent=2)
    print(f"\nSaved → {args.output}")

    if args.plot:
        n = len(all_results)
        fig, axes = plt.subplots(n, 2, figsize=(10, 3 * n), squeeze=False)
        for i, summary in enumerate(all_results):
            d_psnr = [r["delta_psnr"] for r in summary["rows"]]
            d_bpp = [r["delta_bpp"] for r in summary["rows"]]
            axes[i, 0].hist(d_psnr, bins=15, color="steelblue", alpha=0.85)
            axes[i, 0].axvline(0, color="red", ls="--")
            axes[i, 0].set_title(f"Δ PSNR — {os.path.basename(summary['checkpoint'])}")
            axes[i, 0].set_xlabel("PSNR(train) − PSNR(codec)  [dB]")
            axes[i, 0].grid(alpha=0.3)
            axes[i, 1].hist(d_bpp, bins=15, color="salmon", alpha=0.85)
            axes[i, 1].axvline(0, color="red", ls="--")
            axes[i, 1].set_title(f"Δ BPP — {os.path.basename(summary['checkpoint'])}")
            axes[i, 1].set_xlabel("BPP(train) − BPP(codec)")
            axes[i, 1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved plot → {args.plot}")


if __name__ == "__main__":
    main()
