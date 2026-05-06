"""
analyze/visualize_failure_cases.py
==================================
Identifies and visualises the worst-performing images for the model on a
test set. Useful for the paper's "limitations" subsection or rebuttal,
where reviewers often ask "when does your method fail?".

What it does
------------
1. Runs full codec round-trip on every image in --dataset.
2. Ranks images by (a) PSNR and (b) BPP-vs-PSNR efficiency relative to
   a reference RD curve (e.g. VTM, if --reference-rd is provided).
3. Outputs a multi-page PDF gallery with:
       Original | Reconstructed | Error map | Per-image stats

Usage
-----
    python analyze/visualize_failure_cases.py \
        --checkpoint  ckpts/full/lam_0.013/best.pth.tar \
        --dataset     /data/kodak \
        --top-k       5 \
        --output      failure_cases.pdf \
        --cuda

    # With reference RD curve to also flag underperformers:
    python analyze/visualize_failure_cases.py \
        --checkpoint    ...  \
        --dataset       /data/kodak \
        --reference-rd  anchors/vtm_kodak.json \
        --top-k         5 \
        --output        failure_cases.pdf \
        --cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from _common import compute_psnr, evaluate_codec, iter_dataset, load_model


def _interpolate_psnr(
    ref_bpp: list[float], ref_psnr: list[float], target_bpp: float
) -> float:
    """Linearly interpolate the reference PSNR at a given bpp."""
    if target_bpp <= ref_bpp[0]:
        return ref_psnr[0]
    if target_bpp >= ref_bpp[-1]:
        return ref_psnr[-1]
    for i in range(len(ref_bpp) - 1):
        if ref_bpp[i] <= target_bpp <= ref_bpp[i + 1]:
            t = (target_bpp - ref_bpp[i]) / (ref_bpp[i + 1] - ref_bpp[i])
            return ref_psnr[i] * (1 - t) + ref_psnr[i + 1] * t
    return float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument(
        "--top-k", type=int, default=5, help="Number of worst-case images to visualise."
    )
    p.add_argument(
        "--reference-rd",
        type=str,
        default=None,
        help="Optional JSON with reference rd_points to compute PSNR-gap-vs-VTM.",
    )
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("--output", type=str, default="failure_cases.pdf")
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = load_model(
        args.checkpoint,
        device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        update_for_inference=True,
    )

    # Optional reference RD for efficiency gap.
    ref_bpp = ref_psnr = None
    if args.reference_rd:
        with open(args.reference_rd) as f:
            ref = json.load(f)
        pts = sorted(ref["rd_points"], key=lambda r: r["bpp"])
        ref_bpp = [r["bpp"] for r in pts]
        ref_psnr = [r["psnr"] for r in pts]

    # ── Evaluate every image ────────────────────────────────────────────
    records: list[dict] = []
    for path, x, x_padded, H, W in iter_dataset(args.dataset, device=device):
        with torch.no_grad():
            out_enc = model.compress(x_padded)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            x_hat = out_dec["x_hat"][:, :, :H, :W].clamp(0, 1)

        psnr = compute_psnr(x, x_hat)
        # Quick eval of bpp.
        from _common import compute_actual_bpp

        bpp = compute_actual_bpp(out_enc["strings"], H * W)

        psnr_ref = _interpolate_psnr(ref_bpp, ref_psnr, bpp) if ref_bpp else None
        psnr_gap = (psnr - psnr_ref) if psnr_ref is not None else None

        # Error map for visualisation.
        err = (x - x_hat).abs().mean(dim=1, keepdim=True).squeeze().cpu().numpy()

        records.append(
            {
                "path": path,
                "name": os.path.basename(path),
                "x": x.squeeze().permute(1, 2, 0).cpu().numpy(),
                "x_hat": x_hat.squeeze().permute(1, 2, 0).cpu().numpy(),
                "err": err,
                "psnr": psnr,
                "bpp": bpp,
                "psnr_ref": psnr_ref,
                "psnr_gap": psnr_gap,
            }
        )
        gap_str = f", gap = {psnr_gap:+.3f}" if psnr_gap is not None else ""
        print(
            f"  {os.path.basename(path):>30s}: psnr={psnr:.3f}  bpp={bpp:.4f}{gap_str}"
        )

    # ── Pick worst-K ────────────────────────────────────────────────────
    if ref_bpp:
        # Worst = most negative psnr_gap.
        records.sort(key=lambda r: r["psnr_gap"] if r["psnr_gap"] is not None else 0.0)
        rank_label = "PSNR-vs-anchor gap (most negative first)"
    else:
        records.sort(key=lambda r: r["psnr"])
        rank_label = "PSNR (lowest first)"

    worst = records[: args.top_k]
    print(f"\n=== Top {args.top_k} failure cases ranked by {rank_label} ===")
    for i, r in enumerate(worst, 1):
        gap_str = f", gap={r['psnr_gap']:+.3f}" if r["psnr_gap"] is not None else ""
        print(f"  {i}. {r['name']}: psnr={r['psnr']:.3f}, bpp={r['bpp']:.4f}{gap_str}")

    # ── Plot ────────────────────────────────────────────────────────────
    n = len(worst)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.3 * n), squeeze=False)
    for r_idx, r in enumerate(worst):
        axes[r_idx, 0].imshow(r["x"])
        axes[r_idx, 0].set_title(f"{r['name']} — original", fontsize=10)
        axes[r_idx, 0].axis("off")

        title2 = f"reconstruction\nPSNR={r['psnr']:.2f} dB, BPP={r['bpp']:.3f}"
        if r["psnr_gap"] is not None:
            title2 += f", gap={r['psnr_gap']:+.2f} dB"
        axes[r_idx, 1].imshow(r["x_hat"])
        axes[r_idx, 1].set_title(title2, fontsize=10)
        axes[r_idx, 1].axis("off")

        # Error map: amplify small differences for visibility.
        err = r["err"]
        vmax = max(0.05, float(np.percentile(err, 99)))
        im = axes[r_idx, 2].imshow(err, cmap="hot", vmin=0, vmax=vmax)
        axes[r_idx, 2].set_title(f"|x − x̂| (clipped at p99={vmax:.3f})", fontsize=10)
        axes[r_idx, 2].axis("off")
        plt.colorbar(im, ax=axes[r_idx, 2], fraction=0.046)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.output}")

    if args.json_out:
        # Strip image arrays before dumping.
        slim = []
        for r in records:
            slim.append(
                {
                    "name": r["name"],
                    "psnr": r["psnr"],
                    "bpp": r["bpp"],
                    "psnr_ref": r["psnr_ref"],
                    "psnr_gap": r["psnr_gap"],
                }
            )
        with open(args.json_out, "w") as f:
            json.dump({"records": slim, "ranking_metric": rank_label}, f, indent=2)
        print(f"Saved JSON → {args.json_out}")


if __name__ == "__main__":
    main()
