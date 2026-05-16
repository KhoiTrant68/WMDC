"""
analyze/compare_against_sota.py
================================
Compare a WMDC RD curve (as produced by `compute_bd_rate.py evaluate`)
against the suite of published SOTA codecs in `sota_kodak_anchors.json`.

Purpose
-------
This is the script you run *after* you have RD points for WMDC on Kodak.
It produces (a) a BD-rate table vs every SOTA anchor and (b) a single
verdict line — "WMDC beats N of M SOTA methods" — which is the question
that determines whether the paper has a shot at CVPR.

Usage
-----
    # 1) Produce WMDC RD points (six lambdas):
    python analyze/compute_bd_rate.py evaluate \
        --variant-name wmdc_full \
        --checkpoints ckpts/full/lam_0.0018/best.pth.tar ... \
        --dataset /data/kodak \
        --output rd_results/wmdc_full_kodak.json \
        --cuda

    # 2) Compare against all published SOTA on Kodak:
    python analyze/compare_against_sota.py \
        --wmdc-json   rd_results/wmdc_full_kodak.json \
        --anchors     analyze/sota_kodak_anchors.json \
        --output      rd_results/sota_comparison_kodak.json

Reading the output
------------------
- "wmdc_vs_<X>_bd_rate_pct" is the BD-rate of WMDC relative to X.
  NEGATIVE = WMDC saves bits at equal PSNR = WMDC wins.
  POSITIVE = WMDC needs more bits at equal PSNR = X wins.
- The verdict line at the end summarises wins/losses.

Caveats
-------
* `sota_kodak_anchors.json` contains numbers extracted from published
  figures. They are within ~0.05 dB / ~0.005 bpp of the true value —
  good enough to decide *direction* (WMDC > MLIC++ or not), NOT good
  enough for the camera-ready table. For final numbers, reproduce
  baselines with their public checkpoints.
* BD-rate is sensitive to BPP range overlap. If WMDC's lowest BPP is
  0.15 and the anchor starts at 0.10, the comparison is extrapolated
  and unreliable below 0.15. The script warns when overlap < 80%.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Tuple

try:
    import bjontegaard as bd
except ImportError:
    bd = None
    print(
        "[ERROR] `bjontegaard` not installed. `pip install bjontegaard`",
        file=sys.stderr,
    )
    sys.exit(1)


def _extract_curve(rd_points: list) -> Tuple[List[float], List[float]]:
    pts = sorted(rd_points, key=lambda r: r["bpp"])
    return [r["bpp"] for r in pts], [r["psnr"] for r in pts]


def _overlap_ratio(
    bpp_a: List[float], bpp_b: List[float]
) -> float:
    lo = max(min(bpp_a), min(bpp_b))
    hi = min(max(bpp_a), max(bpp_b))
    span_b = max(bpp_b) - min(bpp_b)
    if span_b <= 0:
        return 0.0
    return max(0.0, (hi - lo) / span_b)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wmdc-json", type=str, required=True)
    p.add_argument("--anchors", type=str, required=True)
    p.add_argument("--output", type=str, required=True)
    p.add_argument(
        "--method",
        type=str,
        default="akima",
        choices=["akima", "pchip", "cubic"],
    )
    p.add_argument(
        "--vs-anchor",
        type=str,
        default="vtm_17_intra",
        help="Anchor against which the BD-rate column in the paper is reported.",
    )
    args = p.parse_args()

    with open(args.wmdc_json) as f:
        wmdc = json.load(f)
    with open(args.anchors) as f:
        anchors = json.load(f)

    wmdc_bpp, wmdc_psnr = _extract_curve(wmdc["rd_points"])
    print(f"WMDC RD curve: {len(wmdc_bpp)} points, "
          f"bpp ∈ [{min(wmdc_bpp):.3f}, {max(wmdc_bpp):.3f}], "
          f"psnr ∈ [{min(wmdc_psnr):.2f}, {max(wmdc_psnr):.2f}]")

    if len(wmdc_bpp) < 4:
        print("[WARN] BD-rate needs ≥4 RD points; <4 is unreliable.")

    print(f"\n{'Anchor':<20s} {'#pts':>4s} {'overlap':>8s} "
          f"{'BD-rate(%)':>11s} {'BD-PSNR(dB)':>12s}  verdict")
    print("─" * 70)

    rows = []
    wins = 0
    losses = 0
    for name, entry in anchors.get("variants", {}).items():
        a_bpp, a_psnr = _extract_curve(entry["rd_points"])
        overlap = _overlap_ratio(a_bpp, wmdc_bpp)

        try:
            # bd_rate convention: returns the % bitrate change of the
            # SECOND curve relative to the FIRST at equal PSNR.
            # We want WMDC's saving vs anchor → anchor is first.
            bd_rate = bd.bd_rate(a_bpp, a_psnr, wmdc_bpp, wmdc_psnr, args.method)
            bd_psnr = bd.bd_psnr(a_bpp, a_psnr, wmdc_bpp, wmdc_psnr, args.method)
        except Exception as e:
            bd_rate = float("nan")
            bd_psnr = float("nan")
            print(f"  [{name}] BD-rate failed: {e}")

        # Negative BD-rate = WMDC saves bits = WMDC wins.
        verdict = "WIN " if (bd_rate < 0 and not (bd_rate != bd_rate)) else "loss"
        if bd_rate < 0:
            wins += 1
        elif bd_rate > 0:
            losses += 1

        overlap_str = f"{100 * overlap:.0f}%"
        if overlap < 0.8:
            overlap_str += "!"  # extrapolation warning

        print(
            f"{name:<20s} {len(a_bpp):>4d} {overlap_str:>8s} "
            f"{bd_rate:>+10.3f}% {bd_psnr:>+11.3f} dB  {verdict}"
        )

        rows.append({
            "anchor": name,
            "anchor_cite": entry.get("_cite", ""),
            "n_anchor_points": len(a_bpp),
            "bpp_overlap": overlap,
            "wmdc_vs_anchor_bd_rate_pct": bd_rate,
            "wmdc_vs_anchor_bd_psnr_db": bd_psnr,
            "wmdc_wins": bd_rate < 0,
        })

    # Headline result vs the chosen "main" anchor.
    main_row = next((r for r in rows if r["anchor"] == args.vs_anchor), None)
    print("\n" + "═" * 70)
    if main_row is not None:
        print(f"HEADLINE (vs {args.vs_anchor}): "
              f"BD-rate = {main_row['wmdc_vs_anchor_bd_rate_pct']:+.2f}%")
    print(f"VERDICT: WMDC wins {wins}/{wins + losses} comparisons "
          f"({100.0 * wins / max(wins + losses, 1):.0f}%)")

    if main_row and main_row["wmdc_vs_anchor_bd_rate_pct"] < -13.0:
        cvpr_call = "STRONG — comparable to / better than MLIC++ (-13.4%)"
    elif main_row and main_row["wmdc_vs_anchor_bd_rate_pct"] < -10.0:
        cvpr_call = "COMPETITIVE — in the TCM/FTIC range, needs strong novelty story"
    elif main_row and main_row["wmdc_vs_anchor_bd_rate_pct"] < -7.0:
        cvpr_call = "MARGINAL — ELIC-level, CVPR difficult, ECCV/DCC realistic"
    elif main_row and main_row["wmdc_vs_anchor_bd_rate_pct"] < 0:
        cvpr_call = "WEAK — beats VTM but not modern neural baselines; revise scope"
    else:
        cvpr_call = "FAIL — does not beat VTM-17.0; needs serious rework"
    print(f"CVPR call:    {cvpr_call}")
    print("═" * 70)

    out = {
        "wmdc_json": args.wmdc_json,
        "anchors_json": args.anchors,
        "interpolation": args.method,
        "rows": rows,
        "summary": {
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(wins + losses, 1),
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull comparison saved → {args.output}")


if __name__ == "__main__":
    main()
