"""
analyze/aggregate_to_latex.py
==============================
Aggregates the JSON outputs of the various ablation scripts into
LaTeX-ready table fragments. Drop the output of this script directly
into the paper's `\begin{tabular}` blocks.

Supported tables
----------------
  table1-routing       Table 1 — routing ablation (BD-rate, H, Δ_row)
  table2-backbone      Table 2 — backbone ablation (PSNR, latency, GFLOPs)
  table3-context       Table 3 — slice context ablation (slice-1 BPP, BD-rate, VRAM)
  table4-components    Table 4 — component removal (BD-rate per variant)

Usage
-----
    # Table 1 (routing):
    python analyze/aggregate_to_latex.py table1-routing \
        --bd-rate-json   bd_rate_routing.json \
        --entropy-json   dictionary_utilization.json \
        --output-tex     tables/table1_routing.tex

    # Table 2 (backbone):
    python analyze/aggregate_to_latex.py table2-backbone \
        --benchmark-json backbone_benchmark.json \
        --rd-jsons       rd_results/fdm.json rd_results/cnn.json \
                         rd_results/swin.json rd_results/ss2d.json \
                         rd_results/fdm_reversed.json \
        --bpp-target     0.4 \
        --output-tex     tables/table2_backbone.tex

    # Table 3 (context):
    python analyze/aggregate_to_latex.py table3-context \
        --bit-allocation-jsons   bit_allocations/*.json \
        --vram-json              vram_results.json \
        --bd-rate-json           bd_rate_context.json \
        --output-tex             tables/table3_context.tex

    # Table 4 (components):
    python analyze/aggregate_to_latex.py table4-components \
        --bd-rate-json   bd_rate_table.json \
        --output-tex     tables/table4_components.tex
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np


def _fmt_pct(x: float, decimals: int = 2, force_sign: bool = True) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    sign = "+" if force_sign and x >= 0 else ""
    return f"{sign}{x:.{decimals}f}\\,\\%"


def _fmt_num(x: Any, decimals: int = 2) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    return f"{x:.{decimals}f}"


# ──────────────────────────────────────────────────────────────────────────
# Table 1 — routing ablation
# ──────────────────────────────────────────────────────────────────────────
def cmd_table1(args):
    """
    Inputs:
      --bd-rate-json : compute_bd_rate.py output, anchor=full WMDC
                       variants=[softmax, balanced_eot, unbalanced_eot]
      --entropy-json : ablation_dictionary_utils.py JSON output
                       (will be looked up by routing-mode tag)
    """
    with open(args.bd_rate_json) as f:
        bd = json.load(f)
    rows_bd = {r["variant"]: r for r in bd["rows"]}

    entropy_data = {}
    if args.entropy_json:
        with open(args.entropy_json) as f:
            ent = json.load(f)
        # Generic schema: {variant_tag: {mean_H_norm, mean_row_dev}}
        # User must format their entropy JSON with these keys.
        entropy_data = ent.get("by_variant", {})

    out_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Routing} & \textbf{BD-rate vs.\ full} & "
        r"\textbf{$\mathcal{H}(\bar m)$ (nats)} & \textbf{$\Delta_{\text{row}}$} \\",
        r"\midrule",
    ]

    label_map = {
        "softmax": "Softmax baseline",
        "balanced_eot": "Balanced EOT",
        "unbalanced_eot": r"\textbf{UEOT (ours)}",
    }
    for tag in ["softmax", "balanced_eot", "unbalanced_eot"]:
        bd_row = rows_bd.get(tag, {})
        ent_row = entropy_data.get(tag, {})
        bd_pct = bd_row.get("bd_rate_pct")
        H_val = ent_row.get("mean_H_norm")
        row_dev = ent_row.get("mean_row_dev")
        line = (
            f"{label_map[tag]} & "
            f"{_fmt_pct(bd_pct)} & "
            f"{_fmt_num(H_val, 2)} & "
            f"{_fmt_num(row_dev, 3)} \\\\"
        )
        out_lines.append(line)

    out_lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(out_lines)

    with open(args.output_tex, "w") as f:
        f.write(tex)
    print(tex)
    print(f"\nSaved → {args.output_tex}")


# ──────────────────────────────────────────────────────────────────────────
# Table 2 — backbone ablation
# ──────────────────────────────────────────────────────────────────────────
def cmd_table2(args):
    """
    Inputs:
      --benchmark-json : analyze/benchmark_backbone.py output
      --rd-jsons       : per-backbone RD JSONs from compute_bd_rate.py evaluate
      --bpp-target     : interpolate PSNR at this bpp from each RD curve
    """
    with open(args.benchmark_json) as f:
        bench = json.load(f)
    bench_by_name = {r["backbone"]: r for r in bench["results"] if "error" not in r}

    rd_by_name: dict[str, dict] = {}
    for path in args.rd_jsons:
        with open(path) as f:
            d = json.load(f)
        rd_by_name[d["variant"]] = d

    def _interpolate_psnr(rd: dict, target_bpp: float) -> float:
        pts = sorted(rd["rd_points"], key=lambda r: r["bpp"])
        bpps = [r["bpp"] for r in pts]
        psnrs = [r["psnr"] for r in pts]
        if target_bpp <= bpps[0]:
            return psnrs[0]
        if target_bpp >= bpps[-1]:
            return psnrs[-1]
        for i in range(len(bpps) - 1):
            if bpps[i] <= target_bpp <= bpps[i + 1]:
                t = (target_bpp - bpps[i]) / (bpps[i + 1] - bpps[i])
                return psnrs[i] * (1 - t) + psnrs[i + 1] * t
        return float("nan")

    out_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Block} & \textbf{PSNR @ "
        f"{args.bpp_target:.1f}"
        r"\,bpp} & \textbf{Latency} & \textbf{GFLOPs} \\",
        r"\midrule",
    ]

    label_map = {
        "cnn": "Residual CNN",
        "swin": "Swin Transformer",
        "ss2d": "SS2D Mamba",
        "fdm_reversed": r"FDM, $H{\to}LL$ (rev.)",
        "fdm": r"\textbf{FDM, $LL{\to}H$ (ours)}",
    }
    order = ["cnn", "swin", "ss2d", "fdm_reversed", "fdm"]
    for tag in order:
        b = bench_by_name.get(tag, {})
        rd = rd_by_name.get(tag)
        psnr = _interpolate_psnr(rd, args.bpp_target) if rd else None
        latency = b.get("full_latency", {}).get("median_ms")
        gflops = b.get("flops", {}).get("gflops")
        line = (
            f"{label_map.get(tag, tag)} & "
            f"{_fmt_num(psnr, 2)}\\,dB & "
            f"{_fmt_num(latency, 0)}\\,ms & "
            f"{_fmt_num(gflops, 1)} \\\\"
        )
        out_lines.append(line)

    out_lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(out_lines)
    with open(args.output_tex, "w") as f:
        f.write(tex)
    print(tex)
    print(f"\nSaved → {args.output_tex}")


# ──────────────────────────────────────────────────────────────────────────
# Table 3 — slice context ablation
# ──────────────────────────────────────────────────────────────────────────
def cmd_table3(args):
    """
    Inputs:
      --bit-allocation-jsons : multiple JSONs from analysis_bit_allocation.py
                               One for "dense" variant, one for "stateful".
                               JSON must contain {"variant": "...", "slice_bpps": [...]}.
      --vram-json            : measure_vram.py JSON
      --bd-rate-json         : compute_bd_rate.py BD-rate output, anchor=dense, target=stateful
    """
    # Average slice-1 BPP per variant.
    slice1_by_variant: dict[str, list[float]] = {}
    for path in args.bit_allocation_jsons:
        with open(path) as f:
            d = json.load(f)
        v = d.get("variant", "unknown")
        bpps = d.get("slice_bpps", [])
        if bpps:
            slice1_by_variant.setdefault(v, []).append(bpps[0])

    # VRAM: pick the largest-resolution train-mode entry per variant.
    vram_by_variant: dict[str, float] = {}
    if args.vram_json:
        with open(args.vram_json) as f:
            vd = json.load(f)
        for r in vd["results"]:
            if "error" in r:
                continue
            if r.get("mode") != "train":
                continue
            v = r["variant"]
            res = r["H"] * r["W"]
            cur = vram_by_variant.get(v, (0, 0.0))
            if res > cur[0]:
                vram_by_variant[v] = (res, r["peak_vram_GB"])

    # BD-rate of stateful vs dense (single number).
    bd_pct = None
    if args.bd_rate_json:
        with open(args.bd_rate_json) as f:
            bd = json.load(f)
        for r in bd["rows"]:
            if r["variant"] == "stateful":
                bd_pct = r["bd_rate_pct"]
                break

    out_lines = [
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\textbf{Context strategy} & \textbf{Slice 1 BPP} & "
        r"\textbf{BD-rate} & \textbf{Peak VRAM} \\",
        r"\midrule",
    ]

    rows = [
        ("dense", r"Dense concatenation, $\mathcal{O}(K^2)$", "0.00\\,\\%"),
        (
            "stateful",
            r"\textbf{Stateful memory, $\mathcal{O}(K)$ (ours)}",
            _fmt_pct(bd_pct) if bd_pct is not None else "---",
        ),
    ]
    for v_tag, label, bd_str in rows:
        s1 = (
            np.mean(slice1_by_variant.get(v_tag, [float("nan")]))
            if slice1_by_variant.get(v_tag)
            else float("nan")
        )
        vram_pair = vram_by_variant.get(v_tag, (0, float("nan")))
        line = (
            f"{label} & "
            f"{_fmt_num(s1, 3)} & "
            f"{bd_str} & "
            f"{_fmt_num(vram_pair[1], 1)}\\,GB \\\\"
        )
        out_lines.append(line)

    out_lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(out_lines)
    with open(args.output_tex, "w") as f:
        f.write(tex)
    print(tex)
    print(f"\nSaved → {args.output_tex}")


# ──────────────────────────────────────────────────────────────────────────
# Table 4 — component removal
# ──────────────────────────────────────────────────────────────────────────
def cmd_table4(args):
    """
    Input: BD-rate JSON with anchor=full, variants=removal cases.
    """
    with open(args.bd_rate_json) as f:
        bd = json.load(f)

    # Map variant tag → human label.
    label_map = {
        "no_ueot": r"UEOT $\to$ softmax routing",
        "no_fdm": r"FDM $\to$ standard SS2D Mamba",
        "no_stateful_mem": r"Stateful memory $\to$ dense concatenation",
        "no_bootstrap_M1": r"Bootstrap $M_1$ from $\Phi$ $\to$ zero init",
        "no_disp_bonus": r"Remove dispersion bonus $-\alpha \mathcal{H}(\bar m)$",
        "no_dict_penalty": r"Remove dictionary penalty $\beta \mathcal{P}_{\text{dict}}$",
    }

    rows_by_variant = {r["variant"]: r for r in bd["rows"]}

    out_lines = [
        r"\begin{tabular}{lc}",
        r"\toprule",
        r"\textbf{Variant} & \textbf{$\Delta$ BD-rate vs.\ full} \\",
        r"\midrule",
        r"Full WMDC (reference) & 0.00\,\% \\",
    ]
    for tag in [
        "no_ueot",
        "no_fdm",
        "no_stateful_mem",
        "no_bootstrap_M1",
        "no_disp_bonus",
        "no_dict_penalty",
    ]:
        r = rows_by_variant.get(tag, {})
        bd_pct = r.get("bd_rate_pct")
        line = f"\\quad {label_map[tag]} & {_fmt_pct(bd_pct)} \\\\"
        out_lines.append(line)

    out_lines += [r"\bottomrule", r"\end{tabular}"]
    tex = "\n".join(out_lines)
    with open(args.output_tex, "w") as f:
        f.write(tex)
    print(tex)
    print(f"\nSaved → {args.output_tex}")


# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("table1-routing")
    p1.add_argument("--bd-rate-json", required=True)
    p1.add_argument("--entropy-json", default=None)
    p1.add_argument("--output-tex", required=True)
    p1.set_defaults(func=cmd_table1)

    p2 = sub.add_parser("table2-backbone")
    p2.add_argument("--benchmark-json", required=True)
    p2.add_argument("--rd-jsons", nargs="+", required=True)
    p2.add_argument("--bpp-target", type=float, default=0.4)
    p2.add_argument("--output-tex", required=True)
    p2.set_defaults(func=cmd_table2)

    p3 = sub.add_parser("table3-context")
    p3.add_argument("--bit-allocation-jsons", nargs="+", default=[])
    p3.add_argument("--vram-json", default=None)
    p3.add_argument("--bd-rate-json", default=None)
    p3.add_argument("--output-tex", required=True)
    p3.set_defaults(func=cmd_table3)

    p4 = sub.add_parser("table4-components")
    p4.add_argument("--bd-rate-json", required=True)
    p4.add_argument("--output-tex", required=True)
    p4.set_defaults(func=cmd_table4)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
