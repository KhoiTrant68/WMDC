"""
analyze/trace_magnitudes.py
===========================
Trace activation magnitudes through every module of WMDC.decompress()
to confirm the "residual cascade amplification" hypothesis.

Output: for each image and each module, log10(|x|_max) of the output.
A clean image shows magnitudes O(10⁰–10²) throughout.
A failing image shows monotonic growth across decoder layers.

Usage
-----
    python analyze/trace_magnitudes.py \\
        --dataset /path/to/kodak \\
        --checkpoint /path/to/checkpoint_best.pth.tar \\
        --output results/magnitude_trace.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze._common import iter_dataset, load_model  # noqa: E402


def _log10_abs_max(t) -> float:
    """log10 of |t|_max, with safe handling of zero / non-tensor / NaN."""
    if not isinstance(t, torch.Tensor):
        return float("nan")
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return float("inf")
    m = finite.abs().max().item()
    return math.log10(max(m, 1e-30))


class MagnitudeTracker:
    def __init__(self):
        self.records: list[tuple[str, float, bool]] = []
        self.handles: list = []

    def install(self, model) -> None:
        for name, module in model.named_modules():
            if not name:
                continue
            self.handles.append(
                module.register_forward_hook(self._make_hook(name))
            )

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self) -> None:
        self.records = []

    def _make_hook(self, name):
        def hook(_module, _inputs, output):
            if isinstance(output, torch.Tensor):
                t = output
            elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
                t = output[0]
            else:
                return
            lm = _log10_abs_max(t)
            has_nonfinite = not torch.isfinite(t).all().item()
            self.records.append((name, lm, has_nonfinite))
        return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument("--backbone", type=str, default="fdm")
    parser.add_argument("--use-wls-shortcut", action="store_true")
    parser.add_argument(
        "--memory-init", type=str, default="bootstrap",
        choices=["bootstrap", "zero"],
    )
    parser.add_argument("--content-adaptive", action="store_true")
    parser.add_argument("--cluster-num", type=int, default=8)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument(
        "--filter-decoder", action="store_true",
        help="Only log modules under g_s.* (the decoder where NaN appears).",
    )
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = load_model(
        args.checkpoint, device=device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps, sinkhorn_iters=args.sinkhorn_iters,
        backbone=args.backbone,
        use_wls_shortcut=args.use_wls_shortcut,
        memory_init=args.memory_init,
        use_content_adaptive=args.content_adaptive,
        cluster_num=args.cluster_num,
        strict_load=False,
    )

    tracker = MagnitudeTracker()
    tracker.install(model)

    per_image: list[dict] = []
    for path, _x, x_padded, H, W in iter_dataset(args.dataset, device=device):
        tracker.reset()
        with torch.no_grad():
            out_enc = model.compress(x_padded)
            tracker.reset()  # discard compress trace
            _ = model.decompress(out_enc["strings"], out_enc["shape"])

        records = tracker.records
        if args.filter_decoder:
            records = [(n, m, b) for (n, m, b) in records if n.startswith("g_s")]

        # Find first non-finite + max magnitude reached
        first_bad_idx = next(
            (i for i, (n, m, b) in enumerate(records) if b),
            None,
        )
        max_record = max(records, key=lambda r: r[1]) if records else None

        per_image.append({
            "file": os.path.basename(path),
            "resolution": f"{W}x{H}",
            "n_modules": len(records),
            "first_nonfinite": records[first_bad_idx] if first_bad_idx is not None
                              else None,
            "max_log10_magnitude": max_record,
            "trace_compressed": [
                {"module": n, "log10_max": round(m, 2), "nonfinite": b}
                for (n, m, b) in records[::4]  # subsample every 4th to keep JSON small
            ],
        })
        print(
            f"{os.path.basename(path):>14s}  "
            f"max_log10|x|={max_record[1]:6.2f} at {max_record[0][:40]:40s}  "
            f"first_bad={records[first_bad_idx][0][:40] if first_bad_idx is not None else '—':40s}"
        )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"per_image": per_image}, f, indent=2)

    print(f"\nReport → {args.output}")


if __name__ == "__main__":
    main()
