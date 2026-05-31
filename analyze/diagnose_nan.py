"""
analyze/diagnose_nan.py
=======================
Localise the first NaN-producing operation in WMDC.decompress().

Strategy
--------
Monkey-patch the model with forward hooks on every nn.Module.  When a hook
sees a NaN/inf in its output it records the module path, input statistics
(min/max/has_nan/has_inf), and stops at the first occurrence per image.

Output: a JSON report with the offending module per failing image, so we
can fix the actual source instead of masking symptoms.

Usage
-----
    python analyze/diagnose_nan.py \\
        --dataset /path/to/kodak \\
        --checkpoint /path/to/checkpoint_best.pth.tar \\
        --output results/nan_diagnostic.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analyze._common import iter_dataset, load_model  # noqa: E402


def _t_stats(t: Any) -> dict | None:
    """Compact stats for a tensor, None for non-tensor."""
    if not isinstance(t, torch.Tensor):
        return None
    t = t.detach()
    finite = torch.isfinite(t)
    n_total = t.numel()
    n_finite = int(finite.sum().item())
    if n_finite == 0:
        return {"shape": list(t.shape), "all_nonfinite": True}
    f = t[finite]
    return {
        "shape": list(t.shape),
        "has_nan": bool(torch.isnan(t).any().item()),
        "has_inf": bool(torch.isinf(t).any().item()),
        "n_nan": int(torch.isnan(t).sum().item()),
        "n_inf": int(torch.isinf(t).sum().item()),
        "finite_min": float(f.min().item()),
        "finite_max": float(f.max().item()),
        "finite_abs_max": float(f.abs().max().item()),
        "frac_nan": int(torch.isnan(t).sum().item()) / max(n_total, 1),
    }


class NaNTracker:
    """Records the FIRST nn.Module whose output contains NaN/inf."""

    def __init__(self):
        self.first_bad: dict | None = None
        self.handles: list = []

    def install(self, model: nn.Module) -> None:
        for name, module in model.named_modules():
            if not name:  # skip the root module
                continue
            handle = module.register_forward_hook(
                self._make_hook(name, type(module).__name__)
            )
            self.handles.append(handle)

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []

    def reset(self) -> None:
        self.first_bad = None

    def _make_hook(self, name: str, cls_name: str):
        def hook(module, inputs, output):
            if self.first_bad is not None:
                return
            # Some modules return tuples / dicts
            tensors = []
            if isinstance(output, torch.Tensor):
                tensors = [(None, output)]
            elif isinstance(output, (tuple, list)):
                tensors = [(i, t) for i, t in enumerate(output)
                           if isinstance(t, torch.Tensor)]
            elif isinstance(output, dict):
                tensors = [(k, t) for k, t in output.items()
                           if isinstance(t, torch.Tensor)]
            for key, t in tensors:
                if not torch.isfinite(t).all():
                    in_stats = []
                    for inp in inputs:
                        in_stats.append(_t_stats(inp))
                    self.first_bad = {
                        "module": name,
                        "class": cls_name,
                        "output_key": key,
                        "output_stats": _t_stats(t),
                        "input_stats": in_stats,
                    }
                    return

        return hook


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--routing-mode", type=str, default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "--backbone", type=str, default="fdm",
        choices=["fdm", "cnn", "swin", "ss2d", "fdm_reversed"],
    )
    parser.add_argument("--use-wls-shortcut", action="store_true")
    parser.add_argument(
        "--memory-init", type=str, default="bootstrap",
        choices=["bootstrap", "zero"],
    )
    parser.add_argument("--content-adaptive", action="store_true")
    parser.add_argument("--cluster-num", type=int, default=8)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    model = load_model(
        args.checkpoint,
        device=device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        backbone=args.backbone,
        use_wls_shortcut=args.use_wls_shortcut,
        memory_init=args.memory_init,
        use_content_adaptive=args.content_adaptive,
        cluster_num=args.cluster_num,
        strict_load=False,
    )

    tracker = NaNTracker()
    tracker.install(model)

    per_image: list[dict] = []
    n_clean = 0
    n_nan = 0

    for path, _x, x_padded, H, W in iter_dataset(args.dataset, device=device):
        tracker.reset()
        with torch.no_grad():
            out_enc = model.compress(x_padded)
            compress_first_bad = tracker.first_bad

            tracker.reset()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            decompress_first_bad = tracker.first_bad

            x_hat_has_nan = bool(
                not torch.isfinite(out_dec["x_hat"]).all().item()
            )

        if compress_first_bad is None and decompress_first_bad is None:
            n_clean += 1
            print(f"  CLEAN   {os.path.basename(path)}")
        else:
            n_nan += 1
            stage = "compress" if compress_first_bad else "decompress"
            bad = compress_first_bad or decompress_first_bad
            print(f"  NAN ({stage:>10s})  {os.path.basename(path):>14s}  "
                  f"first bad: {bad['module']:50s}  ({bad['class']})  "
                  f"frac_nan={bad['output_stats'].get('frac_nan', 0):.2%}")

        per_image.append({
            "file": os.path.basename(path),
            "resolution": f"{W}x{H}",
            "x_hat_has_nan": x_hat_has_nan,
            "compress_first_bad": compress_first_bad,
            "decompress_first_bad": decompress_first_bad,
        })

    tracker.remove()

    summary = {
        "n_total": n_clean + n_nan,
        "n_clean": n_clean,
        "n_nan": n_nan,
        "per_image": per_image,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\nClean: {n_clean}/{n_clean + n_nan}    "
          f"NaN: {n_nan}/{n_clean + n_nan}")
    print(f"Report → {args.output}")


if __name__ == "__main__":
    main()
