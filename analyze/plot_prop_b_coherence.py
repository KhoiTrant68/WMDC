"""
analyze/plot_prop_b_coherence.py
================================
Frame-theoretic diagnostics for the content-adaptive dictionary (Prop. B).

Prop. B framing
---------------
The dictionary loss `‖SSᵀ − I‖_F²` is a convex surrogate for the frame
coherence μ(D) = max_{i≠j} |⟨dᵢ, dⱼ⟩|.  The Welch bound

    μ(D) ≥ √( (N − D) / (D (N − 1)) )       (only meaningful when N > D)

is the tight lower bound; an Equiangular Tight Frame (ETF / WBES) attains it.
In the undercomplete regime (N ≤ D, our default config N=128, D=640) the
bound is degenerate (μ=0 achievable) and the loss enforces orthonormality.

Two modes
---------
* `--checkpoints path1 path2 ...`  : produces a TRAJECTORY plot of
  coherence_mu, tightness_ratio, etf_gap across the listed checkpoints
  (typically one per epoch).
* `--checkpoint path`              : produces a SNAPSHOT bar chart of the
  five Welch-bound statistics for a single checkpoint.

In both modes one image is used to drive the content-adaptive dictionary;
defaults to a random tensor when no image is supplied.

Usage
-----
    # Trajectory across epochs
    python analyze/plot_prop_b_coherence.py \\
        --checkpoints runs/full/checkpoint_ep0010.pth.tar \\
                      runs/full/checkpoint_ep0050.pth.tar \\
                      runs/full/checkpoint_ep0100.pth.tar \\
                      runs/full/checkpoint_ep0200.pth.tar \\
                      runs/full/checkpoint_ep0400.pth.tar \\
        --image /data/kodak/kodim01.png \\
        -o prop_b_coherence.pdf \\
        --cuda

    # Single-checkpoint snapshot
    python analyze/plot_prop_b_coherence.py \\
        --checkpoint runs/full/checkpoint_best.pth.tar \\
        -o prop_b_coherence_snapshot.pdf --cuda
"""

from __future__ import annotations

import argparse
import os
import re

import matplotlib.pyplot as plt
import torch

from _common import load_image, load_model
from modules.dictionary_blocks import QueryDictionaryGenerator


@torch.no_grad()
def _coherence_for_ckpt(
    ckpt_path: str,
    x_padded: torch.Tensor | None,
    device: str,
) -> dict:
    """Build model from ckpt, run one forward, return coherence_stats(dt)."""
    model = load_model(
        ckpt_path,
        device=device,
        strict_load=False,
        update_for_inference=False,
    )
    model.eval()
    if x_padded is None:
        # Synthetic input: random image of fixed size.  Coherence is mostly
        # driven by the learned dt projection layers, not the input content,
        # so this is acceptable when no image is supplied.
        x_padded = torch.rand(1, 3, 256, 256, device=device)
    _ = model(x_padded)
    dt = model.last_dt()
    if dt is None:
        raise RuntimeError(f"{ckpt_path}: model.last_dt() returned None")
    return QueryDictionaryGenerator.coherence_stats(dt)


def _epoch_from_path(p: str) -> int | None:
    """Best-effort epoch parsing for X-axis labels: matches ep0123 / epoch_123."""
    m = re.search(r"(?:ep|epoch[_-]?)(\d+)", os.path.basename(p))
    return int(m.group(1)) if m else None


def _plot_trajectory(ckpts: list[str], stats: list[dict], output: str) -> None:
    epochs = [_epoch_from_path(p) for p in ckpts]
    xs = epochs if all(e is not None for e in epochs) else list(range(len(ckpts)))
    xlabel = "epoch" if all(e is not None for e in epochs) else "checkpoint index"

    mu = [s["coherence_mu"] for s in stats]
    tightness = [s["tightness_ratio"] for s in stats]
    etf_gap = [s["etf_gap"] for s in stats]
    welch = stats[0]["welch_bound"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(xs, mu, marker="o", color="C0", label=r"$\mu(D)$")
    if welch > 0:
        axes[0].axhline(
            welch, ls="--", color="C3", label=f"Welch bound = {welch:.4f}"
        )
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("coherence")
    axes[0].set_title(r"Coherence $\mu(D)$ vs Welch bound")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(xs, tightness, marker="s", color="C1")
    axes[1].axhline(1.0, ls="--", color="k", alpha=0.4, label="Parseval (=1)")
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("frame potential / fp_min")
    axes[1].set_title("Tightness ratio (→1 ⇔ tight frame)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(xs, etf_gap, marker="^", color="C2")
    axes[2].set_xlabel(xlabel)
    axes[2].set_ylabel(r"$\max(0,\ \mu - \mathrm{Welch})$")
    axes[2].set_title("ETF gap (→0 ⇔ Welch-bound equality)")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle(
        f"Prop. B — dictionary frame statistics (N={stats[0]['n_atoms']}, "
        f"D={stats[0]['atom_dim']})"
    )
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"saved → {output}")


def _plot_snapshot(stats: dict, output: str) -> None:
    keys = ["coherence_mu", "welch_bound", "etf_gap", "tightness_ratio"]
    vals = [stats[k] for k in keys]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(range(len(keys)), vals, color=["C0", "C3", "C2", "C1"])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(
        [r"$\mu(D)$", "Welch", "ETF gap", "tightness"], rotation=20
    )
    for bar, v in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_title(
        f"Prop. B snapshot — N={stats['n_atoms']}, D={stats['atom_dim']}"
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output, bbox_inches="tight")
    print(f"saved → {output}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--checkpoint", help="single ckpt → snapshot bar chart")
    g.add_argument(
        "--checkpoints",
        nargs="+",
        help="multiple ckpts (one per epoch) → trajectory plot",
    )
    ap.add_argument("--image", default=None)
    ap.add_argument("-o", "--output", default="prop_b_coherence.pdf")
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    x_padded = None
    if args.image is not None:
        _, x_padded, _, _ = load_image(args.image, device=device)

    if args.checkpoints:
        stats_list = [
            _coherence_for_ckpt(p, x_padded, device) for p in args.checkpoints
        ]
        _plot_trajectory(args.checkpoints, stats_list, args.output)
    else:
        stats = _coherence_for_ckpt(args.checkpoint, x_padded, device)
        _plot_snapshot(stats, args.output)


if __name__ == "__main__":
    main()
