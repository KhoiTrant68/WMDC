"""
analyze/visualize_dictionary_diversity.py
==========================================
Visualises the dictionary tokens (the K vectors used by the routing
attention) to support the paper's claim that the dictionary penalty
P_dict produces diverse, mutually decorrelated tokens.

Two diagnostic figures
----------------------
  Figure 1 — Cosine-similarity heatmap (N_d × N_d).
    Diagonal = 1; off-diagonal should be small. With dispersion+penalty,
    we expect a sharp diagonal; without, we expect bright clusters.

  Figure 2 — t-SNE projection of dictionary tokens to 2D, coloured by
    cluster ID (k-means with k=8). Helps the reader see whether tokens
    are spread out or collapsed into a few clusters.

Two checkpoints can be compared side-by-side: WITH and WITHOUT the
penalty (or one with full WMDC and one with --routing-mode softmax).

Usage
-----
    python analyze/visualize_dictionary_diversity.py \
        --checkpoint-with    ckpts/full/best.pth.tar \
        --checkpoint-without ckpts/no_dict_penalty/best.pth.tar \
        --image              /data/kodak/kodim05.png \
        --output             dictionary_diversity.pdf \
        --cuda

Notes
-----
* The dictionary tokens are produced by `model.hyper_to_dict(z_hat)`,
  which depends on the input image. We therefore extract the tokens for
  ONE reference image; the diversity statistics are stable across images
  by construction (the penalty is computed on the dictionary itself, not
  on per-image queries).
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from _common import load_image, load_model

try:
    from sklearn.cluster import KMeans
    from sklearn.manifold import TSNE
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    print("[WARN] sklearn not available → t-SNE plot will be skipped.")


def _get_dict_tokens(model, x_padded: torch.Tensor) -> torch.Tensor:
    """Returns dictionary tokens of shape (N_d, dict_dim) on CPU."""
    with torch.no_grad():
        y = model.g_a(x_padded)
        z = model.h_a(y)
        try:
            z_str = model.entropy_bottleneck.compress(z)
            z_hat = model.entropy_bottleneck.decompress(z_str, z.size()[-2:])
        except Exception:
            z_hat = torch.round(z)

        out = model.hyper_to_dict(z_hat)
        dt = out[0] if isinstance(out, tuple) else out
        # dt: (B, N_d, dict_dim) → take batch 0.
        return dt[0].detach().cpu()


def _cosine_matrix(K: torch.Tensor) -> np.ndarray:
    """K: (N, D). Returns NxN cosine sim matrix."""
    K_norm = F.normalize(K, p=2, dim=-1)
    return (K_norm @ K_norm.T).numpy()


def plot_one(ax, sim: np.ndarray, title: str):
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Token j")
    ax.set_ylabel("Token i")
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-with", required=True,
                   help="Checkpoint trained WITH dispersion+dict_penalty.")
    p.add_argument("--checkpoint-without", required=True,
                   help="Checkpoint trained WITHOUT dispersion or dict_penalty.")
    p.add_argument("--image", required=True,
                   help="Reference image used to compute dictionary tokens.")
    p.add_argument("--output", type=str, default="dictionary_diversity.pdf")
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    _x, x_padded, _H, _W = load_image(args.image, device=device)

    print(f"== Loading WITH-penalty checkpoint: {args.checkpoint_with}")
    m_with = load_model(args.checkpoint_with, device,
                        routing_mode=args.routing_mode, ot_eps=args.ot_eps,
                        update_for_inference=True)
    K_with = _get_dict_tokens(m_with, x_padded)
    sim_with = _cosine_matrix(K_with)

    print(f"== Loading WITHOUT-penalty checkpoint: {args.checkpoint_without}")
    m_no = load_model(args.checkpoint_without, device,
                      routing_mode=args.routing_mode, ot_eps=args.ot_eps,
                      update_for_inference=True)
    K_no = _get_dict_tokens(m_no, x_padded)
    sim_no = _cosine_matrix(K_no)

    # Diagnostic stats.
    def _offdiag_stats(sim: np.ndarray) -> dict:
        N = sim.shape[0]
        mask = ~np.eye(N, dtype=bool)
        off = sim[mask]
        return {
            "max_abs_off": float(np.max(np.abs(off))),
            "mean_abs_off": float(np.mean(np.abs(off))),
            "frac_high_sim": float(np.mean(off > 0.5)),  # % pairs nearly aligned
        }

    s_with = _offdiag_stats(sim_with)
    s_no = _offdiag_stats(sim_no)
    print(f"\nWITH penalty   : max|off|={s_with['max_abs_off']:.3f}  "
          f"mean|off|={s_with['mean_abs_off']:.3f}  "
          f"frac(>0.5)={s_with['frac_high_sim']*100:.2f}%")
    print(f"WITHOUT penalty: max|off|={s_no['max_abs_off']:.3f}  "
          f"mean|off|={s_no['mean_abs_off']:.3f}  "
          f"frac(>0.5)={s_no['frac_high_sim']*100:.2f}%")

    # ── Plot ─────────────────────────────────────────────────────────────
    if _HAS_SKLEARN:
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        axes = np.array(axes).reshape(1, 2)

    im0 = plot_one(axes[0, 0], sim_with,
                   f"With penalty\nmax|off|={s_with['max_abs_off']:.2f}, "
                   f"mean|off|={s_with['mean_abs_off']:.2f}")
    im1 = plot_one(axes[0, 1], sim_no,
                   f"Without penalty\nmax|off|={s_no['max_abs_off']:.2f}, "
                   f"mean|off|={s_no['mean_abs_off']:.2f}")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046)
    plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    # t-SNE row.
    if _HAS_SKLEARN:
        for ax, K, label in [
            (axes[1, 0], K_with, "With penalty"),
            (axes[1, 1], K_no, "Without penalty"),
        ]:
            try:
                emb = TSNE(n_components=2, perplexity=15, random_state=0).fit_transform(
                    K.numpy())
                cluster_ids = KMeans(n_clusters=8, n_init=10, random_state=0).fit_predict(
                    K.numpy())
                ax.scatter(emb[:, 0], emb[:, 1], c=cluster_ids, cmap="tab10",
                           s=50, alpha=0.85, edgecolor="black", linewidth=0.5)
                ax.set_title(f"{label} — t-SNE (k-means k=8)", fontsize=11)
                ax.grid(alpha=0.3)
            except Exception as e:
                ax.text(0.5, 0.5, f"t-SNE failed:\n{e}",
                        ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
