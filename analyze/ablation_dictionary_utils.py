"""
analyze/ablation_dictionary_utils.py  (FIXED)
=============================================
Measures dictionary token utilisation across a test dataset, comparing a
checkpoint trained WITH dispersion regularisation against one trained
WITHOUT it.

Bugs fixed vs. previous version
-------------------------------
1. Hooks `model.eot_attentions` (ModuleList) instead of the non-existent
   `model.eot_attention` (singular).
2. Reports BOTH the canonical column-marginal entropy used in the loss
   (`mean(P, dim=row)` — matches dispersion_loss) AND the renormalised
   probability-distribution entropy (sum-to-1) so the paper number is
   unambiguous.
3. Crops the padded latent grid to the true latent shape (H/16, W/16)
   BEFORE marginalising, to avoid biased entropy from reflect-padded
   regions that don't carry real content.

Usage
-----
    python analyze/ablation_dictionary_utils.py \
        --checkpoint-with-disp checkpoints/full/best.pth.tar \
        --checkpoint-no-disp   checkpoints/no_disp/best.pth.tar \
        --dataset              /data/kodak \
        -o dictionary_utilization.pdf \
        --cuda
"""

import argparse
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import iter_dataset, load_model

# Dictionary size — must match model.QueryDictionaryGenerator.dict_num.
DICT_NUM = 128
MAX_ENTROPY_NATS = math.log(DICT_NUM)


def collect_routing_plans(model, x_padded: torch.Tensor) -> list[torch.Tensor]:
    """
    Run a forward pass and return the per-slice transport plans.

    Returns a list of length `num_slices`; each element has shape
    (B, padded_HW, N_d) and is on CPU.
    """
    plans: list[torch.Tensor] = []

    def make_hook(slice_idx: int):
        def hook_fn(module, inp, out):
            # The plan is stored on the module by UnifiedDictionaryAttention
            # under `attn_probs` (see dictionary_blocks.py).
            if hasattr(module, "attn_probs") and module.attn_probs is not None:
                plans.append(module.attn_probs.detach().cpu())

        return hook_fn

    # Register one hook per slice (model.eot_attentions is a ModuleList).
    handles = []
    for i, attn_module in enumerate(model.eot_attentions):
        handles.append(attn_module.register_forward_hook(make_hook(i)))

    try:
        with torch.no_grad():
            _ = model(x_padded)
    finally:
        for h in handles:
            h.remove()

    return plans


def per_image_entropy(
    plans: list[torch.Tensor],
    latent_h_pad: int,
    latent_w_pad: int,
    true_h: int,
    true_w: int,
) -> dict:
    """
    Returns a dict with two entropy variants per image:

      H_loss : entropy of  mean_i(P_ij)  over rows — matches dispersion loss
               formulation in dictionary_blocks.py. NOT a probability
               distribution under UEOT (sum != 1) but is the quantity the
               training objective optimises.
      H_norm : entropy of  bar_m / sum(bar_m)  — proper probability with
               sum=1, useful for direct comparison with log(N_d) max.
    """
    H_loss_per_slice = []
    H_norm_per_slice = []
    sum_per_slice = []

    for P in plans:
        B, _, N = P.shape
        # Reshape to spatial grid; padded_HW must equal latent_h_pad * latent_w_pad.
        assert (
            P.shape[1] == latent_h_pad * latent_w_pad
        ), f"plan shape {P.shape} != ({latent_h_pad}*{latent_w_pad})"
        P_spatial = P.view(B, latent_h_pad, latent_w_pad, N)
        # Crop padding before marginalising.
        P_cropped = P_spatial[:, :true_h, :true_w, :]
        # Flatten true spatial → (B, HW_true, N).
        P_flat = P_cropped.reshape(B, -1, N)

        # 1. Loss-aligned: bar_m = mean over rows.
        bar_m_loss = P_flat.mean(dim=1)  # (B, N)
        # Add eps for log stability.
        eps = 1e-12
        H_loss = -(bar_m_loss * (bar_m_loss + eps).log()).sum(dim=1).mean().item()

        # 2. Normalised probability distribution.
        bar_m_sum = bar_m_loss.sum(dim=1, keepdim=True).clamp_min(eps)
        bar_m_prob = bar_m_loss / bar_m_sum
        H_norm = -(bar_m_prob * (bar_m_prob + eps).log()).sum(dim=1).mean().item()

        H_loss_per_slice.append(H_loss)
        H_norm_per_slice.append(H_norm)
        sum_per_slice.append(bar_m_loss.sum(dim=1).mean().item())

    return {
        "H_loss_mean": float(np.mean(H_loss_per_slice)),
        "H_norm_mean": float(np.mean(H_norm_per_slice)),
        "marginal_total_mean": float(np.mean(sum_per_slice)),
        "H_loss_per_slice": H_loss_per_slice,
        "H_norm_per_slice": H_norm_per_slice,
    }


def evaluate_dataset(model, dataset_dir: str, device: str) -> list[dict]:
    """Returns list of per-image entropy dicts."""
    results = []
    for path, _x, x_padded, H, W in iter_dataset(dataset_dir, device=device):
        plans = collect_routing_plans(model, x_padded)
        if not plans:
            print(f"  WARN: no plans captured for {path}")
            continue
        latent_h_pad = x_padded.shape[-2] // 16
        latent_w_pad = x_padded.shape[-1] // 16
        true_h = H // 16
        true_w = W // 16
        entropies = per_image_entropy(plans, latent_h_pad, latent_w_pad, true_h, true_w)
        entropies["path"] = path
        results.append(entropies)
        print(
            f"  {path.split('/')[-1]:>30s}: "
            f"H_loss={entropies['H_loss_mean']:.4f}  "
            f"H_norm={entropies['H_norm_mean']:.4f}  "
            f"sum_marginal={entropies['marginal_total_mean']:.4f}"
        )
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-with-disp", type=str, required=True)
    p.add_argument("--checkpoint-no-disp", type=str, required=True)
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--routing-mode", type=str, default="unbalanced_eot")
    p.add_argument("--ot-eps", type=float, default=0.1)
    p.add_argument("--sinkhorn-iters", type=int, default=20)
    p.add_argument("-o", "--output", type=str, default="dictionary_utilization.pdf")
    p.add_argument("--json-out", type=str, default=None)
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    print("== Evaluating WITH dispersion ==")
    m_with = load_model(
        args.checkpoint_with_disp,
        device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        update_for_inference=False,  # forward() only, faster
    )
    res_with = evaluate_dataset(m_with, args.dataset, device)

    print("\n== Evaluating WITHOUT dispersion ==")
    m_no = load_model(
        args.checkpoint_no_disp,
        device,
        routing_mode=args.routing_mode,
        ot_eps=args.ot_eps,
        sinkhorn_iters=args.sinkhorn_iters,
        update_for_inference=False,
    )
    res_no = evaluate_dataset(m_no, args.dataset, device)

    H_norm_with = [r["H_norm_mean"] for r in res_with]
    H_norm_no = [r["H_norm_mean"] for r in res_no]

    print(
        f"\n=== Summary (probability-normalised entropy, max={MAX_ENTROPY_NATS:.4f}) ==="
    )
    print(
        f"With dispersion:    mean H_norm = {np.mean(H_norm_with):.4f} nats  "
        f"({np.mean(H_norm_with)/MAX_ENTROPY_NATS*100:.1f}% of max)"
    )
    print(
        f"Without dispersion: mean H_norm = {np.mean(H_norm_no):.4f} nats  "
        f"({np.mean(H_norm_no)/MAX_ENTROPY_NATS*100:.1f}% of max)"
    )
    delta = np.mean(H_norm_with) - np.mean(H_norm_no)
    print(f"Δ = {delta:+.4f} nats (positive = dispersion helps)")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    n = max(len(H_norm_with), len(H_norm_no))
    x_axis = np.arange(1, n + 1)
    axes[0].bar(
        x_axis - 0.2, H_norm_with, width=0.4, label="With disp.", color="steelblue"
    )
    axes[0].bar(
        x_axis + 0.2, H_norm_no, width=0.4, label="Without disp.", color="salmon"
    )
    axes[0].axhline(
        MAX_ENTROPY_NATS,
        color="gray",
        ls="--",
        alpha=0.7,
        label=f"Max H = log({DICT_NUM})",
    )
    axes[0].set_xlabel("Image index")
    axes[0].set_ylabel("Token-utilisation entropy (nats)")
    axes[0].set_title("Per-image dictionary utilisation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(H_norm_with, bins=10, alpha=0.7, label="With disp.", color="steelblue")
    axes[1].hist(H_norm_no, bins=10, alpha=0.7, label="Without disp.", color="salmon")
    axes[1].axvline(np.mean(H_norm_with), color="blue", ls="--")
    axes[1].axvline(np.mean(H_norm_no), color="red", ls="--")
    axes[1].set_xlabel("Token-utilisation entropy (nats)")
    axes[1].set_ylabel("Image count")
    axes[1].set_title("Distribution across dataset")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure → {args.output}")

    # ── JSON dump ────────────────────────────────────────────────────────
    if args.json_out:
        json_payload = {
            "dict_num": DICT_NUM,
            "max_entropy_nats": MAX_ENTROPY_NATS,
            "with_disp": {
                "mean_H_norm": float(np.mean(H_norm_with)),
                "mean_H_loss": float(np.mean([r["H_loss_mean"] for r in res_with])),
                "per_image": res_with,
            },
            "no_disp": {
                "mean_H_norm": float(np.mean(H_norm_no)),
                "mean_H_loss": float(np.mean([r["H_loss_mean"] for r in res_no])),
                "per_image": res_no,
            },
            "delta_H_norm": float(delta),
        }
        with open(args.json_out, "w") as f:
            json.dump(json_payload, f, indent=2)
        print(f"Saved JSON → {args.json_out}")


if __name__ == "__main__":
    main()
