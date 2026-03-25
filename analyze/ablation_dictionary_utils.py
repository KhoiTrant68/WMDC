"""
analyze/ablation_dictionary_utilization.py
===========================================
Measures dictionary token utilisation (entropy of target marginal) across a
test dataset, with and without dispersion loss.  Required experiment E4.

Computes per-image token utilisation entropy and its distribution across the
dataset.  Compares two checkpoints: one trained with dispersion loss and one
without (or the same model with dispersion_weight=0 as ablation).

Usage:
    python analyze/ablation_dictionary_utilization.py \
        --checkpoint-with-disp  checkpoints/lambda_0.013_mse/checkpoint_best.pth.tar \
        --checkpoint-no-disp    checkpoints/ablation_no_disp/checkpoint_best.pth.tar \
        --dataset               /data/kodak \
        -o dictionary_utilization.pdf
"""

import argparse
import math
import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.insert(0, parent_dir)

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from models.WMDC import WMDC


def compute_utilization_entropy(
    model, x_padded: torch.Tensor, orig_h: int, orig_w: int
) -> float:
    """
    Compute Shannon entropy of the dictionary token utilisation marginal
    for a single image.

    Higher entropy → more uniform token usage → better dictionary utilisation.
    Max entropy = log(N) = log(128) ≈ 4.85 bits (all tokens equally used).

    Returns entropy in nats.
    """
    model.eval()
    entropies = []

    # Capture routing plans from all slices via hook.
    plans = []

    def hook_fn(module, input, output):
        plans.append(module.attn_probs.detach().cpu())

    hook = model.eot_attention.register_forward_hook(hook_fn)
    try:
        with torch.no_grad():
            _ = model(x_padded)
    finally:
        hook.remove()

    true_latent_h = orig_h // 16
    true_latent_w = orig_w // 16

    for P in plans:
        # P is shape (B, padded_HW, N)
        B, _, N = P.shape

        # Reshape to spatial grid: (B, padded_H, padded_W, N)
        # Note: In WMDC, HW is derived from the latent shape.
        latent_h_pad = x_padded.size(2) // 16
        latent_w_pad = x_padded.size(3) // 16

        P_spatial = P.view(B, latent_h_pad, latent_w_pad, N)

        # Crop out the padding!
        P_cropped = P_spatial[:, :true_latent_h, :true_latent_w, :].reshape(B, -1, N)

        # Now calculate the true target marginal
        target_marginal = P_cropped.sum(dim=1)  # (B, N)
        target_marginal = target_marginal / (
            target_marginal.sum(dim=1, keepdim=True) + 1e-8
        )

        H = -(target_marginal * (target_marginal + 1e-8).log()).sum(dim=1).mean()
        entropies.append(H.item())

    return float(np.mean(entropies))


def evaluate_dataset(model, dataset_dir: str, device: str) -> list:
    """Returns list of per-image utilisation entropies."""
    image_paths = sorted(
        [
            os.path.join(dataset_dir, f)
            for f in os.listdir(dataset_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    )

    entropies = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB")
        x = transforms.ToTensor()(img).unsqueeze(0).to(device)
        H, W = x.size(2), x.size(3)
        pad_h = (64 - H % 64) % 64
        pad_w = (64 - W % 64) % 64
        x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        ent = compute_utilization_entropy(model, x_padded, H, W)
        entropies.append(ent)
        print(f"  {os.path.basename(img_path)}: H = {ent:.4f} nats")

    return entropies


def load_model(
    checkpoint_path: str,
    device: str,
    routing_mode: str = "unbalanced_eot",
    ot_eps: float = 0.1,
    sinkhorn_iters: int = 20,
) -> WMDC:
    model = WMDC(
        N=192,
        M=320,
        num_slices=5,
        routing_mode=routing_mode,
        ot_eps=ot_eps,
        sinkhorn_iters=sinkhorn_iters,
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-with-disp", type=str, required=True)
    parser.add_argument("--checkpoint-no-disp", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--routing_mode",
        type=str,
        default="unbalanced_eot",
        choices=["softmax", "balanced_eot", "unbalanced_eot"],
    )
    parser.add_argument("--ot-eps", type=float, default=0.1)
    parser.add_argument("--sinkhorn-iters", type=int, default=20)
    parser.add_argument(
        "-o", "--output", type=str, default="dictionary_utilization.pdf"
    )
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    max_entropy = math.log(128)  # 4.852 nats for N=128 uniform distribution

    print("Evaluating model WITH dispersion loss...")
    model_with = load_model(
        args.checkpoint_with_disp,
        device,
        args.routing_mode,
        args.ot_eps,
        args.sinkhorn_iters,
    )
    ent_with = evaluate_dataset(model_with, args.dataset, device)

    print("\nEvaluating model WITHOUT dispersion loss...")
    model_no = load_model(
        args.checkpoint_no_disp,
        device,
        args.routing_mode,
        args.ot_eps,
        args.sinkhorn_iters,
    )
    ent_no = evaluate_dataset(model_no, args.dataset, device)

    print(
        f"\nWith dispersion:    mean H = {np.mean(ent_with):.4f} nats "
        f"({np.mean(ent_with)/max_entropy*100:.1f}% of max)"
    )
    print(
        f"Without dispersion: mean H = {np.mean(ent_no):.4f} nats "
        f"({np.mean(ent_no)/max_entropy*100:.1f}% of max)"
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    x_vals = list(range(1, len(ent_with) + 1))
    axes[0].bar(x_vals, ent_with, label="With disp.", alpha=0.7, color="steelblue")
    axes[0].bar(x_vals, ent_no, label="Without disp.", alpha=0.7, color="salmon")
    axes[0].axhline(
        y=max_entropy,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=f"Max H={max_entropy:.2f}",
    )
    axes[0].set_xlabel("Image index")
    axes[0].set_ylabel("Token utilisation entropy (nats)")
    axes[0].set_title("Per-image dictionary utilisation")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(ent_with, bins=10, alpha=0.7, label="With disp.", color="steelblue")
    axes[1].hist(ent_no, bins=10, alpha=0.7, label="Without disp.", color="salmon")
    axes[1].axvline(x=np.mean(ent_with), color="blue", linestyle="--", alpha=0.8)
    axes[1].axvline(x=np.mean(ent_no), color="red", linestyle="--", alpha=0.8)
    axes[1].set_xlabel("Token utilisation entropy (nats)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution across dataset")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
