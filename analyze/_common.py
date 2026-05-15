"""
analyze/_common.py
==================
Shared utilities used by every analysis script.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Iterator, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models.WMDC import WMDC  # noqa: E402

BD_RATE_LAMBDAS_FULL = [0.0018, 0.0035, 0.0067, 0.013, 0.025, 0.0483]
BD_RATE_LAMBDAS_ABLATION = [0.0035, 0.013, 0.0483]


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    *,
    routing_mode: str = "unbalanced_eot",
    ot_eps: float = 0.1,
    sinkhorn_iters: int = 20,
    N: int = 192,
    M: int = 320,
    num_slices: int = 5,
    backbone: str = "fdm",
    use_dense_concat: bool = False,
    memory_init: str = "bootstrap",
    use_content_adaptive: bool = False,
    cluster_num: int = 8,
    update_for_inference: bool = True,
    strict_load: bool = True,
) -> WMDC:
    """
    Build a WMDC, load a checkpoint, and put it in eval mode.

    Parameters
    ----------
    strict_load : if True (default), missing/unexpected keys raise an error.
                  Set to False only for legacy checkpoint migration — never for
                  paper-level evaluation, where architecture mismatch must be
                  caught immediately rather than silently ignored.
    """
    model = WMDC(
        N=N,
        M=M,
        num_slices=num_slices,
        routing_mode=routing_mode,
        ot_eps=ot_eps,
        sinkhorn_iters=sinkhorn_iters,
        backbone=backbone,
        use_dense_concat=use_dense_concat,
        memory_init=memory_init,
        use_content_adaptive=use_content_adaptive,
        cluster_num=cluster_num,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict_load)
    if missing:
        print(f"[load_model] missing keys: {len(missing)} (showing 5): {missing[:5]}")
    if unexpected:
        print(
            f"[load_model] unexpected keys: {len(unexpected)} (showing 5): {unexpected[:5]}"
        )

    model.eval()
    if update_for_inference:
        model.update(force=True)
    return model


def pad_to_64(x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
    H, W = x.shape[-2:]
    pad_h = (64 - H % 64) % 64
    pad_w = (64 - W % 64) % 64
    if pad_h == 0 and pad_w == 0:
        return x, 0, 0
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), pad_h, pad_w


def load_image(
    image_path: str, device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    img = Image.open(image_path).convert("RGB")
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    H, W = x.shape[-2:]
    x_padded, _, _ = pad_to_64(x)
    return x, x_padded, H, W


def _byte_size(strings) -> int:
    """Recursively count bytes in nested (list/tuple/bytes) string container."""
    if isinstance(strings, (bytes, bytearray)):
        return len(strings)
    if isinstance(strings, (list, tuple)):
        return sum(_byte_size(s) for s in strings)
    return 0


def compute_actual_bpp(
    strings,
    num_pixels: int,
    *,
    include_header: bool = True,
) -> float:
    bits = _byte_size(strings) * 8
    if include_header:
        bits += 2 * 16
        bits += 2 * 8
    return bits / num_pixels


def compute_psnr(x: torch.Tensor, x_hat: torch.Tensor) -> float:
    mse = F.mse_loss(x.float(), x_hat.float()).item()
    if mse <= 0.0:
        return 100.0
    return -10.0 * math.log10(mse)


def iter_dataset(
    dataset_dir: str,
    device: str = "cuda",
) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor, int, int]]:
    image_paths = sorted(
        os.path.join(dataset_dir, f)
        for f in os.listdir(dataset_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    )
    if not image_paths:
        raise RuntimeError(f"No images in {dataset_dir}")
    for p in image_paths:
        x, x_padded, H, W = load_image(p, device=device)
        yield p, x, x_padded, H, W


def evaluate_codec(
    model: WMDC,
    x: torch.Tensor,
    x_padded: torch.Tensor,
    H: int,
    W: int,
    *,
    use_actual_codec: bool = True,
) -> dict:
    num_pixels = H * W
    with torch.no_grad():
        if use_actual_codec:
            out_enc = model.compress(x_padded)
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            x_hat = out_dec["x_hat"][:, :, :H, :W].clamp(0, 1)
            bpp = compute_actual_bpp(out_enc["strings"], num_pixels)
            mode = "actual"
        else:
            out_net = model(x_padded)
            x_hat = out_net["x_hat"][:, :, :H, :W].clamp(0, 1)
            bpp = sum(
                torch.log(lh).sum() for lh in out_net["likelihoods"].values()
            ).item() / (-math.log(2) * num_pixels)
            mode = "estimated"

    return {
        "psnr": compute_psnr(x, x_hat),
        "bpp": bpp,
        "num_pixels": num_pixels,
        "mode": mode,
    }
