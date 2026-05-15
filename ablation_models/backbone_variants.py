"""
ablation_models/backbone_variants.py
=====================================
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────
# Variant 1 — Residual CNN
# ──────────────────────────────────────────────────────────────────────────
class ResidualCNNBlock(nn.Module):
    """Plain residual CNN block: conv → GELU → conv + skip."""

    def __init__(self, dim: int, hidden_mult: float = 1.0, drop_path: float = 0.0):
        super().__init__()
        h = max(int(dim * hidden_mult), dim)
        self.conv1 = nn.Conv2d(dim, h, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(h, dim, kernel_size=3, padding=1)
        self.act = nn.GELU()
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.act(self.conv1(x)))


# ──────────────────────────────────────────────────────────────────────────
# Variant 2 — Swin Transformer block (windowed self-attention)
# ──────────────────────────────────────────────────────────────────────────
class WindowAttention(nn.Module):
    """Window-based multi-head self-attention with relative position bias."""

    def __init__(self, dim: int, num_heads: int, window_size: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(
                f"WindowAttention: dim={dim} must be divisible by num_heads={num_heads}"
            )
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.rel_pos_bias = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.rel_pos_bias, std=0.02)

        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flat = coords.flatten(1)
        rel_coords = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel_coords = rel_coords.permute(1, 2, 0).contiguous()
        rel_coords[:, :, 0] += window_size - 1
        rel_coords[:, :, 1] += window_size - 1
        rel_coords[:, :, 0] *= 2 * window_size - 1
        rel_index = rel_coords.sum(-1)
        self.register_buffer("rel_index", rel_index)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        bias = self.rel_pos_bias[self.rel_index.view(-1)].view(N, N, -1)
        attn = attn + bias.permute(2, 0, 1).unsqueeze(0)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


def _window_partition(x: torch.Tensor, win: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // win, win, W // win, win, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, win, win, C)


def _window_reverse(x: torch.Tensor, win: int, H: int, W: int) -> torch.Tensor:
    B = int(x.shape[0] / (H * W / win / win))
    x = x.view(B, H // win, W // win, win, win, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


class SwinBlock(nn.Module):
    """Single Swin block (no shifted window; sufficient as a backbone unit)."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        window_size: int = 8,
        drop_path: float = 0.0,
    ):
        super().__init__()
        # Guard: dim must be divisible by num_heads
        if dim % num_heads != 0:
            # Find largest valid num_heads ≤ requested
            for nh in range(num_heads, 0, -1):
                if dim % nh == 0:
                    num_heads = nh
                    break
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        win = self.window_size
        pad_h = (win - H % win) % win
        pad_w = (win - W % win) % win
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = x.permute(0, 2, 3, 1)
        Hp, Wp = x.shape[1], x.shape[2]

        shortcut = x
        x = self.norm1(x)
        windows = _window_partition(x, win).view(-1, win * win, C)
        windows = self.attn(windows)
        x = _window_reverse(windows.view(-1, win, win, C), win, Hp, Wp)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        x = x.permute(0, 3, 1, 2)
        if pad_h or pad_w:
            x = x[:, :, :H, :W]
        return x


# ──────────────────────────────────────────────────────────────────────────
# Variant 3 — Standard SS2D Mamba block (no wavelet)
# ──────────────────────────────────────────────────────────────────────────
class SS2DBlock(nn.Module):
    """Standard Vision Mamba block — no wavelet decomposition."""

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        from modules.VSS_module import VSSBlock

        self.vss = VSSBlock(hidden_dim=dim, drop_path=drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vss(x)


# ──────────────────────────────────────────────────────────────────────────
# Variant 4 — FDM with REVERSED FiLM direction (controlled ablation)
# ──────────────────────────────────────────────────────────────────────────
class FDMReversedBlock(nn.Module):
    """
    Same wavelet pathway as FDM but FiLM direction reversed:
    HF context modulates LL instead of LL context modulating HF bands.
    ------------------------
    fusion_pw is now zero-initialized (not identity-matrix initialized).
    This matches FrequencyDisentangledMamba's initialization exactly so
    both blocks start from a true identity mapping at t=0:

        fused = fusion_pw(GELU(fusion_dw(merged))) + merged
              = zeros + merged   (at init, since fusion_pw.weight = 0)
              = merged           ← identity mapping ✓

    The previous identity-matrix init on fusion_pw gave:
        fused = fusion_pw(GELU(fusion_dw(merged))) + merged
              = GELU(random_dw(merged)) + merged  ← NOT identity ✗
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        from modules.VSS_module import VSSBlock
        from modules.wavelet_blocks import DWT_2D, IDWT_2D

        self.dim = dim
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.vss = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        self.film_scale = nn.Conv2d(dim, dim, kernel_size=1)
        self.film_shift = nn.Conv2d(dim, dim, kernel_size=1)
        nn.init.zeros_(self.film_scale.weight)
        nn.init.constant_(self.film_scale.bias, math.log(math.e - 1.0))
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

        self.fusion_dw = nn.Conv2d(dim * 4, dim * 4, 3, padding=1, groups=dim * 4)
        self.fusion_act = nn.GELU()
        self.fusion_pw = nn.Conv2d(dim * 4, dim * 4, 1)

        nn.init.zeros_(self.fusion_pw.weight)
        nn.init.zeros_(self.fusion_pw.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        subbands = self.dwt(x)
        ll, lh, hl, hh = subbands.chunk(4, dim=1)

        ll_proc = self.vss(ll)

        hf_context = (lh + hl + hh) / 3.0
        gamma = F.softplus(self.film_scale(hf_context))
        beta = self.film_shift(hf_context)
        ll_modulated = ll_proc * gamma + beta

        cat = torch.cat([ll_modulated, lh, hl, hh], dim=1)
        cat = self.fusion_pw(self.fusion_act(self.fusion_dw(cat))) + cat
        return self.idwt(cat)


# ──────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────
def build_backbone(name: str, dim: int, **kwargs) -> nn.Module:
    name = name.lower()
    if name == "cnn":
        return ResidualCNNBlock(dim, **kwargs)
    if name == "swin":
        return SwinBlock(dim, **kwargs)
    if name == "ss2d":
        return SS2DBlock(dim, **kwargs)
    if name in ("fdm_reversed", "fdm_rev", "reversed"):
        return FDMReversedBlock(dim, **kwargs)
    if name == "fdm":
        from modules.wavelet_blocks import FrequencyDisentangledMamba

        return FrequencyDisentangledMamba(dim, **kwargs)
    raise ValueError(f"Unknown backbone: {name!r}")


__all__ = [
    "ResidualCNNBlock",
    "SwinBlock",
    "SS2DBlock",
    "FDMReversedBlock",
    "build_backbone",
]
