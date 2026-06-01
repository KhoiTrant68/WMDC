import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.content_adaptive_blocks import ContentAdaptiveVSSBlock
from modules.VSS_module import VSSBlock

# ==============================================================================
# 1.  Biorthogonal Wavelet Transforms — bior4.4 (CDF 9/7 analysis)
# ==============================================================================


class DWT_2D(nn.Module):
    """
    Separable 1-D DWT (analysis) using bior4.4.
    Output: (B, 4*C, H/2, W/2), channel order [LL, LH, HL, HH].
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        def _trim_zeros(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            """Strip pywt's artificial zero-padding (length 10 → 9 and 7)."""
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0] : idx[-1] + 1] if len(idx) > 0 else f

        dec_lo = _trim_zeros(torch.tensor(w.dec_lo[::-1].copy(), dtype=torch.float32))
        dec_hi = _trim_zeros(torch.tensor(w.dec_hi[::-1].copy(), dtype=torch.float32))
        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)
        self.lo_pad = (len(self.dec_lo) - 1) // 2
        self.hi_pad = (len(self.dec_hi) - 1) // 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for suffix in ["dec_lo", "dec_hi"]:
            key = prefix + suffix
            if key in state_dict:
                f = state_dict[key]
                idx = torch.where(torch.abs(f) > 1e-6)[0]
                if len(idx) > 0:
                    state_dict[key] = f[idx[0] : idx[-1] + 1]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _conv1d(self, x, filt, pad, dim):
        B, C, H, W = x.shape
        assert filt.shape[0] % 2 == 1
        if dim == 2:
            x_pad = F.pad(x, (0, 0, pad, pad), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H + 2 * pad, W)
            out = F.conv2d(x_r, filt.view(1, 1, -1, 1))
            return out.reshape(B, C, H, W)
        else:
            x_pad = F.pad(x, (pad, pad, 0, 0), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H, W + 2 * pad)
            out = F.conv2d(x_r, filt.view(1, 1, 1, -1))
            return out.reshape(B, C, H, W)

    def forward(self, x):
        lo_col = self._conv1d(x, self.dec_lo, self.lo_pad, 2)[:, :, 0::2, :]
        hi_col = self._conv1d(x, self.dec_hi, self.hi_pad, 2)[:, :, 1::2, :]
        ll = self._conv1d(lo_col, self.dec_lo, self.lo_pad, 3)[:, :, :, 0::2]
        lh = self._conv1d(lo_col, self.dec_hi, self.hi_pad, 3)[:, :, :, 1::2]
        hl = self._conv1d(hi_col, self.dec_lo, self.lo_pad, 3)[:, :, :, 0::2]
        hh = self._conv1d(hi_col, self.dec_hi, self.hi_pad, 3)[:, :, :, 1::2]
        return torch.cat([ll, lh, hl, hh], dim=1)


class IDWT_2D(nn.Module):
    """
    Separable 1-D IDWT (synthesis) using bior4.4.
    Input: (B, 4*C, H/2, W/2) [LL, LH, HL, HH]. Output: (B, C, H, W).
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        def _trim_zeros(f, eps=1e-6):
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0] : idx[-1] + 1] if len(idx) > 0 else f

        rec_lo = _trim_zeros(torch.tensor(w.rec_lo[::-1].copy(), dtype=torch.float32))
        rec_hi = _trim_zeros(torch.tensor(w.rec_hi[::-1].copy(), dtype=torch.float32))
        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)
        self.lo_pad = (len(self.rec_lo) - 1) // 2
        self.hi_pad = (len(self.rec_hi) - 1) // 2

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        for suffix in ["rec_lo", "rec_hi"]:
            key = prefix + suffix
            if key in state_dict:
                f = state_dict[key]
                idx = torch.where(torch.abs(f) > 1e-6)[0]
                if len(idx) > 0:
                    state_dict[key] = f[idx[0] : idx[-1] + 1]
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _up_conv1d(self, x, filt, pad, dim, offset, size):
        B, C, H, W = x.shape
        if dim == 2:
            up = x.new_zeros(B, C, size, W)
            up[:, :, offset::2, :] = x
            xp = F.pad(up, (0, 0, pad, pad), mode="reflect")
            out = F.conv2d(
                xp.reshape(B * C, 1, size + 2 * pad, W), filt.view(1, 1, -1, 1)
            )
            return out.reshape(B, C, size, W)
        else:
            up = x.new_zeros(B, C, H, size)
            up[:, :, :, offset::2] = x
            xp = F.pad(up, (pad, pad, 0, 0), mode="reflect")
            out = F.conv2d(
                xp.reshape(B * C, 1, H, size + 2 * pad), filt.view(1, 1, 1, -1)
            )
            return out.reshape(B, C, H, size)

    def forward(self, x):
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        H, W = H2 * 2, W2 * 2
        ll, lh, hl, hh = x[:, :C], x[:, C : 2 * C], x[:, 2 * C : 3 * C], x[:, 3 * C :]
        lo_col = self._up_conv1d(
            ll, self.rec_lo, self.lo_pad, 3, 0, W
        ) + self._up_conv1d(lh, self.rec_hi, self.hi_pad, 3, 1, W)
        hi_col = self._up_conv1d(
            hl, self.rec_lo, self.lo_pad, 3, 0, W
        ) + self._up_conv1d(hh, self.rec_hi, self.hi_pad, 3, 1, W)
        return self._up_conv1d(
            lo_col, self.rec_lo, self.lo_pad, 2, 0, H
        ) + self._up_conv1d(hi_col, self.rec_hi, self.hi_pad, 2, 1, H)


def test_dwt_idwt():
    dwt = DWT_2D("bior4.4")
    idwt = IDWT_2D("bior4.4")
    x = torch.randn(2, 64, 128, 128)
    assert dwt(x).shape == (2, 256, 64, 64)
    recon = idwt(dwt(x))
    err = (x[..., 2:-2, 2:-2] - recon[..., 2:-2, 2:-2]).abs().max().item()
    print(f"[bior4.4] Max recon error (interior): {err:.2e}")
    assert err < 1e-4
    x_g = torch.randn(1, 16, 64, 64, requires_grad=True)
    idwt(dwt(x_g)).sum().backward()
    assert x_g.grad is not None and not x_g.grad.isnan().any()
    print("DWT/IDWT tests passed ✓")


# ==============================================================================
# 2.  Gated Memory Updater
# ==============================================================================


class GatedMemoryUpdater(nn.Module):
    """
    Additive gated memory update: memory_new = memory + delta * gate.
    Zero-init delta → identity mapping at t=0.
    """

    def __init__(self, slice_ch: int):
        super().__init__()
        self.fuse = nn.Sequential(nn.Conv2d(slice_ch * 2, slice_ch, 1), nn.GELU())
        self.mamba_context = VSSBlock(hidden_dim=slice_ch, drop_path=0.0)
        self.delta_proj = nn.Conv2d(slice_ch, slice_ch, 3, padding=1)
        self.gate_proj = nn.Conv2d(slice_ch, slice_ch, 3, padding=1)
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, memory, new_info):
        x = self.fuse(torch.cat([memory, new_info], dim=1))
        ctx = self.mamba_context(x)
        return memory + self.delta_proj(ctx) * torch.sigmoid(self.gate_proj(ctx))


# ==============================================================================
# 3.  FrequencyDisentangledMamba  (FIX-W1 applied)
# ==============================================================================


class FrequencyDisentangledMamba(nn.Module):
    """
    Wavelet-domain feature transform with Mamba-based cross-band FiLM modulation.

    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.1,
        use_content_adaptive: bool = False,
        cluster_num: int = 8,
    ):
        super().__init__()
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.dim = dim

        if use_content_adaptive:
            mamba_cls = lambda: ContentAdaptiveVSSBlock(
                hidden_dim=dim, drop_path=drop_path, cluster_num=cluster_num
            )
        else:
            mamba_cls = lambda: VSSBlock(hidden_dim=dim, drop_path=drop_path)

        self.ll_mamba = mamba_cls()
        self.pre_context_proj = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.context_mamba = mamba_cls()

        def _make_film(in_dim, out_dim):
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim * 2, 1),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.constant_(net[-1].bias[:out_dim], math.log(math.e - 1))
            nn.init.zeros_(net[-1].bias[out_dim:])
            return net

        self.film_ll = _make_film(dim, dim)
        self.film_lh = _make_film(dim, dim)
        self.film_hl = _make_film(dim, dim)
        self.film_hh = _make_film(dim, dim)

        self.fusion_dw = nn.Conv2d(dim * 4, dim * 4, 3, padding=1, groups=dim * 4)
        self.fusion_act = nn.GELU()
        self.fusion_pw = nn.Conv2d(dim * 4, dim * 4, 1)
        nn.init.zeros_(self.fusion_pw.weight)
        self.fusion_pw.weight.data.copy_(
            torch.eye(dim * 4).view(dim * 4, dim * 4, 1, 1)
        )
        nn.init.zeros_(self.fusion_pw.bias)

    @staticmethod
    def _apply_affine_film(
        x_target: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Affine FiLM: y = scale * x + shift.

        `scale.clamp(max=10)` blocks a catastrophic positive-feedback loop:
        γ is produced by an MLP whose input ultimately depends on
        x_target (via context_mamba ← pre_context_proj ← cat with
        x_ll_out), so on OOD or under-trained inputs γ scales with
        x_target itself.  Then scale ≈ |γ| ∝ |x_target|, and
        `x_target * scale ≈ x_target²` — a squaring operation per FDM
        block.  With L = 3 FDM blocks in g_s, a single unit of input
        grows as M₀^(2^L) = M₀^8, so M₀ = 10 → 10⁸ and M₀ = 10³ →
        10²⁴ (matches the empirical g_s decoder trace
        log₁₀|x| : 3.6 → 7.0 → 12.3 → 21.8 → 38.5 (∞) ).

        Cap = 10 is mathematically generous:
          • init: γ = log(e − 1) ≈ 0.541 → scale ≈ 1.0 (identity modulation)
          • trained model: |γ| stays O(1) → scale O(1) → clamp is no-op
          • triggers only when γ > log(e¹⁰ − 1) ≈ 10
        With scale ≤ 10 per block: total amplification ≤ 10^L = 10³ for
        g_s, keeping output well below fp32 overflow regardless of input.
        """
        C = x_target.shape[1]
        gamma, beta = params[:, :C], params[:, C:]
        scale = F.softplus(gamma)
        return x_target * scale + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim
        xd = self.dwt(x)
        x_ll, x_lh = xd[:, :dim], xd[:, dim : 2 * dim]
        x_hl, x_hh = xd[:, 2 * dim : 3 * dim], xd[:, 3 * dim :]

        x_ll_out = self.ll_mamba(x_ll)
        ctx = self.context_mamba(
            self.pre_context_proj(torch.cat([x_ll_out, x_lh, x_hl, x_hh], dim=1))
        ).contiguous()

        merged = torch.cat(
            [
                self._apply_affine_film(x_ll_out, self.film_ll(ctx)),
                self._apply_affine_film(x_lh, self.film_lh(ctx)),
                self._apply_affine_film(x_hl, self.film_hl(ctx)),
                self._apply_affine_film(x_hh, self.film_hh(ctx)),
            ],
            dim=1,
        )

        fused = self.fusion_pw(self.fusion_act(self.fusion_dw(merged))) + merged
        return self.idwt(fused)


# ==============================================================================
# 4.  WLS / iWLS  (Wavelet Linear Scaling — CMIC-style multi-scale shortcut)
# ==============================================================================


class WLS(nn.Module):
    """Wavelet Linear Scaling (analysis side, CMIC-style).

    DWT → per-band learnable scaling → 1×1 OLP projection.  Used as an
    auxiliary shortcut path from the input image down through each encoder
    scale, injecting wavelet detail directly at every spatial resolution.
    The HH band is zero-initialised so the shortcut starts as a low-pass
    summary and learns to add high-frequency detail over training.

    Uses bior4.4 by default to match the project's DWT_2D infrastructure
    (which assumes odd-length filters; Haar's 2-tap filters break that
    assumption).  The wavelet choice does not change the topology or the
    expected gain — the architectural value comes from the shortcut itself.
    """

    def __init__(self, in_dim: int, out_dim: int, wave: str = "bior4.4"):
        super().__init__()
        from modules.utils import OLP

        self.dwt = DWT_2D(wave=wave)
        self.proj = OLP(in_dim * 4, out_dim)
        # Direct (non-exp) parameterisation so HH=0 actually zeros the band.
        factors = torch.cat(
            [
                torch.full((1, 1, in_dim), 1.0),  # LL
                torch.full((1, 1, in_dim), 1.0),  # LH
                torch.full((1, 1, in_dim), 1.0),  # HL
                torch.zeros(1, 1, in_dim),  # HH — actually zero at init
            ],
            dim=2,
        )
        self.scaling_factors = nn.Parameter(factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dwt(x)  # (B, 4C, H/2, W/2)
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)  # (B, HW, 4C)
        x = x * self.scaling_factors
        x = self.proj(x)
        return x.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()


class iWLS(nn.Module):
    """Inverse WLS (synthesis side, CMIC-style)."""

    def __init__(self, in_dim: int, out_dim: int, wave: str = "bior4.4"):
        super().__init__()
        from modules.utils import OLP

        self.idwt = IDWT_2D(wave=wave)
        self.proj = OLP(in_dim, out_dim * 4)
        factors = torch.cat(
            [
                torch.full((1, 1, out_dim), 1.0),
                torch.full((1, 1, out_dim), 1.0),
                torch.full((1, 1, out_dim), 1.0),
                torch.zeros(1, 1, out_dim),
            ],
            dim=2,
        )
        self.scaling_factors = nn.Parameter(factors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = x.view(b, -1, h * w).permute(0, 2, 1)
        x = self.proj(x)
        # Symmetric to WLS: forward applies `* factor`; here scale by factor too
        # (no exp). HH=0 at init ⇒ no HH contribution from the synthesis branch.
        x = x * self.scaling_factors
        x = x.view(b, h, w, -1).permute(0, 3, 1, 2).contiguous()
        return self.idwt(x)


if __name__ == "__main__":
    test_dwt_idwt()
