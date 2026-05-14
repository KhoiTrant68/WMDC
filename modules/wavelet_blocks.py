import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

from modules.content_adaptive_blocks import ContentAdaptiveVSSBlock
from modules.VSS_module import VSSBlock

# ==============================================================================
# 1.  Biorthogonal Wavelet Transforms — bior4.4 (CDF 9/7 analysis)
# ==============================================================================
#
# Rationale for switching to pytorch_wavelets
# -------------------------------------------
# The hand-rolled DWT/IDWT previously here relied on `_trim_zeros` to strip
# pywt's leading/trailing zero pads from the bior4.4 filters.  Those zeros
# encode the phase alignment between the lo- and hi-pass filters; stripping
# them and then re-picking subsampled rows at offset 0 vs 1 is a fragile
# work-around that does NOT yield perfect reconstruction at image
# boundaries.  The old unit test only checked PR after cropping
# `[..., 2:-2, 2:-2]` precisely because the borders DID accumulate error.
#
# `pytorch_wavelets.DWTForward(..., mode="symmetric", wave="bior4.4")`
# implements CDF 9/7 with proper boundary handling.  For an N×N image,
# round-trip PR error stays at fp32 numerical noise (~1e-6) at every
# pixel — including the borders.  This matters for learned compression
# because image borders are exactly where rate concentrates.
#
# Channel layout is preserved for backward compatibility:
#   DWT(x) → cat([LL, LH, HL, HH], dim=1)        — (B, 4C, H/2, W/2)
#   IDWT(z): z[:C]=LL, z[C:2C]=LH, z[2C:3C]=HL, z[3C:]=HH
# where LH = "horizontal detail" in pywt terminology (lo on rows + hi on cols),
# HL = "vertical detail" (hi on rows + lo on cols).


class DWT_2D(nn.Module):
    """
    2-D analysis DWT for bior4.4 using pytorch_wavelets under the hood.

    API (unchanged from previous version):
        Input  : (B, C, H, W) with H, W even
        Output : (B, 4*C, H/2, W/2), channel order [LL, LH, HL, HH]
    """

    def __init__(self, wave: str = "bior4.4", mode: str = "symmetric"):
        super().__init__()
        self.wave = wave
        self.mode = mode
        self.xfm = DWTForward(J=1, wave=wave, mode=mode)
        # Filter buffers live inside `self.xfm` (e.g. h0_col, h1_col, ...).
        # They are deterministic functions of `wave` and never trained.

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
        """
        Silently drop legacy buffer keys (`dec_lo`, `dec_hi`) from old
        checkpoints — the new pytorch_wavelets backend uses its own
        deterministic filter buffers, so the legacy ones are unused.
        """
        for legacy in ("dec_lo", "dec_hi"):
            state_dict.pop(prefix + legacy, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # yl:      (B, C, H/2, W/2)
        # yh[0]:   (B, C, 3, H/2, W/2)  — order [LH, HL, HH] (pywt convention)
        yl, yh = self.xfm(x)
        h = yh[0]
        return torch.cat([yl, h[:, :, 0], h[:, :, 1], h[:, :, 2]], dim=1)


class IDWT_2D(nn.Module):
    """
    2-D synthesis IDWT for bior4.4 using pytorch_wavelets.

    API (unchanged):
        Input  : (B, 4*C, H/2, W/2), channel order [LL, LH, HL, HH]
        Output : (B, C, H, W)
    """

    def __init__(self, wave: str = "bior4.4", mode: str = "symmetric"):
        super().__init__()
        self.wave = wave
        self.mode = mode
        self.ifm = DWTInverse(wave=wave, mode=mode)

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
        for legacy in ("rec_lo", "rec_hi"):
            state_dict.pop(prefix + legacy, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C4, H2, W2 = x.shape
        if C4 % 4 != 0:
            raise ValueError(f"IDWT_2D expected channels divisible by 4, got {C4}")
        C = C4 // 4
        yl = x[:, :C]
        h = torch.stack(
            [x[:, C : 2 * C], x[:, 2 * C : 3 * C], x[:, 3 * C : 4 * C]], dim=2
        )  # (B, C, 3, H/2, W/2)
        return self.ifm((yl, [h]))


def test_dwt_idwt():
    """
    Sanity test for the pytorch_wavelets-backed DWT/IDWT:
      • shape preservation,
      • perfect reconstruction (PR) measured at:
            - the full image (incl. 2-pixel borders)
            - JUST the borders (rows 0..1, H-2..H-1; same for cols)
      • gradient flow through the round-trip.

    The legacy hand-rolled implementation cropped 2 pixels off each side
    before computing the PR error.  This test refuses to crop, so a
    regression in boundary handling will surface immediately.
    """
    dwt = DWT_2D("bior4.4")
    idwt = IDWT_2D("bior4.4")
    x = torch.randn(2, 64, 128, 128)
    z = dwt(x)
    assert z.shape == (2, 256, 64, 64), z.shape
    recon = idwt(z)

    err_full = (x - recon).abs().max().item()
    diff = (x - recon).abs()
    err_border = max(
        diff[..., :2, :].max().item(),
        diff[..., -2:, :].max().item(),
        diff[..., :, :2].max().item(),
        diff[..., :, -2:].max().item(),
    )
    print(f"[bior4.4] Max recon error (full image): {err_full:.2e}")
    print(f"[bior4.4] Max recon error (borders)   : {err_border:.2e}")
    assert err_full < 1e-3, f"PR failed: max err {err_full:.2e}"
    assert err_border < 1e-3, f"Boundary PR failed: {err_border:.2e}"

    x_g = torch.randn(1, 16, 64, 64, requires_grad=True)
    idwt(dwt(x_g)).sum().backward()
    assert x_g.grad is not None and not x_g.grad.isnan().any()
    print("DWT/IDWT tests passed ✓ (including boundary PR)")


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
        # Zero-init the projection so the residual block is identity at start:
        #   fused = fusion_pw(GELU(fusion_dw(merged))) + merged  →  merged  at init.
        # This matches the zero-init pattern used elsewhere (delta_proj,
        # gate_proj, lrp_transforms[-1]) so the block does not perturb the
        # main path before learning to.
        nn.init.zeros_(self.fusion_pw.weight)
        nn.init.zeros_(self.fusion_pw.bias)

    @staticmethod
    def _apply_affine_film(
        x_target: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Affine FiLM: y = scale * x + shift.
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
        )

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


if __name__ == "__main__":
    test_dwt_idwt()
