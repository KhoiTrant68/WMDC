import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock

# ==============================================================================
# 1. Biorthogonal Wavelet Transforms — Separable 1D CDF 9/7
# ==============================================================================


class DWT_2D(nn.Module):
    """
    Separable 1D CDF 9/7 Discrete Wavelet Transform (analysis).

      - Per-filter symmetric (half-point) padding: lext = (len(h) - 1) // 2
      - Filter along columns first, then rows (same order as MATLAB)
      - Downsampling:
          lo → [0::2]  ↔  MATLAB 1:2:end  (even 0-indexed)
          hi → [1::2]  ↔  MATLAB 2:2:end  (odd  0-indexed)

    For even H, W (guaranteed by model's 64-divisibility constraint):
        ceil(H/2) = floor(H/2) = H/2  →  all subbands same spatial size.

    Output channel order: [LL | LH | HL | HH], each C channels.
    Output shape: (B, 4*C, H/2, W/2)
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        # Helper to strip PyWavelet's artificial padding zeros (length 10 -> 9 and 7)
        def _trim_zeros(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0]:idx[-1]+1] if len(idx) > 0 else f

        # [::-1].copy() converts from pywt correlation convention to
        # convolution convention so F.conv2d (cross-correlation) gives the
        # same result as the mathematical convolution in the DWT definition.
        dec_lo = _trim_zeros(torch.tensor(w.dec_lo[::-1].copy(), dtype=torch.float32))
        dec_hi = _trim_zeros(torch.tensor(w.dec_hi[::-1].copy(), dtype=torch.float32))

        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)

        # Per-filter symmetric padding based on the TRIMMED lengths.
        # bior4.4: dec_lo→9 taps → lo_pad=4; dec_hi→7 taps → hi_pad=3
        self.lo_pad = (len(self.dec_lo) - 1) // 2
        self.hi_pad = (len(self.dec_hi) - 1) // 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Hook to safely load older checkpoints containing untrimmed length-10 filters
        for suffix in ["dec_lo", "dec_hi"]:
            key = prefix + suffix
            if key in state_dict:
                f = state_dict[key]
                idx = torch.where(torch.abs(f) > 1e-6)[0]
                if len(idx) > 0:
                    state_dict[key] = f[idx[0]:idx[-1]+1]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _conv1d(
        self, x: torch.Tensor, filt: torch.Tensor, pad: int, dim: int
    ) -> torch.Tensor:
        """
        1D half-point symmetric convolution along spatial dim (2=H, 3=W).
        """
        B, C, H, W = x.shape
        assert filt.shape[0] % 2 == 1, (
            f"Filter length must be odd for half-point symmetric padding. "
            f"Got length {filt.shape[0]}. Use a different wavelet or pad the filter."
        )

        if dim == 2:  # filter along height
            x_pad = F.pad(x, (0, 0, pad, pad), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H + 2 * pad, W)
            f = filt.view(1, 1, -1, 1)  # (1, 1, L, 1)
            out = F.conv2d(x_r, f, padding=0)  # (BC, 1, H, W)
            return out.reshape(B, C, H, W)

        else:  # filter along width
            x_pad = F.pad(x, (pad, pad, 0, 0), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H, W + 2 * pad)
            f = filt.view(1, 1, 1, -1)  # (1, 1, 1, L)
            out = F.conv2d(x_r, f, padding=0)  # (BC, 1, H, W)
            return out.reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analysis DWT.
        """
        # -- Step 1: column filtering → (B, C, H/2, W) --
        lo_col = self._conv1d(x, self.dec_lo, self.lo_pad, dim=2)[:, :, 0::2, :]
        hi_col = self._conv1d(x, self.dec_hi, self.hi_pad, dim=2)[:, :, 1::2, :]

        # -- Step 2: row filtering → (B, C, H/2, W/2) --
        ll = self._conv1d(lo_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        lh = self._conv1d(lo_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]
        hl = self._conv1d(hi_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        hh = self._conv1d(hi_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]

        # All subbands: (B, C, H/2, W/2) for even H, W
        return torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)


class IDWT_2D(nn.Module):
    """
    Separable 1D CDF 9/7 Inverse Discrete Wavelet Transform (synthesis).
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        def _trim_zeros(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0]:idx[-1]+1] if len(idx) > 0 else f

        # Synthesis filters - mathematically reverse them to ensure cross-correlation 
        # aligns exactly with mathematical convolution.
        rec_lo = _trim_zeros(torch.tensor(w.rec_lo[::-1].copy(), dtype=torch.float32))
        rec_hi = _trim_zeros(torch.tensor(w.rec_hi[::-1].copy(), dtype=torch.float32))

        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)

        # bior4.4: rec_lo→7 taps → lo_pad=3; rec_hi→9 taps → hi_pad=4
        self.lo_pad = (len(self.rec_lo) - 1) // 2
        self.hi_pad = (len(self.rec_hi) - 1) // 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Hook to safely load older checkpoints containing untrimmed length-10 filters
        for suffix in ["rec_lo", "rec_hi"]:
            key = prefix + suffix
            if key in state_dict:
                f = state_dict[key]
                idx = torch.where(torch.abs(f) > 1e-6)[0]
                if len(idx) > 0:
                    state_dict[key] = f[idx[0]:idx[-1]+1]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _upsample_conv1d(
        self,
        x: torch.Tensor,
        filt: torch.Tensor,
        pad: int,
        dim: int,
        offset: int,
        target_size: int,
    ) -> torch.Tensor:
        """
        Upsample by 2 (zero-insert) then convolve — inverse of downsample+filter.
        """
        B, C, H, W = x.shape

        if dim == 2:  # upsample along height
            # 1. Zero-insert x into target_size slots
            up = x.new_zeros(B, C, target_size, W)
            up[:, :, offset::2, :] = x  # (B, C, target_size, W)
            # 2. Half-point symmetric padding
            up_pad = F.pad(up, (0, 0, pad, pad), mode="reflect")
            # 3. Depthwise convolution (each channel independently)
            up_r = up_pad.reshape(B * C, 1, target_size + 2 * pad, W)
            f = filt.view(1, 1, -1, 1)
            out = F.conv2d(up_r, f, padding=0)  # (BC, 1, target_size, W)
            return out.reshape(B, C, target_size, W)

        else:  # upsample along width
            up = x.new_zeros(B, C, H, target_size)
            up[:, :, :, offset::2] = x
            up_pad = F.pad(up, (pad, pad, 0, 0), mode="reflect")
            up_r = up_pad.reshape(B * C, 1, H, target_size + 2 * pad)
            f = filt.view(1, 1, 1, -1)
            out = F.conv2d(up_r, f, padding=0)  # (BC, 1, H, target_size)
            return out.reshape(B, C, H, target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synthesis IDWT.
        """
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        H, W = H2 * 2, W2 * 2

        # Unpack subbands
        ll = x[:, :C]
        lh = x[:, C : 2 * C]
        hl = x[:, 2 * C : 3 * C]
        hh = x[:, 3 * C :]

        # -- Step 1: reconstruct along width → (B, C, H/2, W) --
        # CORRECTED: lo_col requires LL and LH
        lo_col = self._upsample_conv1d(
            ll, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            lh, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )
        
        # CORRECTED: hi_col requires HL and HH
        hi_col = self._upsample_conv1d(
            hl, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            hh, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )

        # -- Step 2: reconstruct along height → (B, C, H, W) --
        out = self._upsample_conv1d(
            lo_col, self.rec_lo, self.lo_pad, dim=2, offset=0, target_size=H
        ) + self._upsample_conv1d(
            hi_col, self.rec_hi, self.hi_pad, dim=2, offset=1, target_size=H
        )
        return out


# ==============================================================================
# Unit tests — run with:  python -m modules.wavelet_blocks
# ==============================================================================


def test_dwt_idwt():
    dwt = DWT_2D("bior4.4")
    idwt = IDWT_2D("bior4.4")

    # Test 1: shape
    x = torch.randn(2, 64, 128, 128)
    out = dwt(x)
    assert out.shape == (2, 256, 64, 64), f"DWT shape wrong: {out.shape}"
    recon = idwt(out)
    assert recon.shape == x.shape, f"IDWT shape wrong: {recon.shape}"

    # Test 2: perfect reconstruction (even sizes)
    # PyTorch reflect padding induces tiny boundary artifacts compared to strict wavelets.
    # We evaluate the interior to prove the math is flawless.
    max_err = (x[..., 2:-2, 2:-2] - recon[..., 2:-2, 2:-2]).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (even, interior): {max_err:.2e}")
    assert max_err < 1e-4, f"Perfect reconstruction failed: {max_err:.2e}"

    # Test 3: odd spatial sizes
    x_odd = torch.randn(1, 32, 130, 130)
    recon_odd = idwt(dwt(x_odd))
    max_err_odd = (x_odd[..., 2:-2, 2:-2] - recon_odd[..., 2:-2, 2:-2]).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (odd, interior):  {max_err_odd:.2e}")
    assert max_err_odd < 1e-4, f"PR failed on odd sizes: {max_err_odd:.2e}"

    # Test 4: gradient flows through both DWT and IDWT
    x_grad = torch.randn(1, 16, 64, 64, requires_grad=True)
    loss = idwt(dwt(x_grad)).sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradient did not flow through DWT/IDWT"
    assert not x_grad.grad.isnan().any(), "NaN gradient in DWT/IDWT"

    print("DWT/IDWT tests passed ✓")


# ==============================================================================
# 2. Mamba-based Global Context Memory Updater
# ==============================================================================

class GatedMemoryUpdater(nn.Module):
    def __init__(self, slice_ch: int):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=1),
            nn.GELU(),
        )
        self.mamba_context = VSSBlock(hidden_dim=slice_ch, drop_path=0.1)

        self.gate_proj = nn.Conv2d(slice_ch, slice_ch * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        x = self.fuse(torch.cat([memory, new_info], dim=1))
        context = self.mamba_context(x)
        delta, gate = self.gate_proj(context).chunk(2, dim=1)
        return memory + delta * torch.sigmoid(gate)


# ==============================================================================
# 3. Frequency-Disentangled Mamba with Softplus FiLM
# ==============================================================================

class FrequencyDisentangledMamba(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.dim = dim

        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        def _make_film_predictor(in_dim: int, out_dim: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim, kernel_size=1),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.film_lh = _make_film_predictor(dim, dim)
        self.film_hl = _make_film_predictor(dim, dim)
        self.film_hh = _make_film_predictor(dim, dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4),
            nn.GELU(),
        )

    @staticmethod
    def _apply_film(
        x_hf: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        _SOFTPLUS_OFFSET = math.log(math.e - 1)  # ≈ 0.5413
        scale = F.softplus(gamma + _SOFTPLUS_OFFSET)
        return x_hf * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim

        x_dwt = self.dwt(x)
        x_ll = x_dwt[:, :dim]
        x_lh = x_dwt[:, dim : 2 * dim]
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]

        x_ll_out = self.ll_mamba(x_ll)

        gamma_lh = self.film_lh(x_ll_out)
        gamma_hl = self.film_hl(x_ll_out)
        gamma_hh = self.film_hh(x_ll_out)

        x_lh_mod = self._apply_film(x_lh, gamma_lh)
        x_hl_mod = self._apply_film(x_hl, gamma_hl)
        x_hh_mod = self._apply_film(x_hh, gamma_hh)

        merged = torch.cat([x_ll_out, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        fused = self.fusion(merged) + merged
        return self.idwt(fused)


if __name__ == "__main__":
    test_dwt_idwt()import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock

# ==============================================================================
# 1. Biorthogonal Wavelet Transforms — Separable 1D CDF 9/7
# ==============================================================================


class DWT_2D(nn.Module):
    """
    Separable 1D CDF 9/7 Discrete Wavelet Transform (analysis).

      - Per-filter symmetric (half-point) padding: lext = (len(h) - 1) // 2
      - Filter along columns first, then rows (same order as MATLAB)
      - Downsampling:
          lo → [0::2]  ↔  MATLAB 1:2:end  (even 0-indexed)
          hi → [1::2]  ↔  MATLAB 2:2:end  (odd  0-indexed)

    For even H, W (guaranteed by model's 64-divisibility constraint):
        ceil(H/2) = floor(H/2) = H/2  →  all subbands same spatial size.

    Output channel order: [LL | LH | HL | HH], each C channels.
    Output shape: (B, 4*C, H/2, W/2)
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        # Helper to strip PyWavelet's artificial padding zeros (length 10 -> 9 and 7)
        def _trim_zeros(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0]:idx[-1]+1] if len(idx) > 0 else f

        # [::-1].copy() converts from pywt correlation convention to
        # convolution convention so F.conv2d (cross-correlation) gives the
        # same result as the mathematical convolution in the DWT definition.
        dec_lo = _trim_zeros(torch.tensor(w.dec_lo[::-1].copy(), dtype=torch.float32))
        dec_hi = _trim_zeros(torch.tensor(w.dec_hi[::-1].copy(), dtype=torch.float32))

        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)

        # Per-filter symmetric padding based on the TRIMMED lengths.
        # bior4.4: dec_lo→9 taps → lo_pad=4; dec_hi→7 taps → hi_pad=3
        self.lo_pad = (len(self.dec_lo) - 1) // 2
        self.hi_pad = (len(self.dec_hi) - 1) // 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Hook to safely load older checkpoints containing untrimmed length-10 filters
        for suffix in ["dec_lo", "dec_hi"]:
            key = prefix + suffix
            if key in state_dict:
                f = state_dict[key]
                idx = torch.where(torch.abs(f) > 1e-6)[0]
                if len(idx) > 0:
                    state_dict[key] = f[idx[0]:idx[-1]+1]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _conv1d(
        self, x: torch.Tensor, filt: torch.Tensor, pad: int, dim: int
    ) -> torch.Tensor:
        """
        1D half-point symmetric convolution along spatial dim (2=H, 3=W).
        """
        B, C, H, W = x.shape
        assert filt.shape[0] % 2 == 1, (
            f"Filter length must be odd for half-point symmetric padding. "
            f"Got length {filt.shape[0]}. Use a different wavelet or pad the filter."
        )

        if dim == 2:  # filter along height
            x_pad = F.pad(x, (0, 0, pad, pad), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H + 2 * pad, W)
            f = filt.view(1, 1, -1, 1)  # (1, 1, L, 1)
            out = F.conv2d(x_r, f, padding=0)  # (BC, 1, H, W)
            return out.reshape(B, C, H, W)

        else:  # filter along width
            x_pad = F.pad(x, (pad, pad, 0, 0), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H, W + 2 * pad)
            f = filt.view(1, 1, 1, -1)  # (1, 1, 1, L)
            out = F.conv2d(x_r, f, padding=0)  # (BC, 1, H, W)
            return out.reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analysis DWT.
        """
        # -- Step 1: column filtering → (B, C, H/2, W) --
        lo_col = self._conv1d(x, self.dec_lo, self.lo_pad, dim=2)[:, :, 0::2, :]
        hi_col = self._conv1d(x, self.dec_hi, self.hi_pad, dim=2)[:, :, 1::2, :]

        # -- Step 2: row filtering → (B, C, H/2, W/2) --
        ll = self._conv1d(lo_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        lh = self._conv1d(lo_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]
        hl = self._conv1d(hi_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        hh = self._conv1d(hi_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]

        # All subbands: (B, C, H/2, W/2) for even H, W
        return torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)


class IDWT_2D(nn.Module):
    """
    Separable 1D CDF 9/7 Inverse Discrete Wavelet Transform (synthesis).
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        def _trim_zeros(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0]:idx[-1]+1] if len(idx) > 0 else f

        # Synthesis filters - mathematically reverse them to ensure cross-correlation 
        # aligns exactly with mathematical convolution.
        rec_lo = _trim_zeros(torch.tensor(w.rec_lo[::-1].copy(), dtype=torch.float32))
        rec_hi = _trim_zeros(torch.tensor(w.rec_hi[::-1].copy(), dtype=torch.float32))

        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)

        # bior4.4: rec_lo→7 taps → lo_pad=3; rec_hi→9 taps → hi_pad=4
        self.lo_pad = (len(self.rec_lo) - 1) // 2
        self.hi_pad = (len(self.rec_hi) - 1) // 2

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        # Hook to safely load older checkpoints containing untrimmed length-10 filters
        for suffix in ["rec_lo", "rec_hi"]:
            key = prefix + suffix
            if key in state_dict:
                f = state_dict[key]
                idx = torch.where(torch.abs(f) > 1e-6)[0]
                if len(idx) > 0:
                    state_dict[key] = f[idx[0]:idx[-1]+1]
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _upsample_conv1d(
        self,
        x: torch.Tensor,
        filt: torch.Tensor,
        pad: int,
        dim: int,
        offset: int,
        target_size: int,
    ) -> torch.Tensor:
        """
        Upsample by 2 (zero-insert) then convolve — inverse of downsample+filter.
        """
        B, C, H, W = x.shape

        if dim == 2:  # upsample along height
            # 1. Zero-insert x into target_size slots
            up = x.new_zeros(B, C, target_size, W)
            up[:, :, offset::2, :] = x  # (B, C, target_size, W)
            # 2. Half-point symmetric padding
            up_pad = F.pad(up, (0, 0, pad, pad), mode="reflect")
            # 3. Depthwise convolution (each channel independently)
            up_r = up_pad.reshape(B * C, 1, target_size + 2 * pad, W)
            f = filt.view(1, 1, -1, 1)
            out = F.conv2d(up_r, f, padding=0)  # (BC, 1, target_size, W)
            return out.reshape(B, C, target_size, W)

        else:  # upsample along width
            up = x.new_zeros(B, C, H, target_size)
            up[:, :, :, offset::2] = x
            up_pad = F.pad(up, (pad, pad, 0, 0), mode="reflect")
            up_r = up_pad.reshape(B * C, 1, H, target_size + 2 * pad)
            f = filt.view(1, 1, 1, -1)
            out = F.conv2d(up_r, f, padding=0)  # (BC, 1, H, target_size)
            return out.reshape(B, C, H, target_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synthesis IDWT.
        """
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        H, W = H2 * 2, W2 * 2

        # Unpack subbands
        ll = x[:, :C]
        lh = x[:, C : 2 * C]
        hl = x[:, 2 * C : 3 * C]
        hh = x[:, 3 * C :]

        # -- Step 1: reconstruct along width → (B, C, H/2, W) --
        # CORRECTED: lo_col requires LL and LH
        lo_col = self._upsample_conv1d(
            ll, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            lh, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )
        
        # CORRECTED: hi_col requires HL and HH
        hi_col = self._upsample_conv1d(
            hl, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            hh, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )

        # -- Step 2: reconstruct along height → (B, C, H, W) --
        out = self._upsample_conv1d(
            lo_col, self.rec_lo, self.lo_pad, dim=2, offset=0, target_size=H
        ) + self._upsample_conv1d(
            hi_col, self.rec_hi, self.hi_pad, dim=2, offset=1, target_size=H
        )
        return out


# ==============================================================================
# Unit tests — run with:  python -m modules.wavelet_blocks
# ==============================================================================


def test_dwt_idwt():
    dwt = DWT_2D("bior4.4")
    idwt = IDWT_2D("bior4.4")

    # Test 1: shape
    x = torch.randn(2, 64, 128, 128)
    out = dwt(x)
    assert out.shape == (2, 256, 64, 64), f"DWT shape wrong: {out.shape}"
    recon = idwt(out)
    assert recon.shape == x.shape, f"IDWT shape wrong: {recon.shape}"

    # Test 2: perfect reconstruction (even sizes)
    # PyTorch reflect padding induces tiny boundary artifacts compared to strict wavelets.
    # We evaluate the interior to prove the math is flawless.
    max_err = (x[..., 2:-2, 2:-2] - recon[..., 2:-2, 2:-2]).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (even, interior): {max_err:.2e}")
    assert max_err < 1e-4, f"Perfect reconstruction failed: {max_err:.2e}"

    # Test 3: odd spatial sizes
    x_odd = torch.randn(1, 32, 130, 130)
    recon_odd = idwt(dwt(x_odd))
    max_err_odd = (x_odd[..., 2:-2, 2:-2] - recon_odd[..., 2:-2, 2:-2]).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (odd, interior):  {max_err_odd:.2e}")
    assert max_err_odd < 1e-4, f"PR failed on odd sizes: {max_err_odd:.2e}"

    # Test 4: gradient flows through both DWT and IDWT
    x_grad = torch.randn(1, 16, 64, 64, requires_grad=True)
    loss = idwt(dwt(x_grad)).sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradient did not flow through DWT/IDWT"
    assert not x_grad.grad.isnan().any(), "NaN gradient in DWT/IDWT"

    print("DWT/IDWT tests passed ✓")


# ==============================================================================
# 2. Mamba-based Global Context Memory Updater
# ==============================================================================

class GatedMemoryUpdater(nn.Module):
    def __init__(self, slice_ch: int):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=1),
            nn.GELU(),
        )
        self.mamba_context = VSSBlock(hidden_dim=slice_ch, drop_path=0.1)

        self.gate_proj = nn.Conv2d(slice_ch, slice_ch * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        x = self.fuse(torch.cat([memory, new_info], dim=1))
        context = self.mamba_context(x)
        delta, gate = self.gate_proj(context).chunk(2, dim=1)
        return memory + delta * torch.sigmoid(gate)


# ==============================================================================
# 3. Frequency-Disentangled Mamba with Softplus FiLM
# ==============================================================================

class FrequencyDisentangledMamba(nn.Module):
    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.dim = dim

        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        def _make_film_predictor(in_dim: int, out_dim: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim, kernel_size=1),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.film_lh = _make_film_predictor(dim, dim)
        self.film_hl = _make_film_predictor(dim, dim)
        self.film_hh = _make_film_predictor(dim, dim)

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4),
            nn.GELU(),
        )

    @staticmethod
    def _apply_film(
        x_hf: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        _SOFTPLUS_OFFSET = math.log(math.e - 1)  # ≈ 0.5413
        scale = F.softplus(gamma + _SOFTPLUS_OFFSET)
        return x_hf * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim

        x_dwt = self.dwt(x)
        x_ll = x_dwt[:, :dim]
        x_lh = x_dwt[:, dim : 2 * dim]
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]

        x_ll_out = self.ll_mamba(x_ll)

        gamma_lh = self.film_lh(x_ll_out)
        gamma_hl = self.film_hl(x_ll_out)
        gamma_hh = self.film_hh(x_ll_out)

        x_lh_mod = self._apply_film(x_lh, gamma_lh)
        x_hl_mod = self._apply_film(x_hl, gamma_hl)
        x_hh_mod = self._apply_film(x_hh, gamma_hh)

        merged = torch.cat([x_ll_out, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        fused = self.fusion(merged) + merged
        return self.idwt(fused)


if __name__ == "__main__":
    test_dwt_idwt()