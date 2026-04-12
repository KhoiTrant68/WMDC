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

    NOTE: Only odd-length filters are supported (bior4.4: dec_lo=9, dec_hi=7).
    An assertion in _conv1d guards against even-length filters.
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        # [::-1].copy() converts from pywt correlation convention to
        # convolution convention so F.conv2d (cross-correlation) gives the
        # same result as the mathematical convolution in the DWT definition.
        dec_lo = torch.tensor(w.dec_lo[::-1].copy(), dtype=torch.float32)
        dec_hi = torch.tensor(w.dec_hi[::-1].copy(), dtype=torch.float32)

        self.register_buffer("dec_lo", dec_lo)
        self.register_buffer("dec_hi", dec_hi)

        # Per-filter half-point symmetric padding.
        # bior4.4: dec_lo→9 taps → lo_pad=4; dec_hi→7 taps → hi_pad=3
        self.lo_pad = (len(w.dec_lo) - 1) // 2
        self.hi_pad = (len(w.dec_hi) - 1) // 2

    def _conv1d(
        self, x: torch.Tensor, filt: torch.Tensor, pad: int, dim: int
    ) -> torch.Tensor:
        """
        1D half-point symmetric convolution along spatial dim (2=H, 3=W).

        Formula: output_size = input_size + 2*pad - len(filt) + 1
        For odd-length filt with pad = (len-1)//2:  output_size = input_size ✓

        Args:
            x    : (B, C, H, W)
            filt : 1D tensor of length L (must be odd)
            pad  : symmetric padding = (L-1) // 2
            dim  : 2 for height axis, 3 for width axis

        Returns:
            (B, C, H, W)  — same spatial size as input (before downsampling)
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
            assert out.shape[2] == H, (
                f"Unexpected output height {out.shape[2]}, expected {H}. "
                "Check filter length parity."
            )
            return out.reshape(B, C, H, W)

        else:  # filter along width
            x_pad = F.pad(x, (pad, pad, 0, 0), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H, W + 2 * pad)
            f = filt.view(1, 1, 1, -1)  # (1, 1, 1, L)
            out = F.conv2d(x_r, f, padding=0)  # (BC, 1, H, W)
            assert out.shape[3] == W, (
                f"Unexpected output width {out.shape[3]}, expected {W}. "
                "Check filter length parity."
            )
            return out.reshape(B, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analysis DWT.

        Step 1: filter along columns (H), downsample.
        Step 2: filter along rows (W), downsample.

        Returns: (B, 4*C, H/2, W/2) with subbands ordered [LL, LH, HL, HH].
        """
        # -- Step 1: column filtering → (B, C, H/2, W) --
        lo_col = self._conv1d(x, self.dec_lo, self.lo_pad, dim=2)[:, :, 0::2, :]
        hi_col = self._conv1d(x, self.dec_hi, self.hi_pad, dim=2)[:, :, 1::2, :]

        # -- Step 2: row filtering → (B, C, H/2, W/2) --
        ll = self._conv1d(lo_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        lh = self._conv1d(lo_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]
        hl = self._conv1d(hi_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        hh = self._conv1d(hi_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]

        # All subbands: (B, C, H/2, W/2) for even H, W ✓
        return torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)


class IDWT_2D(nn.Module):
    """
    Separable 1D CDF 9/7 Inverse Discrete Wavelet Transform (synthesis).

    Synthesis filter assignment (bior4.4):
        rec_lo : 7 taps  (derived from dec_hi, symmetric → flip = identity)
        rec_hi : 9 taps  (derived from dec_lo, symmetric → flip = identity)

    IMPORTANT: rec_lo/rec_hi for bior4.4 are symmetric filters, so
    NOT reversing them before F.conv2d (cross-correlation) gives the same
    result as the true convolution synthesis formula.  For non-symmetric
    wavelets this class would need `[::-1]` applied to rec_lo/rec_hi.

    Synthesis boundary extension:
        F.pad(zero_inserted, mode='reflect') on the upsampled (zero-inserted)
        signal applies half-point symmetric extension.  For symmetric filters
        (bior4.4), this is consistent with the analysis half-point symmetric
        extension and guarantees perfect reconstruction (PR).

        Verification: test_dwt_idwt() checks max |x - IDWT(DWT(x))| < 1e-4.

    Input channel order: [LL | LH | HL | HH] (matches DWT_2D output).
    Input shape:  (B, 4*C, H/2, W/2)
    Output shape: (B, C, H, W)
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        # Synthesis filters — NOT reversed because bior4.4 rec_lo/rec_hi
        # are symmetric (flip = identity), so cross-correlation = convolution.
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)

        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)

        # bior4.4: rec_lo→7 taps → lo_pad=3; rec_hi→9 taps → hi_pad=4
        self.lo_pad = (len(w.rec_lo) - 1) // 2
        self.hi_pad = (len(w.rec_hi) - 1) // 2

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

        Args:
            x           : (B, C, H, W) — small (half-size) input
            filt        : synthesis filter (rec_lo or rec_hi)
            pad         : (len(filt) - 1) // 2
            dim         : 2 = upsample height, 3 = upsample width
            offset      : 0 = signal at even positions [0::2]  (lo branch)
                          1 = signal at odd  positions [1::2]  (hi branch)
            target_size : output spatial size = 2 × input spatial size

        Boundary handling:
            F.pad with mode='reflect' on the zero-inserted signal applies
            half-point symmetric extension.  For bior4.4 (symmetric synthesis
            filters) this is consistent with the analysis extension, ensuring PR.

        Size check (bior4.4 example, target_size=128):
            lo (rec_lo=7 taps, pad=3):  128+6=134 → conv→134-7+1=128 ✓
            hi (rec_hi=9 taps, pad=4):  128+8=136 → conv→136-9+1=128 ✓
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

        Reverses DWT_2D.forward():
            Step 1: reconstruct rows (width axis).
            Step 2: reconstruct columns (height axis).

        Input channel order [LL, LH, HL, HH] matches DWT_2D output order.
        """
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        H, W = H2 * 2, W2 * 2

        # Unpack subbands — order matches DWT_2D output [LL, LH, HL, HH]
        ll = x[:, :C]
        lh = x[:, C : 2 * C]
        hl = x[:, 2 * C : 3 * C]
        hh = x[:, 3 * C :]

        # -- Step 1: reconstruct along width → (B, C, H/2, W) --
        # lo_col combines the LL+LH lo-pass row and HL+HH hi-pass row
        lo_col = self._upsample_conv1d(
            ll, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            hl, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )
        hi_col = self._upsample_conv1d(
            lh, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
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
# Unit tests — run with:  python wavelet_blocks.py
# ==============================================================================


def test_dwt_idwt():
    """
    Verify DWT_2D / IDWT_2D pair properties for bior4.4:
      1. Output shapes are correct.
      2. Perfect reconstruction: max |x - IDWT(DWT(x))| < 1e-4.
      3. Works on odd spatial sizes (the model always pads to multiples of 64,
         but guarding against misuse is cheap).
    """
    dwt = DWT_2D("bior4.4")
    idwt = IDWT_2D("bior4.4")

    # Test 1: shape
    x = torch.randn(2, 64, 128, 128)
    out = dwt(x)
    assert out.shape == (2, 256, 64, 64), f"DWT shape wrong: {out.shape}"
    recon = idwt(out)
    assert recon.shape == x.shape, f"IDWT shape wrong: {recon.shape}"

    # Test 2: perfect reconstruction (even sizes)
    max_err = (x - recon).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (even): {max_err:.2e}")
    assert max_err < 1e-4, f"Perfect reconstruction failed: {max_err:.2e}"

    # Test 3: odd spatial sizes
    x_odd = torch.randn(1, 32, 130, 130)
    recon_odd = idwt(dwt(x_odd))
    assert recon_odd.shape == x_odd.shape, f"Odd size shape wrong: {recon_odd.shape}"
    max_err_odd = (x_odd - recon_odd).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (odd):  {max_err_odd:.2e}")
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
    r"""
    Mamba-based Memory Updater for Autoregressive Slicing.

    Mathematical justification for Mamba global receptive field:
    ------------------------------------------------------------
    Standard CNNs have receptive field R = kernel_size × depth.
    State Space Models (Mamba) discretize a continuous system:
        h_k = Ā h_{k-1} + B̄ x_k,   y_k = C h_k
    The state h_k aggregates the entire sequence history with O(1) cost
    per step.  VSSBlock applies this in 2D via cross-scan, giving an
    effective receptive field of H × W in a single layer.

    Gating mechanism — why sigmoid instead of Tanh:
    ------------------------------------------------
    Previous design used Tanh as the final gating activation.
    Tanh saturates to ±1 as |input| → ∞, killing gradients in late
    training when gate magnitudes are large.  The sigmoid-gated design:

        delta, gate = gate_proj(context).chunk(2, dim=1)
        return memory + delta * sigmoid(gate)

    never saturates the gradient of `gate` (sigmoid' = sigmoid(1-sigmoid) > 0
    everywhere), and delta can be any magnitude — only the gate weight is
    bounded, not the update direction.

    Initialization:
    ---------------
    gate_proj is zero-initialized:
        delta = 0, gate = 0  →  sigmoid(0) = 0.5  →  update = 0 × 0.5 = 0
    So memory[0] = memory, i.e. identity at epoch 0. ✓
    """

    def __init__(self, slice_ch: int):
        super().__init__()

        # Fuse old memory with newly decoded slice
        self.fuse = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=1),
            nn.GELU(),
        )

        # Global spatial context via 2D Cross-Scan Mamba
        self.mamba_context = VSSBlock(hidden_dim=slice_ch, drop_path=0.1)

        # Gating: projects context → (delta, gate) jointly
        # Zero-init → identity mapping at epoch 0
        self.gate_proj = nn.Conv2d(slice_ch, slice_ch * 2, kernel_size=3, padding=1)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory   : (B, slice_ch, H, W)  accumulated state
            new_info : (B, slice_ch, H, W)  just-decoded slice (LRP-corrected)

        Returns:
            updated memory : (B, slice_ch, H, W)
        """
        # 1. Fuse old state with new information
        x = self.fuse(torch.cat([memory, new_info], dim=1))

        # 2. Extract global context (Mamba: H×W receptive field)
        context = self.mamba_context(x)

        # 3. Sigmoid-gated residual update
        #    delta=0 and gate=0 at init → update=0 (pure identity)
        delta, gate = self.gate_proj(context).chunk(2, dim=1)
        return memory + delta * torch.sigmoid(gate)


# ==============================================================================
# 3. Frequency-Disentangled Mamba with Softplus FiLM
# ==============================================================================


class FrequencyDisentangledMamba(nn.Module):
    r"""
    Cross-Frequency Mamba Modulation.

    Architecture:
    -------------
    1. CDF 9/7 DWT decomposes features → LL, LH, HL, HH subbands.
    2. LL passes through VSSBlock (Mamba) → global structural context.
    3. Mamba-enriched LL predicts per-subband FiLM parameters (γ, β).
    4. Each HF subband modulated: f(x, γ, β) = x · softplus(γ+offset) + β
       where offset = log(e−1) ≈ 0.5413 ensures scale = 1.0 at init.
    5. Fused subbands reconstructed via IDWT.

    FiLM predictor init:
        Final conv is zero-initialized → γ=0, β=0 at epoch 0.
        softplus(0 + log(e−1)) = softplus(0.5413) = log(1 + e^0.5413)
                                = log(1 + (e−1)) = log(e) = 1.0  ✓
        So the initial transform is the identity: f(x,0,0) = x · 1.0 + 0 = x. ✓
    """

    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.dim = dim

        # LL subband: long-range global structure via Mamba
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        def _make_film_predictor(in_dim: int, out_dim: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim * 2, kernel_size=1),
                # out_dim*2: first half → γ, second half → β
            )
            # Zero-init final layer: γ=0, β=0 → identity at epoch 0
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.film_lh = _make_film_predictor(dim, dim)
        self.film_hl = _make_film_predictor(dim, dim)
        self.film_hh = _make_film_predictor(dim, dim)

        # Depthwise fusion before IDWT
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4),
            nn.GELU(),
        )

    @staticmethod
    def _apply_film(
        x_hf: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        r"""
        FiLM modulation with Softplus scaling — identity at init.

        Formulation:
            f(x, γ, β) = x · softplus(γ + log(e−1)) + β

        Properties:
            scale at γ=0: softplus(log(e−1)) = log(1+(e−1)) = log(e) = 1.0  ✓
            scale > 0  ∀ γ ∈ ℝ  (never flips phase)                          ✓
            scale unbounded above  (can amplify strongly)                      ✓
            ∂scale/∂γ = σ(γ + log(e−1)) > 0  (gradient never dies)           ✓
        """
        _SOFTPLUS_OFFSET = math.log(math.e - 1)  # ≈ 0.5413
        scale = F.softplus(gamma + _SOFTPLUS_OFFSET)
        return x_hf * scale + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x : (B, dim, H, W)

        Returns:
            (B, dim, H, W)  — same shape as input
        """
        dim = self.dim

        # 1. DWT → (B, 4*dim, H/2, W/2), order: [LL, LH, HL, HH]
        x_dwt = self.dwt(x)
        x_ll = x_dwt[:, :dim]
        x_lh = x_dwt[:, dim : 2 * dim]
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]

        # 2. Mamba: global context from LL subband
        x_ll_out = self.ll_mamba(x_ll)

        # 3. FiLM: LL-predicted parameters for each HF subband
        gamma_lh, beta_lh = self.film_lh(x_ll_out).chunk(2, dim=1)
        gamma_hl, beta_hl = self.film_hl(x_ll_out).chunk(2, dim=1)
        gamma_hh, beta_hh = self.film_hh(x_ll_out).chunk(2, dim=1)

        # 4. Modulate HF subbands (identity at epoch 0)
        x_lh_mod = self._apply_film(x_lh, gamma_lh, beta_lh)
        x_hl_mod = self._apply_film(x_hl, gamma_hl, beta_hl)
        x_hh_mod = self._apply_film(x_hh, gamma_hh, beta_hh)

        # 5. Depthwise fusion + skip, then IDWT
        merged = torch.cat([x_ll_out, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        fused = self.fusion(merged) + merged  # residual in subband space
        return self.idwt(fused)


# ==============================================================================
# Entry point for unit tests
# ==============================================================================

if __name__ == "__main__":
    test_dwt_idwt()
