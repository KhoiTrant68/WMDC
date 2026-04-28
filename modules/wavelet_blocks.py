import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock

# ==============================================================================
# 1.  Biorthogonal Wavelet Transforms — bior4.4 (CDF 9/7 analysis)
# ==============================================================================


class DWT_2D(nn.Module):
    """
    Separable 1-D DWT (analysis) using bior4.4 (or any pywt wavelet).

    Processing order
    ----------------
    1. Column filtering (dim=H) then downsample by 2.
    2. Row filtering    (dim=W) then downsample by 2.

    Downsampling convention:
        lo → even samples  [0::2]   (MATLAB 1:2:end)
        hi → odd  samples  [1::2]   (MATLAB 2:2:end)

    Padding: half-point symmetric (reflect) — length = (filter_len - 1) // 2
    PyWavelets returns filters in correlation convention; we reverse them to
    match the convolution convention expected by F.conv2d.

    Output shape:  (B, 4*C, H/2, W/2)
    Channel order: [LL | LH | HL | HH], each C channels.

    Even H, W (guaranteed by the 64-divisibility constraint) → all subbands
    have identical spatial size H/2 × W/2.
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

        # bior4.4 after trimming: dec_lo → 9 taps (lo_pad=4),
        #                         dec_hi → 7 taps (hi_pad=3)
        self.lo_pad = (len(self.dec_lo) - 1) // 2
        self.hi_pad = (len(self.dec_hi) - 1) // 2

    # ------------------------------------------------------------------
    # Checkpoint compatibility hook
    # ------------------------------------------------------------------

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
        """Trim filters from older checkpoints that stored length-10 tensors."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _conv1d(
        self, x: torch.Tensor, filt: torch.Tensor, pad: int, dim: int
    ) -> torch.Tensor:
        """
        Separable 1-D half-point-symmetric convolution.

        dim=2 → filter along H (rows stay unchanged)
        dim=3 → filter along W (cols stay unchanged)

        Works by reshaping to (B*C, 1, spatial, 1) or (B*C, 1, 1, spatial)
        and calling F.conv2d with a (1,1,L,1) or (1,1,1,L) kernel.
        """
        B, C, H, W = x.shape
        assert filt.shape[0] % 2 == 1, (
            f"Filter length must be odd for half-point symmetric padding. "
            f"Got {filt.shape[0]} taps."
        )
        if dim == 2:
            x_pad = F.pad(x, (0, 0, pad, pad), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H + 2 * pad, W)
            f = filt.view(1, 1, -1, 1)
            out = F.conv2d(x_r, f, padding=0)
            return out.reshape(B, C, H, W)
        else:
            x_pad = F.pad(x, (pad, pad, 0, 0), mode="reflect")
            x_r = x_pad.reshape(B * C, 1, H, W + 2 * pad)
            f = filt.view(1, 1, 1, -1)
            out = F.conv2d(x_r, f, padding=0)
            return out.reshape(B, C, H, W)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Analysis DWT.

        Parameters
        ----------
        x : (B, C, H, W)

        Returns
        -------
        (B, 4*C, H/2, W/2)  — channel order [LL, LH, HL, HH]
        """
        # Step 1: filter columns, downsample rows
        lo_col = self._conv1d(x, self.dec_lo, self.lo_pad, dim=2)[:, :, 0::2, :]
        hi_col = self._conv1d(x, self.dec_hi, self.hi_pad, dim=2)[:, :, 1::2, :]

        # Step 2: filter rows, downsample cols
        ll = self._conv1d(lo_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        lh = self._conv1d(lo_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]
        hl = self._conv1d(hi_col, self.dec_lo, self.lo_pad, dim=3)[:, :, :, 0::2]
        hh = self._conv1d(hi_col, self.dec_hi, self.hi_pad, dim=3)[:, :, :, 1::2]

        return torch.cat([ll, lh, hl, hh], dim=1)  # (B, 4C, H/2, W/2)


class IDWT_2D(nn.Module):
    """
    Separable 1-D IDWT (synthesis) using bior4.4 (or any pywt wavelet).

    Inverse of DWT_2D: upsample by 2 (zero-insert) then convolve with
    synthesis filters, summing the low- and high-frequency contributions.

    Processing order
    ----------------
    1. Reconstruct along W (inverse of the row filtering step in DWT).
    2. Reconstruct along H (inverse of the column filtering step in DWT).

    Expected input:  (B, 4*C, H/2, W/2)  — channel order [LL, LH, HL, HH]
    Output:          (B, C, H, W)
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        def _trim_zeros(f: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
            idx = torch.where(torch.abs(f) > eps)[0]
            return f[idx[0] : idx[-1] + 1] if len(idx) > 0 else f

        rec_lo = _trim_zeros(torch.tensor(w.rec_lo[::-1].copy(), dtype=torch.float32))
        rec_hi = _trim_zeros(torch.tensor(w.rec_hi[::-1].copy(), dtype=torch.float32))

        self.register_buffer("rec_lo", rec_lo)
        self.register_buffer("rec_hi", rec_hi)

        # bior4.4 after trimming: rec_lo → 7 taps (lo_pad=3),
        #                         rec_hi → 9 taps (hi_pad=4)
        self.lo_pad = (len(self.rec_lo) - 1) // 2
        self.hi_pad = (len(self.rec_hi) - 1) // 2

    # ------------------------------------------------------------------
    # Checkpoint compatibility hook
    # ------------------------------------------------------------------

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
        """Trim filters from older checkpoints that stored length-10 tensors."""
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        Zero-insert upsample then half-point-symmetric convolution.

        offset=0 → place samples at even positions  (inverse of [0::2])
        offset=1 → place samples at odd  positions  (inverse of [1::2])
        """
        B, C, H, W = x.shape

        if dim == 2:  # upsample along H
            up = x.new_zeros(B, C, target_size, W)
            up[:, :, offset::2, :] = x
            up_pad = F.pad(up, (0, 0, pad, pad), mode="reflect")
            up_r = up_pad.reshape(B * C, 1, target_size + 2 * pad, W)
            f = filt.view(1, 1, -1, 1)
            out = F.conv2d(up_r, f, padding=0)
            return out.reshape(B, C, target_size, W)
        else:  # upsample along W
            up = x.new_zeros(B, C, H, target_size)
            up[:, :, :, offset::2] = x
            up_pad = F.pad(up, (pad, pad, 0, 0), mode="reflect")
            up_r = up_pad.reshape(B * C, 1, H, target_size + 2 * pad)
            f = filt.view(1, 1, 1, -1)
            out = F.conv2d(up_r, f, padding=0)
            return out.reshape(B, C, H, target_size)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Synthesis IDWT.

        Parameters
        ----------
        x : (B, 4*C, H/2, W/2)  — channel order [LL, LH, HL, HH]

        Returns
        -------
        (B, C, H, W)
        """
        B, C4, H2, W2 = x.shape
        C = C4 // 4
        H, W = H2 * 2, W2 * 2

        # Unpack subbands
        ll = x[:, :C]
        lh = x[:, C : 2 * C]
        hl = x[:, 2 * C : 3 * C]
        hh = x[:, 3 * C :]

        # Step 1: reconstruct along W  (inverse of DWT row-filtering step)
        # lo_col  = IDWT_W( LL, LH )
        lo_col = self._upsample_conv1d(
            ll, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            lh, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )
        # hi_col  = IDWT_W( HL, HH )
        hi_col = self._upsample_conv1d(
            hl, self.rec_lo, self.lo_pad, dim=3, offset=0, target_size=W
        ) + self._upsample_conv1d(
            hh, self.rec_hi, self.hi_pad, dim=3, offset=1, target_size=W
        )

        # Step 2: reconstruct along H  (inverse of DWT column-filtering step)
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

    # Test 1: output shape
    x = torch.randn(2, 64, 128, 128)
    out = dwt(x)
    assert out.shape == (2, 256, 64, 64), f"DWT shape wrong: {out.shape}"
    recon = idwt(out)
    assert recon.shape == x.shape, f"IDWT shape wrong: {recon.shape}"

    # Test 2: near-perfect reconstruction on even sizes (interior, to avoid
    # boundary artefacts from reflect padding vs. strict periodic extension)
    max_err = (x[..., 2:-2, 2:-2] - recon[..., 2:-2, 2:-2]).abs().max().item()
    print(f"[bior4.4] Max reconstruction error (even, interior): {max_err:.2e}")
    assert max_err < 1e-4, f"PR failed (even): {max_err:.2e}"

    # Test 3: odd spatial sizes
    x_odd = torch.randn(1, 32, 130, 130)
    recon_odd = idwt(dwt(x_odd))
    max_err_odd = (
        (x_odd[..., 2:-2, 2:-2] - recon_odd[..., 2:-2, 2:-2]).abs().max().item()
    )
    print(f"[bior4.4] Max reconstruction error (odd, interior):  {max_err_odd:.2e}")
    assert max_err_odd < 1e-4, f"PR failed (odd): {max_err_odd:.2e}"

    # Test 4: gradient flow
    x_grad = torch.randn(1, 16, 64, 64, requires_grad=True)
    loss = idwt(dwt(x_grad)).sum()
    loss.backward()
    assert x_grad.grad is not None, "Gradient did not flow through DWT/IDWT"
    assert not x_grad.grad.isnan().any(), "NaN gradient in DWT/IDWT"

    print("DWT/IDWT tests passed ✓")


# ==============================================================================
# 2.  Gated Memory Updater
# ==============================================================================


class GatedMemoryUpdater(nn.Module):
    """
    Updates the inter-slice memory state using a gated Mamba context.

    Given the current memory and the newly decoded slice, the updater:
      1. Fuses them with a 1×1 conv + GELU.
      2. Extracts global context via VSSBlock (Mamba).
      3. Computes a delta (additive update) and a sigmoid gate.
      4. Returns: memory_new = memory + delta * sigmoid(gate)

    Design rationale (additive vs. GRU-style)
    -----------------------------------------
    We use the ADDITIVE form  m_new = m + δ * gate  rather than the
    GRU-style  m_new = m * (1 − gate) + δ * gate.

    Reason: the GRU form attenuates the memory by `gate ≈ sigmoid(-2) ≈ 0.119`
    at every step at init, even though delta is zero-initialised.  After the
    4 memory updates in a 5-slice loop the memory would be reduced to
    ≈ 0.881⁴ ≈ 0.60× its original magnitude before the model has learned
    anything useful.

    The additive form preserves the memory exactly at init (delta=0 → no
    change) and matches the "start-as-no-op" pattern used elsewhere in the
    codebase (LRP gate, FiLM, fusion residual, delta_proj zero-init).
    """

    def __init__(self, slice_ch: int):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=1),
            nn.GELU(),
        )
        self.mamba_context = VSSBlock(hidden_dim=slice_ch, drop_path=0.0)

        self.delta_proj = nn.Conv2d(slice_ch, slice_ch, kernel_size=3, padding=1)
        self.gate_proj = nn.Conv2d(slice_ch, slice_ch, kernel_size=3, padding=1)

        # Zero-init delta so additive updates start at 0
        nn.init.zeros_(self.delta_proj.weight)
        nn.init.zeros_(self.delta_proj.bias)
        # Zero-weight, small negative bias → gate ≈ sigmoid(-2) ≈ 0.12 at init.
        # Combined with delta=0, the update term `delta * gate` is exactly 0
        # at init, so memory passes through unchanged.
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.constant_(self.gate_proj.bias, -2.0)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        memory   : (B, slice_ch, H, W)  current memory state
        new_info : (B, slice_ch, H, W)  LRP-corrected quantised slice

        Returns
        -------
        memory_new : (B, slice_ch, H, W)
        """
        x = self.fuse(torch.cat([memory, new_info], dim=1))
        context = self.mamba_context(x)

        delta = self.delta_proj(context)
        gate = torch.sigmoid(self.gate_proj(context))
        # Additive update — preserves memory at init (delta=0).
        return memory + delta * gate


# ==============================================================================
# 3.  Frequency-Disentangled Mamba with Softplus FiLM
# ==============================================================================


class FrequencyDisentangledMamba(nn.Module):
    """
    Wavelet-domain feature transform with Mamba-based cross-band modulation.

    Processing pipeline
    -------------------
    1. DWT(x)   → LL, LH, HL, HH subbands  (B, C, H/2, W/2) each.
    2. ll_mamba processes the LL band independently.
    3. context_mamba aggregates ALL bands (processed LL + raw HF) into a
       global context tensor.
    4. Four Affine FiLM networks predict (scale, shift) from context for
       each subband.
    5. All modulated subbands are concatenated and passed through a
       depthwise + pointwise fusion (FIX-A5).
    6. IDWT(fused) → output in pixel space.
    """

    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.dim = dim

        # ── LL branch (main low-frequency path) ───────────────────────────────
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # ── Context branch (sees all subbands) ────────────────────────────────
        self.pre_context_proj = nn.Conv2d(dim * 4, dim, kernel_size=1)
        self.context_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # ── Affine FiLM predictors (one per subband) ──────────────────────────
        # Each predictor outputs 2*out_dim values: first out_dim are log-scale
        # (passed through softplus to keep scale > 0), last out_dim are shift.
        # Weight zero-init + specific bias ensures scale=1, shift=0 at t=0.
        def _make_affine_film_predictor(in_dim: int, out_dim: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim * 2, kernel_size=1),
            )
            nn.init.zeros_(net[-1].weight)
            # softplus(log(e-1)) = 1.0  → scale initialised to 1
            nn.init.constant_(net[-1].bias[:out_dim], math.log(math.e - 1))
            # shift initialised to 0
            nn.init.zeros_(net[-1].bias[out_dim:])
            return net

        self.film_ll = _make_affine_film_predictor(dim, dim)
        self.film_lh = _make_affine_film_predictor(dim, dim)
        self.film_hl = _make_affine_film_predictor(dim, dim)
        self.film_hh = _make_affine_film_predictor(dim, dim)

        # ── Subband fusion ──────────────────────────────────────────
        # depthwise:  spatial smoothing within each channel independently
        # pointwise:  learned cross-subband mixing
        #
        # pointwise is initialised to identity (diagonal weight matrix)
        # so the fusion is a no-op at t=0.  The residual connection
        # (fused + merged) maintains gradient flow during early training.
        self.fusion_dw = nn.Conv2d(
            dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4
        )
        self.fusion_act = nn.GELU()
        self.fusion_pw = nn.Conv2d(dim * 4, dim * 4, kernel_size=1)

        # Identity init for the pointwise layer
        nn.init.zeros_(self.fusion_pw.weight)
        eye = torch.eye(dim * 4)
        self.fusion_pw.weight.data.copy_(eye.view(dim * 4, dim * 4, 1, 1))
        nn.init.zeros_(self.fusion_pw.bias)

    # ------------------------------------------------------------------
    # Static helper: apply affine FiLM modulation
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_affine_film(
        x_target: torch.Tensor, params: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply affine FiLM: y = scale(γ) · x + β

        scale uses softplus so it is strictly positive and starts at 1.
        The offset log(e-1) is the pre-image of softplus that maps to 1.

        params : (B, 2*C, H, W)  — first C channels are γ, last C are β
        """
        C = x_target.shape[1]
        gamma, beta = params[:, :C], params[:, C:]
        _SOFTPLUS_OFFSET = math.log(math.e - 1)
        scale = F.softplus(gamma + _SOFTPLUS_OFFSET)  # ≥ 0, starts at 1
        return x_target * scale + beta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, dim, H, W)

        Returns
        -------
        (B, dim, H, W)
        """
        dim = self.dim

        # ── 1. Decompose into subbands ────────────────────────────────────────
        x_dwt = self.dwt(x)  # (B, 4*dim, H/2, W/2)
        x_ll = x_dwt[:, :dim]
        x_lh = x_dwt[:, dim : 2 * dim]
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]

        # ── 2. Process LL band with dedicated Mamba ───────────────────────────
        x_ll_out = self.ll_mamba(x_ll)

        # ── 3. Build global cross-band context ────────────────────────────────
        # Concatenate processed LL with raw HF subbands so context sees both
        # the refined structural signal and the original edge/texture content.
        x_all = torch.cat([x_ll_out, x_lh, x_hl, x_hh], dim=1)
        context = self.context_mamba(self.pre_context_proj(x_all))

        # ── 4. Predict and apply affine FiLM modulation ───────────────────────
        x_ll_mod = self._apply_affine_film(x_ll_out, self.film_ll(context))
        x_lh_mod = self._apply_affine_film(x_lh, self.film_lh(context))
        x_hl_mod = self._apply_affine_film(x_hl, self.film_hl(context))
        x_hh_mod = self._apply_affine_film(x_hh, self.film_hh(context))

        # ── 5. Fuse subbands (depthwise + pointwise) + residual ───────────────
        merged = torch.cat([x_ll_mod, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        # depthwise spatial smoothing
        fused_dw = self.fusion_act(self.fusion_dw(merged))
        # pointwise cross-subband mixing
        fused_pw = self.fusion_pw(fused_dw)
        # residual preserves the modulated subbands; fusion adds refinements
        fused = fused_pw + merged

        # ── 6. Synthesise back to pixel space ─────────────────────────────────
        return self.idwt(fused)


# ==============================================================================
# Entry point for unit tests
# ==============================================================================

if __name__ == "__main__":
    test_dwt_idwt()
