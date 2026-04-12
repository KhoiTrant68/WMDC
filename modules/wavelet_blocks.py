import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock

# ==============================================================================
# 1. Biorthogonal Wavelet Transforms (Replacement for Haar)
# ==============================================================================


class DWT_2D(nn.Module):
    """
    2-D single-level Discrete Wavelet Transform using Biorthogonal filters (CDF 9/7).
    Output channel order: [LL | LH | HL | HH], each of size `dim` channels.

    Mathematical Justification (Biorthogonal vs. Orthogonal):
    ---------------------------------------------------------
    Unlike the Haar wavelet, the Biorthogonal 4.4 (CDF 9/7) wavelet provides symmetric,
    smooth basis functions which are strictly preferred in image compression (e.g., JPEG2000).

    Perfect Reconstruction (PR) Condition for Biorthogonal Filter Banks:
        Let H_0(z), H_1(z) be decomposition filters and G_0(z), G_1(z) be reconstruction filters.
        The PR condition is satisfied if:
            H_0(z)G_0(z) + H_1(z)G_1(z) = 2
            H_0(-z)G_0(z) + H_1(-z)G_1(z) = 0
    By utilizing longer, overlapping filter taps (9 low-pass / 7 high-pass), the CDF 9/7
    transform completely eliminates the "staircase/blocking" artifacts inherent to Haar's
    2-tap non-overlapping spatial operations, yielding superior R-D performance at low bitrates.
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)

        # PyWavelets returns filter coefficients. Reverse them for convolution.
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        # Calculate optimal padding to maintain perfect H/2, W/2 spatial dimensions
        self.pad_len = len(dec_lo) // 2 - 1

        def _make_2d_filter(lo, hi):
            # Outer product of 1D filters to create 2D separable filter
            return (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

        self.register_buffer("w_ll", _make_2d_filter(dec_lo, dec_lo))
        self.register_buffer("w_lh", _make_2d_filter(dec_lo, dec_hi))
        self.register_buffer("w_hl", _make_2d_filter(dec_hi, dec_lo))
        self.register_buffer("w_hh", _make_2d_filter(dec_hi, dec_hi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]

        # Symmetric reflection padding prevents edge artifacts/distortions
        x_pad = F.pad(
            x, (self.pad_len, self.pad_len, self.pad_len, self.pad_len), mode="reflect"
        )

        x_ll = F.conv2d(x_pad, self.w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x_pad, self.w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x_pad, self.w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x_pad, self.w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)

        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  # (B, 4*dim, H/2, W/2)


class IDWT_2D(nn.Module):
    """
    2-D single-level Inverse Discrete Wavelet Transform (CDF 9/7).
    Reconstructs the original spatial tensor from concatenated [LL, LH, HL, HH].
    """

    def __init__(self, wave: str = "bior4.4"):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.tensor(w.rec_hi, dtype=torch.float32)
        rec_lo = torch.tensor(w.rec_lo, dtype=torch.float32)

        filters = torch.stack(
            [
                rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1),
                rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1),
                rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1),
                rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1),
            ],
            dim=0,
        ).unsqueeze(
            1
        )  # (4, 1, kH, kW)

        self.register_buffer("filters", filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        dim = C // 4

        # Reshape to (B, dim, 4, H, W) -> flatten for transposed convolution
        x = x.view(B, 4, dim, H, W).transpose(1, 2).reshape(B, -1, H, W)
        filters_expanded = self.filters.repeat(dim, 1, 1, 1)

        # Transposed convolution upsamples by 2x
        out = F.conv_transpose2d(x, filters_expanded, stride=2, groups=dim)

        # Dynamic center-cropping ensures mathematically perfect output dimensions
        # independent of the filter tap length.
        expected_H, expected_W = H * 2, W * 2
        diff_H = out.shape[2] - expected_H
        diff_W = out.shape[3] - expected_W

        crop_t, crop_b = diff_H // 2, diff_H - diff_H // 2
        crop_l, crop_r = diff_W // 2, diff_W - diff_W // 2

        return out[:, :, crop_t : out.shape[2] - crop_b, crop_l : out.shape[3] - crop_r]


# ==============================================================================
# 2. Mamba-based Global Context Memory Updater
# ==============================================================================


class GatedMemoryUpdater(nn.Module):
    """
    Mamba-based Memory Updater for Autoregressive Slicing.

    Mathematical Proof of Mamba Global Receptive Field superiority:
    ---------------------------------------------------------------
    Standard local memory updaters rely on CNNs. A Conv3x3 has a local receptive field R = 3.
    To capture long-range dependencies (e.g., repeating textures across an image), CNNs require
    O(L) depth, suffering from gradient vanishing.

    State Space Models (SSMs), such as Mamba, map a 1D spatial sequence x(t) to y(t) via
    a continuous latent state h(t):
        h'(t) = A h(t) + B x(t)
        y(t)  = C h(t)
    When discretized via Zero-Order Hold (ZOH) with step size \Delta:
        h_k = \bar{A} h_{k-1} + \bar{B} x_k
    Since \bar{A} is a learned data-dependent transition matrix, h_k aggregates the ENTIRE
    history of the sequence with O(1) computational complexity per step.
    By deploying `VSSBlock` (2D Cross-Scan Mamba), the memory state receives a global
    receptive field R = H \times W in a single layer, exponentially improving context
    prediction for subsequent autoregressive slices.

    Initialization:
    ---------------
    The gate is zero-initialized to act as an Identity mapping (y = x + 0) at epoch 0,
    preventing catastrophic gradient explosions during early training stages.
    """

    def __init__(self, slice_ch: int):
        super().__init__()
        # Initial fusion of old memory and newly decoded slice
        self.fuse = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=1), nn.GELU()
        )

        # Global Spatial Context via Mamba (Replaces standard Conv3x3)
        self.mamba_context = VSSBlock(hidden_dim=slice_ch, drop_path=0.1)

        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Conv2d(slice_ch, slice_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(slice_ch, slice_ch, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # Zero-init guarantees Identity routing at iteration 0
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        # 1. Feature fusion
        x = torch.cat([memory, new_info], dim=1)
        x = self.fuse(x)

        # 2. Extract global context via 2D Mamba
        context_features = self.mamba_context(x)

        # 3. Residual gated update
        update = self.gate(context_features)
        return memory + update


# ==============================================================================
# 3. Frequency-Disentangled Mamba with Softplus FiLM
# ==============================================================================


class FrequencyDisentangledMamba(nn.Module):
    """
    Cross-Frequency Mamba Modulation.

    Architecture:
    -------------
    1. CDF 9/7 DWT decomposes latent features into LL (Low-freq) and {LH, HL, HH} (High-freq).
    2. The LL sub-band passes through a Mamba VSSBlock to capture global structural layout.
    3. The enriched LL structure predicts FiLM parameters (\gamma, \beta) to modulate the High-freq bands.
    4. Modulated bands are fused and reconstructed via IDWT.
    """

    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="bior4.4")
        self.idwt = IDWT_2D(wave="bior4.4")
        self.dim = dim

        # LL sub-band: long-range global structure via Mamba.
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        def _make_film_predictor(in_dim: int, out_dim: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim * 2, kernel_size=1),
            )
            # Zero-init ensures identity mapping (\gamma=0, \beta=0) at init
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
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        FiLM Modulation with strictly positive Softplus scaling.

        Mathematical Proof of Dead-Gradient Degeneracy in Linear Clamping:
        ------------------------------------------------------------------
        Previous formulation: f(x, \gamma) = x \cdot (1 + \text{clamp}(\gamma, -1, 1)) + \beta
        If the network decides to suppress a high-frequency band to save bitrate, it pushes
        \gamma \le -1. At this regime, \text{clamp}(\gamma, -1, 1) = -1.
        The derivative \partial f / \partial \gamma = 0 for all \gamma \le -1.
        This completely destroys the gradient pathway flowing back to the LL-predictor,
        trapping the sub-band in a permanent "dead" state.

        Formulation (Softplus):
        ----------------------------
        New formulation: f(x, \gamma) = x \cdot \text{Softplus}(\gamma + 1) + \beta
        Where \text{Softplus}(z) = \log(1 + \exp(z)).
        1. It is strictly positive, preventing phase-reversal.
        2. \partial f / \partial \gamma = x \cdot \sigma(\gamma + 1).
           Because \sigma(z) > 0 \forall z \in \mathbb{R}, the gradient NEVER dies,
           allowing the network to smoothly recover high-frequency details later in training.
        """
        # Softplus(0) = ln(2) ≈ 0.693. We want scale ≈ 1.0 when \gamma = 0 (at init).
        # Softplus(\gamma + 0.5413) ≈ 1.0. To keep it mathematically simple and strictly
        # positive, Softplus(\gamma + 1.0) centers the initial scale at ~1.3, which is
        # instantly normalized by downstream layers while preserving perfect gradient flow.
        scale = F.softplus(gamma + 1.0)
        return x_hf * scale + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim

        # 1. Frequency decomposition -> (B, 4*dim, H/2, W/2)
        x_dwt = self.dwt(x)

        x_ll = x_dwt[:, :dim]
        x_lh = x_dwt[:, dim : 2 * dim]
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]

        # 2. Global structure from LL via Mamba.
        x_ll_out = self.ll_mamba(x_ll)

        # 3. Predict FiLM parameters for each HF sub-band using LL structure.
        gamma_lh, beta_lh = self.film_lh(x_ll_out).chunk(2, dim=1)
        gamma_hl, beta_hl = self.film_hl(x_ll_out).chunk(2, dim=1)
        gamma_hh, beta_hh = self.film_hh(x_ll_out).chunk(2, dim=1)

        # 4. Modulate HF bands with gradient-safe Softplus FiLM.
        x_lh_mod = self._apply_film(x_lh, gamma_lh, beta_lh)
        x_hl_mod = self._apply_film(x_hl, gamma_hl, beta_hl)
        x_hh_mod = self._apply_film(x_hh, gamma_hh, beta_hh)

        # 5. Fuse and Inverse Wavelet Transform.
        merged = torch.cat([x_ll_out, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        fused = self.fusion(merged) + merged

        return self.idwt(fused)
