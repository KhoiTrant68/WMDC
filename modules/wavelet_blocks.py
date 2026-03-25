import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock

# ---------------------------------------------------------------------------
# Discrete Wavelet Transform (2-D, single level, Haar)
# ---------------------------------------------------------------------------


class DWT_2D(nn.Module):
    """
    2-D single-level DWT using fixed Haar filters.
    Output channel order: [LL | LH | HL | HH], each of size `dim` channels.
    """

    def __init__(self, wave: str = "haar"):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.tensor(w.dec_hi[::-1], dtype=torch.float32)
        dec_lo = torch.tensor(w.dec_lo[::-1], dtype=torch.float32)

        def _make(lo, hi):
            return (lo.unsqueeze(0) * hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0)

        self.register_buffer("w_ll", _make(dec_lo, dec_lo))
        self.register_buffer("w_lh", _make(dec_lo, dec_hi))
        self.register_buffer("w_hl", _make(dec_hi, dec_lo))
        self.register_buffer("w_hh", _make(dec_hi, dec_hi))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        x_ll = F.conv2d(x, self.w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, self.w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, self.w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, self.w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  # (B, 4*dim, H/2, W/2)


# ---------------------------------------------------------------------------
# Inverse Discrete Wavelet Transform (2-D, single level, Haar)
# ---------------------------------------------------------------------------


class IDWT_2D(nn.Module):
    """
    2-D single-level IDWT using fixed Haar reconstruction filters.
    Reconstructs from (LL, LH, HL, HH) concatenated along channel dim.
    """

    def __init__(self, wave: str = "haar"):
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
        x = x.view(B, 4, -1, H, W).transpose(1, 2).reshape(B, -1, H, W)
        filters_expanded = self.filters.repeat(C // 4, 1, 1, 1)
        return F.conv_transpose2d(x, filters_expanded, stride=2, groups=C // 4)


# ---------------------------------------------------------------------------
# GatedMemoryUpdater
# ---------------------------------------------------------------------------


class GatedMemoryUpdater(nn.Module):
    """
    GRU-style gated memory update.

        new_memory = g ⊙ memory + (1 − g) ⊙ update(memory, new_info)

    Initialisation:
      - Gate bias = +3.0  → sigmoid(3) ≈ 0.95 at init
        → new_memory ≈ memory  (near-identity, stable first epoch)
      - Update branch final conv weight/bias = 0
        → update(x) = tanh(0) = 0 at init
        → new_memory = 0.95 * memory + 0.05 * 0 ≈ memory  ✓
    """

    def __init__(self, slice_ch: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=1),
            nn.Sigmoid(),
        )
        self.update = nn.Sequential(
            nn.Conv2d(slice_ch * 2, slice_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(slice_ch, slice_ch, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        # gate bias → +3.0 so the gate starts nearly open.
        nn.init.constant_(self.gate[0].bias, 3.0)

        # Zero-init update branch: zero output at init → no update at init.
        nn.init.zeros_(self.update[-2].weight)
        nn.init.zeros_(self.update[-2].bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        x = torch.cat([memory, new_info], dim=1)
        g = self.gate(x)  # ≈ 0.95 at init
        u = self.update(x)  # ≈ 0.0 at init
        return g * memory + (1.0 - g) * u


# ---------------------------------------------------------------------------
# FrequencyDisentangledMamba
# ---------------------------------------------------------------------------


class FrequencyDisentangledMamba(nn.Module):
    """
    Cross-Frequency Mamba Modulation.

    Architecture
    ------------
    1. DWT decomposes input into LL (low-freq) + {LH, HL, HH} (high-freq).
    2. LL is processed by a Mamba VSSBlock (global long-range structure).
    3. LL structure predicts FiLM parameters (gamma, beta) for each HF
       sub-band via SEPARATE sub-band-specific predictors.
    4. Modulated HF is concatenated with LL, fused, and reconstructed via IDWT.

    FiLM modulation per sub-band b:
        x_hf_b_mod = x_hf_b ⊙ (1 + clamp(gamma_b, −1, 1)) + beta_b

    The gamma clamp prevents the degenerate mode where the network sets
    gamma ≈ −1 to zero-out all high-frequency content.  Without this guard,
    high-lambda (low-bitrate) models may collapse to blur-only reconstructions.

    Zero-init of the FiLM output layers ensures identity at training start.
    """

    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")
        self.dim = dim

        # LL sub-band: long-range global structure via Mamba.
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # Per-sub-band FiLM predictors (LH, HL, HH treated independently).
        # Input: dim channels (LL output)
        # Output: 2*dim channels per sub-band (gamma + beta)
        # Zero-init final layer → identity modulation at init.
        def _make_film_predictor(in_dim: int, out_dim: int) -> nn.Sequential:
            net = nn.Sequential(
                nn.Conv2d(in_dim, in_dim * 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(in_dim * 2, out_dim * 2, kernel_size=1),
            )
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
            return net

        self.film_lh = _make_film_predictor(dim, dim)  # for LH sub-band
        self.film_hl = _make_film_predictor(dim, dim)  # for HL sub-band
        self.film_hh = _make_film_predictor(dim, dim)  # for HH sub-band

        # Fusion before IDWT: depthwise spatial + pointwise channel mixing.
        self.fusion = nn.Sequential(
            # Spatial mixing (3x3 depthwise) to capture local high-frequency geometry
            nn.Conv2d(dim * 4, dim * 4, kernel_size=3, padding=1, groups=dim * 4),
            nn.GELU(),
            # Channel mixing (1x1) to fuse the modulated sub-bands
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
        )

    @staticmethod
    def _apply_film(
        x_hf: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply FiLM modulation with gamma clamped to (−1, +1).

        The clamp prevents the degenerate case where gamma → −1 zeroes out
        all high-frequency content, causing blur-only reconstructions at high
        lambda / low bitrate.
        """
        gamma_clamped = gamma.clamp(-1.0, 1.0)
        return x_hf * (1.0 + gamma_clamped) + beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim

        # 1. Frequency decomposition → (B, 4*dim, H/2, W/2)
        x_dwt = self.dwt(x)

        x_ll = x_dwt[:, :dim]  # (B, dim, H/2, W/2)
        x_lh = x_dwt[:, dim : 2 * dim]  # (B, dim, H/2, W/2)
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]

        # 2. Global structure from LL via Mamba.
        x_ll_out = self.ll_mamba(x_ll)  # (B, dim, H/2, W/2)

        # 3. Per-sub-band FiLM modulation predicted from global LL structure.
        gamma_lh, beta_lh = self.film_lh(x_ll_out).chunk(2, dim=1)
        gamma_hl, beta_hl = self.film_hl(x_ll_out).chunk(2, dim=1)
        gamma_hh, beta_hh = self.film_hh(x_ll_out).chunk(2, dim=1)

        # 4. Modulate each HF sub-band independently.
        x_lh_mod = self._apply_film(x_lh, gamma_lh, beta_lh)
        x_hl_mod = self._apply_film(x_hl, gamma_hl, beta_hl)
        x_hh_mod = self._apply_film(x_hh, gamma_hh, beta_hh)

        # 5. Fuse all four sub-bands and reconstruct.
        merged = torch.cat([x_ll_out, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        fused = self.fusion(merged) + merged  # (B, 4*dim, H/2, W/2)

        return self.idwt(fused)  # (B, dim, H, W)
