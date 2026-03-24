import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock

# ---------------------------------------------------------------------------
# Discrete Wavelet Transform (2-D, single level)
# ---------------------------------------------------------------------------


class DWT_2D(nn.Module):
    """
    2-D single-level DWT using fixed Haar filters.
    Decomposes input into LL, LH, HL, HH sub-bands.
    Output channel order: [LL | LH | HL | HH], each of size `dim` channels.
    """

    def __init__(self, wave: str = "haar"):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1])
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        self.register_buffer(
            "w_ll",
            (dec_lo.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "w_lh",
            (dec_lo.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "w_hl",
            (dec_hi.unsqueeze(0) * dec_lo.unsqueeze(1)).unsqueeze(0).unsqueeze(0),
        )
        self.register_buffer(
            "w_hh",
            (dec_hi.unsqueeze(0) * dec_hi.unsqueeze(1)).unsqueeze(0).unsqueeze(0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = x.shape[1]
        x_ll = F.conv2d(x, self.w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, self.w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, self.w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, self.w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)  # (B, 4*dim, H/2, W/2)


# ---------------------------------------------------------------------------
# Inverse Discrete Wavelet Transform (2-D, single level)
# ---------------------------------------------------------------------------


class IDWT_2D(nn.Module):
    """
    2-D single-level IDWT using fixed Haar reconstruction filters.
    Reconstructs from (LL, LH, HL, HH) concatenated along channel dim.
    """

    def __init__(self, wave: str = "haar"):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)

        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)

        filters = torch.cat(
            [
                w_ll.unsqueeze(0).unsqueeze(1),
                w_lh.unsqueeze(0).unsqueeze(1),
                w_hl.unsqueeze(0).unsqueeze(1),
                w_hh.unsqueeze(0).unsqueeze(1),
            ],
            dim=0,
        )
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
        nn.init.zeros_(self.update[-2].weight)
        nn.init.zeros_(self.update[-2].bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        x = torch.cat([memory, new_info], dim=1)
        g = self.gate(x)
        u = self.update(x)
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
       sub-band via **separate** sub-band-specific predictors.
       Separate predictors are justified because LH (horizontal edges),
       HL (vertical edges) and HH (diagonal / texture) have fundamentally
       different statistics and should be modulated independently.
    4. Modulated HF is concatenated with LL, fused, and reconstructed via
       IDWT.

    FiLM modulation per sub-band b:
        x_hf_b_modulated = x_hf_b * (1 + gamma_b) + beta_b

    Zero-init of the FiLM output layers ensures identity at start of training.
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

        # Fusion before IDWT: mix LL with modulated HF sub-bands.
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1, groups=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = self.dim

        # 1. Frequency decomposition → (B, 4*dim, H/2, W/2)
        x_dwt = self.dwt(x)

        # 2. Global structure from LL sub-band (channels 0..dim-1).
        x_ll = x_dwt[:, :dim]  # (B, dim, H/2, W/2)
        x_lh = x_dwt[:, dim : 2 * dim]  # (B, dim, H/2, W/2)
        x_hl = x_dwt[:, 2 * dim : 3 * dim]
        x_hh = x_dwt[:, 3 * dim :]
        x_ll_out = self.ll_mamba(x_ll)  # (B, dim, H/2, W/2)

        # 3. Per-sub-band FiLM modulation predicted from global LL structure.
        gamma_lh, beta_lh = self.film_lh(x_ll_out).chunk(2, dim=1)
        gamma_hl, beta_hl = self.film_hl(x_ll_out).chunk(2, dim=1)
        gamma_hh, beta_hh = self.film_hh(x_ll_out).chunk(2, dim=1)

        # 4. Apply modulation independently to each HF sub-band.
        x_lh_mod = x_lh * (1.0 + gamma_lh) + beta_lh
        x_hl_mod = x_hl * (1.0 + gamma_hl) + beta_hl
        x_hh_mod = x_hh * (1.0 + gamma_hh) + beta_hh

        # 5. Fuse all four sub-bands and reconstruct.
        merged = torch.cat([x_ll_out, x_lh_mod, x_hl_mod, x_hh_mod], dim=1)
        fused = self.fusion(merged) + merged  # (B, 4*dim, H/2, W/2)

        return self.idwt(fused)  # (B, dim, H, W)
