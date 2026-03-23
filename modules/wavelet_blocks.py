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
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)


# ---------------------------------------------------------------------------
# Inverse Discrete Wavelet Transform (2-D, single level)
# ---------------------------------------------------------------------------


class IDWT_2D(nn.Module):
    """
    2-D single-level IDWT using fixed Haar reconstruction filters.
    Reconstructs from (LL, LH, HL, HH) concatenated along the channel dim.
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
# FrequencyDisentangledMamba
# ---------------------------------------------------------------------------


class FrequencyDisentangledMamba(nn.Module):
    """
    Cross-Frequency Mamba Modulation.

    Architecture
    ------------
    1. DWT decomposes input into LL (low-freq) + {LH, HL, HH} (high-freq) sub-bands.
    2. LL is processed by a Mamba-based VSSBlock (global long-range structure).
    3. HF sub-bands predict FiLM parameters (gamma, beta) to modulate the LL output.
    4. Modulated LL is concatenated with HF sub-bands, fused, and reconstructed via IDWT.

    FiLM modulation:
        x_ll_modulated = x_ll_out * (1 + gamma) + beta

    Key fix: hf_conv final layer is zero-initialised so that at the start of
    training gamma=0 and beta=0, giving an identity transform.  This prevents
    the random initialisation from immediately corrupting the Mamba output.

    Args:
        dim      : number of input channels
        drop_path: stochastic depth rate for VSSBlock
    """

    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        # LL sub-band: long-range global structure via Mamba
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # HF sub-bands → FiLM parameters (gamma, beta) for LL modulation
        # Input: 3 * dim  (LH + HL + HH channels)
        # Output: 2 * dim (gamma and beta, each dim channels)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 2, kernel_size=1),
        )
        # ── FiLM zero-initialisation ──────────────────────────────────────
        # At t=0: gamma=0 → scale=(1+0)=1, beta=0 → no shift.
        # The network learns to deviate from identity incrementally.
        # Without this, random gamma/beta immediately corrupts LL Mamba output.
        nn.init.zeros_(self.hf_conv[-1].weight)
        nn.init.zeros_(self.hf_conv[-1].bias)

        # Fusion before IDWT: mix modulated LL with HF sub-bands
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1, groups=4),
        )
        self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        # 1. Frequency decomposition → (B, 4*dim, H/2, W/2)
        x_dwt = self.dwt(x)

        # 2. Global structure from LL sub-band
        x_ll_out = self.ll_mamba(x_dwt[:, : x.size(1)])  # (B, dim, H/2, W/2)

        # 3. Local HF context → FiLM parameters
        hf_gamma_beta = self.hf_conv(x_dwt[:, x.size(1) :])  # (B, 2*dim, H/2, W/2)
        gamma, beta = hf_gamma_beta.chunk(2, dim=1)

        # 4. Cross-frequency FiLM modulation
        x_ll_modulated = x_ll_out * (1 + gamma) + beta  # (B, dim, H/2, W/2)

        # 5. Fuse and reconstruct
        merged = torch.cat([x_ll_modulated, x_dwt[:, x.size(1) :]], dim=1)
        fused = self.fusion(merged) + merged  # (B, 4*dim, H/2, W/2)

        out = self.idwt(fused)  # (B, dim, H, W)
        return out + identity
