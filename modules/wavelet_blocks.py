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

    Replaces the original residual-add updater which had no gate control,
    allowing unbounded memory drift.  The gate learns how much of the previous
    memory state to retain vs. how much to write from the new slice.

        gate   = σ(W_g · [memory ‖ new_info])
        update = tanh(W_u · [memory ‖ new_info])
        output = gate * memory + (1 − gate) * update

    This is equivalent to the GRU update equation without the reset gate
    (reset gate empirically unhelpful in compression memory chains).

    Args:
        slice_ch : number of channels in memory / new_info tensors
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
        # Zero-init update output: memory starts as pure identity pass-through.
        nn.init.zeros_(self.update[-2].weight)
        nn.init.zeros_(self.update[-2].bias)

    def forward(self, memory: torch.Tensor, new_info: torch.Tensor) -> torch.Tensor:
        x = torch.cat([memory, new_info], dim=1)
        g = self.gate(x)  # (B, slice_ch, H, W) ∈ (0,1)
        u = self.update(x)  # (B, slice_ch, H, W) ∈ (-1,1)
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
    3. HF sub-bands predict FiLM parameters (gamma, beta) to modulate LL output.
    4. Modulated LL is concatenated with HF, fused, and reconstructed via IDWT.

    FiLM modulation:
        x_ll_modulated = x_ll_out * (1 + gamma) + beta

    -----------------------------------------------------------------------
    DWT→Mamba→FiLM→IDWT path.
    -----------------------------------------------------------------------

    Args:
        dim      : number of input channels
        drop_path: stochastic depth rate for VSSBlock
    """

    def __init__(self, dim: int, drop_path: float = 0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")
        self.dim = dim

        # LL sub-band: long-range global structure via Mamba.
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # HF sub-bands → FiLM parameters (gamma, beta) for LL modulation.
        # Input : 3 * dim  (LH + HL + HH)
        # Output: 2 * dim  (gamma and beta, each dim channels)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 2, kernel_size=1),
        )
        # Zero-init: at t=0 gamma=0 → scale=1, beta=0 → no shift.
        nn.init.zeros_(self.hf_conv[-1].weight)
        nn.init.zeros_(self.hf_conv[-1].bias)

        # Fusion before IDWT: mix modulated LL with HF sub-bands.
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
        x_hf = x_dwt[:, dim:]  # (B, 3*dim, H/2, W/2)  [LH, HL, HH]
        x_ll_out = self.ll_mamba(x_ll)  # (B, dim, H/2, W/2)

        # 3. HF context → FiLM parameters.
        hf_gamma_beta = self.hf_conv(x_hf)  # (B, 2*dim, H/2, W/2)
        gamma, beta = hf_gamma_beta.chunk(2, dim=1)  # each (B, dim, H/2, W/2)

        # 4. Cross-frequency FiLM modulation.
        x_ll_modulated = x_ll_out * (1.0 + gamma) + beta  # (B, dim, H/2, W/2)

        # 5. Fuse all four sub-bands and reconstruct.
        merged = torch.cat([x_ll_modulated, x_hf], dim=1)  # (B, 4*dim, H/2, W/2)
        fused = self.fusion(merged) + merged  # (B, 4*dim, H/2, W/2)

        return self.idwt(fused)  # (B, dim, H, W)
