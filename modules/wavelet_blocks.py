import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock


class DWT_2D(nn.Module):
    def __init__(self, wave="haar"):
        super().__init__()
        w = pywt.Wavelet(wave)
        dec_hi, dec_lo = torch.Tensor(w.dec_hi[::-1]), torch.Tensor(w.dec_lo[::-1])
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

    def forward(self, x):
        dim = x.shape[1]
        x_ll = F.conv2d(x, self.w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_lh = F.conv2d(x, self.w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hl = F.conv2d(x, self.w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim)
        x_hh = F.conv2d(x, self.w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim)
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)


class IDWT_2D(nn.Module):
    def __init__(self, wave="haar"):
        super().__init__()
        w = pywt.Wavelet(wave)
        rec_hi, rec_lo = torch.Tensor(w.rec_hi), torch.Tensor(w.rec_lo)
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

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2).reshape(B, -1, H, W)
        filters_expanded = self.filters.repeat(C // 4, 1, 1, 1)
        # Using native F.conv_transpose2d
        return F.conv_transpose2d(x, filters_expanded, stride=2, groups=C // 4)


class FrequencyDisentangledMamba(nn.Module):
    """
    Cross-Frequency Mamba Modulation.
    Routes Smooth LL to Mamba, while Periodic HF components dynamically shift
    and scale the Mamba states (FiLM), achieving true frequency-disentangled fusion.
    """

    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # Predicts both Gamma and Beta for FiLM modulation
        self.hf_conv = nn.Sequential(
            nn.Conv2d(dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3),
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 2, kernel_size=1),
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1, groups=4),
        )
        self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)

        # 1. Frequency Decomposition
        x_dwt = self.dwt(x)

        # 2. Long-Range Global Structure Extraction
        x_ll_out = self.ll_mamba(x_dwt[:, : x.size(1)])

        # 3. Local High-Frequency Context Modulators
        hf_gamma_beta = self.hf_conv(x_dwt[:, x.size(1) :])
        gamma, beta = hf_gamma_beta.chunk(2, dim=1)

        # 4. Cross-Frequency Spatial Modulation (FiLM)
        # Allows edges (HF) to govern how flat regions (LL) are represented
        x_ll_modulated = x_ll_out * (1 + gamma) + beta

        # 5. Fusion and Inverse Wavelet Transform
        merged = torch.cat([x_ll_modulated, x_dwt[:, x.size(1) :]], dim=1)
        fused = self.fusion(merged) + merged

        out = self.idwt(fused)
        return out + identity
