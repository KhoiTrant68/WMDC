import pywt
import torch
import torch.nn as nn
from torch.autograd import Function

from modules.VSS_module import VSSBlock


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(
            x, w_ll.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x_lh = torch.nn.functional.conv2d(
            x, w_lh.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x_hl = torch.nn.functional.conv2d(
            x, w_hl.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        x_hh = torch.nn.functional.conv2d(
            x, w_hh.expand(dim, -1, -1, -1), stride=2, groups=dim
        )
        return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H_out, W_out = dx.shape
            dx = (
                dx.view(B, 4, -1, H_out, W_out)
                .transpose(1, 2)
                .reshape(B, -1, H_out, W_out)
            )
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C // 4, 1, 1, 1)
            grad_x = torch.nn.functional.conv_transpose2d(
                dx, filters, stride=2, groups=C // 4
            )
            return grad_x, None, None, None, None
        return None, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        B, C, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2).reshape(B, -1, H, W)
        filters_expanded = filters.repeat(C // 4, 1, 1, 1)
        return torch.nn.functional.conv_transpose2d(
            x, filters_expanded, stride=2, groups=C // 4
        )

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = dx.shape
            dx = dx.contiguous()
            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(
                dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C
            )
            x_lh = torch.nn.functional.conv2d(
                dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C
            )
            x_hl = torch.nn.functional.conv2d(
                dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C
            )
            x_hh = torch.nn.functional.conv2d(
                dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride=2, groups=C
            )
            return torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1), None
        return None, None


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
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)


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
        return IDWT_Function.apply(x, self.filters)


class FrequencyDisentangledMamba(nn.Module):
    """FD-SSM: Routes Smooth LL to Mamba, and Periodic HF to local CNN."""

    def __init__(self, dim, drop_path=0.1):
        super().__init__()
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        # LL Band: Global Mamba Receptive Field
        self.ll_mamba = VSSBlock(hidden_dim=dim, drop_path=drop_path)

        # HF Band: Local, lightweight texture processing (fixes O(C^2) explosion)
        self.hf_conv = nn.Sequential(
            nn.Conv2d(
                dim * 3, dim * 3, kernel_size=3, padding=1, groups=dim * 3
            ),  # Depthwise
            nn.GELU(),
            nn.Conv2d(dim * 3, dim * 3, kernel_size=1),  # Pointwise
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim * 4, kernel_size=1, groups=4),
        )
        self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        x_dwt = self.dwt(x)

        x_ll_out = self.ll_mamba(x_dwt[:, : x.size(1)])
        x_hf_out = self.hf_conv(x_dwt[:, x.size(1) :])

        merged = torch.cat([x_ll_out, x_hf_out], dim=1)
        fused = self.fusion(merged) + merged

        out = self.idwt(fused)
        return out + identity
