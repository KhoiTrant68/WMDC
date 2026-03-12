import pywt
import torch
import torch.nn as nn
from compressai.layers import GDN, conv3x3, subpel_conv3x3
from torch import Tensor
from torch.autograd import Function

from modules.utils import conv1x1


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape
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
            B, C, H, W = ctx.shape
            dx = (
                dx.view(B, 4, -1, H // 2, W // 2)
                .transpose(1, 2)
                .reshape(B, -1, H // 2, W // 2)
            )
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)
        return dx, None, None, None, None


class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape
        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2).reshape(B, -1, H, W)
        C = x.shape[1]
        filters = filters.repeat(C // 4, 1, 1, 1)
        return torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C // 4)

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors[0]
            B, C, H, W = ctx.shape
            C = C // 4
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
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None


class DWT_2D(nn.Module):
    def __init__(self, wave="haar"):
        super(DWT_2D, self).__init__()
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
        return DWT_Function.apply(
            x,
            self.w_ll.to(x.dtype),
            self.w_lh.to(x.dtype),
            self.w_hl.to(x.dtype),
            self.w_hh.to(x.dtype),
        )


class IDWT_2D(nn.Module):
    def __init__(self, wave="haar"):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi, rec_lo = torch.Tensor(w.rec_hi), torch.Tensor(w.rec_lo)
        w_ll = rec_lo.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0) * rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0) * rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0) * rec_hi.unsqueeze(1)
        self.register_buffer(
            "filters",
            torch.cat(
                [
                    w_ll.unsqueeze(0).unsqueeze(1),
                    w_lh.unsqueeze(0).unsqueeze(1),
                    w_hl.unsqueeze(0).unsqueeze(1),
                    w_hh.unsqueeze(0).unsqueeze(1),
                ],
                dim=0,
            ),
        )

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters.to(x.dtype))


class ResidualBlockWithStride_wave(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, wavelet="haar"):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)
        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.gdn_low = GDN(out_ch)
        self.high_freq_conv = conv3x3(3 * out_ch, 3 * out_ch)
        self.gdn_high = GDN(3 * out_ch)
        self.skip = (
            conv1x1(in_ch, out_ch, stride=stride)
            if (stride != 1 or in_ch != out_ch)
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        out = self.leaky_relu(self.conv1(x))
        dwt_output = self.dwt(out)
        low_freq, high_freq = dwt_output[:, : out.size(1)], dwt_output[:, out.size(1) :]
        low_freq_processed = self.gdn_low(self.low_freq_conv(low_freq))
        high_freq_processed = self.gdn_high(self.high_freq_conv(high_freq))
        return (
            self.idwt(torch.cat([low_freq_processed, high_freq_processed], dim=1))
            + identity
        )


class ResidualBlockUpsample_wave(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2, wavelet="haar"):
        super().__init__()
        self.subpel_conv = subpel_conv3x3(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.dwt = DWT_2D(wave=wavelet)
        self.idwt = IDWT_2D(wave=wavelet)
        self.low_freq_conv = conv3x3(out_ch, out_ch)
        self.igdn_low = GDN(out_ch, inverse=True)
        self.high_freq_conv = conv3x3(3 * out_ch, 3 * out_ch)
        self.igdn_high = GDN(3 * out_ch, inverse=True)
        self.upsample_skip = subpel_conv3x3(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = self.upsample_skip(x)
        out = self.leaky_relu(self.subpel_conv(x))
        dwt_output = self.dwt(out)
        low_freq, high_freq = dwt_output[:, : out.size(1)], dwt_output[:, out.size(1) :]
        low_freq_processed = self.igdn_low(self.low_freq_conv(low_freq))
        high_freq_processed = self.igdn_high(self.high_freq_conv(high_freq))
        return (
            self.idwt(torch.cat([low_freq_processed, high_freq_processed], dim=1))
            + identity
        )
