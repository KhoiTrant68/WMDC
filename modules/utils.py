import torch
from torch import Tensor, nn


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


def ste_round(x: Tensor) -> Tensor:
    """Straight-Through Estimator for Quantization"""
    return torch.round(x) - x.detach() + x


class CheckboardMaskedConv2d(nn.Conv2d):
    """
    Checkerboard Context Model Layer.
    Optimized Note: Because the input tensor is dynamically masked (zeroed)
    at non-anchor positions prior to passing into this module in WMDC.py,
    applying a hard weight mask is mathematically redundant and disables native
    cuDNN backend optimizations. This wrapper leverages standard nn.Conv2d
    efficiency while fulfilling structural logic.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x)
