import torch
from torch import Tensor, nn


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer("mask", torch.ones_like(self.weight))
        self.mask[:, :, 0::2, 0::2] = 0  # mask (even, even)
        self.mask[:, :, 1::2, 1::2] = 0  # mask (odd, odd)

    def forward(self, x):
        return nn.functional.conv2d(
            x, self.weight * self.mask, self.bias, 
            self.stride, self.padding, self.dilation, self.groups
        )