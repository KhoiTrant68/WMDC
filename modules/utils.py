import torch
import torch.nn.functional as F
from torch import nn


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)


def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
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


class OLP(nn.Module):
    """Orthogonal Linear Projection (CMIC-style).

    Drop-in replacement for nn.Linear with an auxiliary orthogonality
    regulariser  ||W W^T − I||²  (or  W^T W  depending on shape, picking the
    smaller Gram so the loss is well-conditioned for both fat and tall W).

    Reduces dictionary/projection collapse by discouraging correlated rows.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.in_features = in_features
        self.out_features = out_features
        eye_dim = min(in_features, out_features)
        self.register_buffer("identity_matrix", torch.eye(eye_dim), persistent=False)

    def loss(self) -> torch.Tensor:
        W = self.linear.weight  # (out, in)
        gram = W @ W.t() if self.in_features > self.out_features else W.t() @ W
        return F.mse_loss(gram, self.identity_matrix.to(gram.device, gram.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
