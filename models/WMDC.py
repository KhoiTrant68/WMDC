import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import Tensor
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_, DropPath

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

# Use your existing modules
from modules.dictionary_blocks import MultiScaleDictionaryCrossAttentionGLU
from modules.utils import CheckboardMaskedConv2d, conv, deconv, ste_round, conv1x1, conv3x3
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import (
    DWT_2D,
    IDWT_2D,
    ResidualBlockUpsample_wave,
    ResidualBlockWithStride_wave,
)


# ==============================================================================
# SWIN ATTENTION MODULES (Robust channel context)
# ==============================================================================

class WMSA(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, type):
        super(WMSA, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.n_heads = input_dim // head_dim
        self.window_size = window_size
        self.type = type
        self.embedding_layer = nn.Linear(self.input_dim, 3 * self.input_dim, bias=True)
        self.relative_position_params = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), self.n_heads)
        )
        self.linear = nn.Linear(self.input_dim, self.output_dim)

        trunc_normal_(self.relative_position_params, std=0.02)
        self.relative_position_params = torch.nn.Parameter(
            self.relative_position_params.view(
                2 * window_size - 1, 2 * window_size - 1, self.n_heads
            )
            .transpose(1, 2)
            .transpose(0, 1)
        )

    def generate_mask(self, h, w, p, shift):
        attn_mask = torch.zeros(
            h,
            w,
            p,
            p,
            p,
            p,
            dtype=torch.bool,
            device=self.relative_position_params.device,
        )
        if self.type == "W":
            return attn_mask
        s = p - shift
        attn_mask[-1, :, :s, :, s:, :] = True
        attn_mask[-1, :, s:, :, :s, :] = True
        attn_mask[:, -1, :, :s, :, s:] = True
        attn_mask[:, -1, :, s:, :, :s] = True
        attn_mask = rearrange(
            attn_mask, "w1 w2 p1 p2 p3 p4 -> 1 1 (w1 w2) (p1 p2) (p3 p4)"
        )
        return attn_mask

    def forward(self, x):
        if self.type != "W":
            x = torch.roll(
                x,
                shifts=(-(self.window_size // 2), -(self.window_size // 2)),
                dims=(1, 2),
            )
        x = rearrange(
            x,
            "b (w1 p1) (w2 p2) c -> b w1 w2 p1 p2 c",
            p1=self.window_size,
            p2=self.window_size,
        )
        h_windows, w_windows = x.size(1), x.size(2)
        x = rearrange(
            x,
            "b w1 w2 p1 p2 c -> b (w1 w2) (p1 p2) c",
            p1=self.window_size,
            p2=self.window_size,
        )

        qkv = self.embedding_layer(x)
        q, k, v = rearrange(
            qkv, "b nw np (threeh c) -> threeh b nw np c", c=self.head_dim
        ).chunk(3, dim=0)
        sim = torch.einsum("hbwpc,hbwqc->hbwpq", q, k) * self.scale
        sim = sim + rearrange(self.relative_embedding(), "h p q -> h 1 1 p q")

        if self.type != "W":
            attn_mask = self.generate_mask(
                h_windows, w_windows, self.window_size, shift=self.window_size // 2
            )
            sim = sim.masked_fill_(attn_mask, float("-inf"))

        probs = nn.functional.softmax(sim, dim=-1)
        output = torch.einsum("hbwij,hbwjc->hbwic", probs, v)
        output = rearrange(output, "h b w p c -> b w p (h c)")
        output = self.linear(output)
        output = rearrange(
            output,
            "b (w1 w2) (p1 p2) c -> b (w1 p1) (w2 p2) c",
            w1=h_windows,
            p1=self.window_size,
        )

        if self.type != "W":
            output = torch.roll(
                output,
                shifts=(self.window_size // 2, self.window_size // 2),
                dims=(1, 2),
            )
        return output

    def relative_embedding(self):
        cord = torch.tensor(
            np.array(
                [
                    [i, j]
                    for i in range(self.window_size)
                    for j in range(self.window_size)
                ]
            )
        )
        relation = cord[:, None, :] - cord[None, :, :] + self.window_size - 1
        return self.relative_position_params[
            :, relation[:, :, 0].long(), relation[:, :, 1].long()
        ]


class Block(nn.Module):
    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, type="W"
    ):
        super(Block, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.type = type
        self.ln1 = nn.LayerNorm(input_dim)
        self.msa = WMSA(input_dim, input_dim, head_dim, window_size, self.type)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.ln2 = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.GELU(),
            nn.Linear(4 * input_dim, output_dim),
        )

    def forward(self, x):
        x = x + self.drop_path(self.msa(self.ln1(x)))
        x = x + self.drop_path(self.mlp(self.ln2(x)))
        return x


class SwinBlock(nn.Module):
    def __init__(self, input_dim, output_dim, head_dim, window_size, drop_path):
        super().__init__()
        self.block_1 = Block(
            input_dim, output_dim, head_dim, window_size, drop_path, type="W"
        )
        self.block_2 = Block(
            input_dim, output_dim, head_dim, window_size, drop_path, type="SW"
        )
        self.window_size = window_size

    def forward(self, x):
        resize = False
        if (x.size(-1) <= self.window_size) or (x.size(-2) <= self.window_size):
            padding_row = (self.window_size - x.size(-2)) // 2
            padding_col = (self.window_size - x.size(-1)) // 2
            x = F.pad(x, (padding_col, padding_col + 1, padding_row, padding_row + 1))
            resize = True
        trans_x = Rearrange("b c h w -> b h w c")(x)
        trans_x = self.block_1(trans_x)
        trans_x = self.block_2(trans_x)
        trans_x = Rearrange("b h w c -> b c h w")(trans_x)
        if resize:
            x = F.pad(
                x, (-padding_col, -padding_col - 1, -padding_row, -padding_row - 1)
            )
        return trans_x


class SWAtten(nn.Module):
    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192
    ):
        super().__init__()
        self.non_local_block = SwinBlock(
            inter_dim, inter_dim, head_dim, window_size, drop_path
        )
        self.in_conv = conv1x1(input_dim, inter_dim)
        self.out_conv = conv1x1(inter_dim, output_dim)
        self.conv_a = nn.Sequential(
            conv3x3(inter_dim, inter_dim), nn.GELU(), conv3x3(inter_dim, inter_dim)
        )
        self.conv_b = nn.Sequential(
            conv3x3(inter_dim, inter_dim), nn.GELU(), conv3x3(inter_dim, inter_dim)
        )

    def forward(self, x):
        x = self.in_conv(x)
        identity = x
        z = self.non_local_block(x)
        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity
        out = self.out_conv(out)
        return out





# ==============================================================================
# MAIN MODEL
# ==============================================================================


class WMDC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, dict_num=128, dict_head_num=20):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.window_size = 8

        self.slice_ch_lf = M // num_slices
        self.slice_ch_hf = (3 * M) // num_slices
        self.slice_ch_dir = self.slice_ch_hf // 3

        # A. MAIN ENCODER
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        # C. HYPER-PRIOR AUTOENCODER
        self.h_a = nn.Sequential(
            conv(4 * M, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.1, ssm_d_state=16),
            conv(N, 192, kernel_size=5, stride=2),
        )
        self.entropy_bottleneck = EntropyBottleneck(192)

        self.h_mean_s = nn.Sequential(
            deconv(192, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.1, ssm_d_state=16),
            deconv(N, 4 * M, kernel_size=5, stride=2),
        )
        self.h_scale_s = nn.Sequential(
            deconv(192, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.1, ssm_d_state=16),
            deconv(N, 4 * M, kernel_size=5, stride=2),
        )

        # D. PATH A: LOW FREQUENCY ENTROPY MODEL
        self.atten_mean_lf = nn.ModuleList(
            nn.Sequential(
                SWAtten(
                    M + self.slice_ch_lf * min(i, 5),
                    M + self.slice_ch_lf * min(i, 5),
                    16,
                    self.window_size,
                    0,
                    inter_dim=128,
                )
            )
            for i in range(self.num_slices)
        )
        self.atten_scale_lf = nn.ModuleList(
            nn.Sequential(
                SWAtten(
                    M + self.slice_ch_lf * min(i, 5),
                    M + self.slice_ch_lf * min(i, 5),
                    16,
                    self.window_size,
                    0,
                    inter_dim=128,
                )
            )
            for i in range(self.num_slices)
        )

        self.cc_mean_transforms_lf = nn.ModuleList(
            nn.Sequential(
                conv(M + self.slice_ch_lf * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.slice_ch_lf, stride=1, kernel_size=3),
            )
            for i in range(self.num_slices)
        )
        self.cc_scale_transforms_lf = nn.ModuleList(
            nn.Sequential(
                conv(M + self.slice_ch_lf * min(i, 5), 224, stride=1, kernel_size=3),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.slice_ch_lf, stride=1, kernel_size=3),
            )
            for i in range(self.num_slices)
        )

        self.anchor_atten_mean_lf = nn.ModuleList(
            nn.Sequential(
                SWAtten(
                    self.slice_ch_lf,
                    self.slice_ch_lf,
                    16,
                    self.window_size,
                    0,
                    inter_dim=128,
                )
            )
            for _ in range(self.num_slices)
        )
        self.anchor_atten_scale_lf = nn.ModuleList(
            nn.Sequential(
                SWAtten(
                    self.slice_ch_lf,
                    self.slice_ch_lf,
                    16,
                    self.window_size,
                    0,
                    inter_dim=128,
                )
            )
            for _ in range(self.num_slices)
        )

        self.context_vss_lf = nn.ModuleList(
            nn.Sequential(
                VSSBlock(
                    hidden_dim=(
                        2 * M + 2 * self.slice_ch_lf
                        if i == 0
                        else 2 * M + 4 * self.slice_ch_lf
                    ),
                    drop_path=0.0,
                    ssm_d_state=16,
                ),
                conv(
                    (
                        2 * M + 2 * self.slice_ch_lf
                        if i == 0
                        else 2 * M + 4 * self.slice_ch_lf
                    ),
                    2 * self.slice_ch_lf,
                    kernel_size=3,
                    stride=1,
                ),
            )
            for i in range(self.num_slices)
        )

        self.context_prediction_lf = nn.ModuleList(
            CheckboardMaskedConv2d(
                self.slice_ch_lf,
                2 * self.slice_ch_lf,
                kernel_size=5,
                padding=2,
                stride=1,
            )
            for _ in range(self.num_slices)
        )

        self.lrp_transforms_lf = nn.ModuleList(
            nn.Sequential(
                conv(
                    M + self.slice_ch_lf * min(i + 1, 6), 224, stride=1, kernel_size=3
                ),
                nn.GELU(),
                conv(224, 128, stride=1, kernel_size=3),
                nn.GELU(),
                conv(128, self.slice_ch_lf, stride=1, kernel_size=3),
            )
            for i in range(self.num_slices)
        )
        self.gaussian_conditional_lf = GaussianConditional(None)

        # E. PATH B: HIGH FREQUENCY ENTROPY MODEL
        self.dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(
            torch.randn([dict_num, self.dict_dim]), requires_grad=True
        )
        nn.init.orthogonal_(self.dt)

        for dir_name in ["lh", "hl", "hh"]:
            query_dim = lambda i: 3 * M + (self.slice_ch_dir * i)
            setattr(
                self,
                f"dt_attn_{dir_name}",
                nn.ModuleList(
                    MultiScaleDictionaryCrossAttentionGLU(
                        input_dim=query_dim(i),
                        output_dim=M,
                        head_num=dict_head_num,
                        mlp_rate=4,
                    )
                    for i in range(self.num_slices)
                ),
            )
            setattr(
                self,
                f"cc_transforms_{dir_name}",
                nn.ModuleList(
                    nn.Sequential(
                        conv(query_dim(i) + M, 512, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(512, 256, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(256, 2 * self.slice_ch_dir, stride=1, kernel_size=3),
                    )
                    for i in range(self.num_slices)
                ),
            )
            setattr(
                self,
                f"lrp_transforms_{dir_name}",
                nn.ModuleList(
                    nn.Sequential(
                        conv(
                            (query_dim(i) + M) + self.slice_ch_dir,
                            512,
                            stride=1,
                            kernel_size=3,
                        ),
                        nn.GELU(),
                        conv(512, 256, stride=1, kernel_size=3),
                        nn.GELU(),
                        conv(256, self.slice_ch_dir, stride=1, kernel_size=3),
                    )
                    for i in range(self.num_slices)
                ),
            )
        self.gaussian_conditional_hf = GaussianConditional(None)

        # F. MAIN DECODER
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def forward(self, x):
        B, C, H, W = x.size()

        y = self.g_a(x)
        y_shape = y.shape[2:]

        y_freq = self.dwt(y)
        y_low = y_freq[:, : self.M, :, :]
        y_high = y_freq[:, self.M :, :, :]

        z = self.h_a(y_freq)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians().detach()
        z_hat = ste_round(z - z_offset) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        latent_scales_lf, latent_scales_hf = (
            latent_scales[:, : self.M],
            latent_scales[:, self.M :],
        )
        latent_means_lf, latent_means_hf = (
            latent_means[:, : self.M],
            latent_means[:, self.M :],
        )

        # ----------------------------------------------------------------------
        # PATH A: LOW FREQUENCY
        # ----------------------------------------------------------------------
        anchor = torch.zeros_like(y_low).to(x.device)
        non_anchor = torch.zeros_like(y_low).to(x.device)
        anchor[:, :, 0::2, 0::2], anchor[:, :, 1::2, 1::2] = (
            y_low[:, :, 0::2, 0::2],
            y_low[:, :, 1::2, 1::2],
        )
        non_anchor[:, :, 0::2, 1::2], non_anchor[:, :, 1::2, 0::2] = (
            y_low[:, :, 0::2, 1::2],
            y_low[:, :, 1::2, 0::2],
        )

        y_low_slices = y_low.chunk(self.num_slices, 1)
        anchor_split = anchor.chunk(self.num_slices, 1)
        non_anchor_split = non_anchor.chunk(self.num_slices, 1)

        ctx_params_anchor_split = torch.split(
            torch.zeros(
                B,
                self.slice_ch_lf * 2 * self.num_slices,
                y_shape[0] // 2,
                y_shape[1] // 2,
            ).to(x.device),
            [2 * self.slice_ch_lf for _ in range(self.num_slices)],
            1,
        )

        y_low_hat_slices, y_low_likelihood = [], []

        for i, y_slice in enumerate(y_low_slices):
            mean_support = torch.cat([latent_means_lf] + y_low_hat_slices, dim=1)
            mean_support = self.atten_mean_lf[i](mean_support)
            mu = self.cc_mean_transforms_lf[i](mean_support)[
                :, :, : y_shape[0] // 2, : y_shape[1] // 2
            ]

            scale_support = torch.cat([latent_scales_lf] + y_low_hat_slices, dim=1)
            scale_support = self.atten_scale_lf[i](scale_support)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, : y_shape[0] // 2, : y_shape[1] // 2
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            y_anchor = anchor_split[i]
            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([ctx_params_anchor_split[i], support], dim=1)
            ).chunk(2, 1)
            means_anchor = self.anchor_atten_mean_lf[i](means_anchor)
            scales_anchor = self.anchor_atten_scale_lf[i](scales_anchor)

            scales_hat_split = torch.zeros_like(y_anchor)
            means_hat_split = torch.zeros_like(y_anchor)
            scales_hat_split[:, :, 0::2, 0::2], scales_hat_split[:, :, 1::2, 1::2] = (
                scales_anchor[:, :, 0::2, 0::2],
                scales_anchor[:, :, 1::2, 1::2],
            )
            means_hat_split[:, :, 0::2, 0::2], means_hat_split[:, :, 1::2, 1::2] = (
                means_anchor[:, :, 0::2, 0::2],
                means_anchor[:, :, 1::2, 1::2],
            )

            y_anchor_quantized = ste_round(y_anchor - means_anchor) + means_anchor
            y_anchor_quantized[:, :, 0::2, 1::2] = 0
            y_anchor_quantized[:, :, 1::2, 0::2] = 0

            masked_context = self.context_prediction_lf[i](y_anchor_quantized)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale_lf[i](scales_non_anchor)

            scales_hat_split[:, :, 0::2, 1::2], scales_hat_split[:, :, 1::2, 0::2] = (
                scales_non_anchor[:, :, 0::2, 1::2],
                scales_non_anchor[:, :, 1::2, 0::2],
            )
            means_hat_split[:, :, 0::2, 1::2], means_hat_split[:, :, 1::2, 0::2] = (
                means_non_anchor[:, :, 0::2, 1::2],
                means_non_anchor[:, :, 1::2, 0::2],
            )

            y_non_anchor = non_anchor_split[i]
            y_non_anchor_quantized = (
                ste_round(y_non_anchor - means_non_anchor) + means_non_anchor
            )
            y_non_anchor_quantized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantized[:, :, 1::2, 1::2] = 0

            _, y_slice_likelihood_val = self.gaussian_conditional_lf(
                y_slice, scales_hat_split, means=means_hat_split
            )
            y_low_likelihood.append(y_slice_likelihood_val)

            y_hat_slice = y_anchor_quantized + y_non_anchor_quantized
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp
            y_low_hat_slices.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)
        y_low_likelihoods = torch.cat(y_low_likelihood, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY
        # ----------------------------------------------------------------------
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_lh, y_hl, y_hh = y_high.chunk(3, dim=1)
        y_lh_slices, y_hl_slices, y_hh_slices = (
            y_lh.chunk(self.num_slices, dim=1),
            y_hl.chunk(self.num_slices, dim=1),
            y_hh.chunk(self.num_slices, dim=1),
        )

        y_high_hat_slices, y_high_likelihood = [], []
        lh_hats, hl_hats, hh_hats = [], [], []

        for i in range(self.num_slices):
            y_slice = torch.cat([y_lh_slices[i], y_hl_slices[i], y_hh_slices[i]], dim=1)

            q_lh = torch.cat([y_low_hat, scales_lh, means_lh] + lh_hats, dim=1)
            q_hl = torch.cat([y_low_hat, scales_hl, means_hl] + hl_hats, dim=1)
            q_hh = torch.cat([y_low_hat, scales_hh, means_hh] + hh_hats, dim=1)

            dict_lh = self.dt_attn_lh[i](q_lh, self.dt)
            dict_hl = self.dt_attn_hl[i](q_hl, self.dt)
            dict_hh = self.dt_attn_hh[i](q_hh, self.dt)

            sup_lh = torch.cat([q_lh, dict_lh], dim=1)
            sup_hl = torch.cat([q_hl, dict_hl], dim=1)
            sup_hh = torch.cat([q_hh, dict_hh], dim=1)

            mu_lh, sc_lh = self.cc_transforms_lh[i](sup_lh).chunk(2, 1)
            mu_hl, sc_hl = self.cc_transforms_hl[i](sup_hl).chunk(2, 1)
            mu_hh, sc_hh = self.cc_transforms_hh[i](sup_hh).chunk(2, 1)

            mu = torch.cat([mu_lh, mu_hl, mu_hh], dim=1)
            scale = torch.cat([sc_lh, sc_hl, sc_hh], dim=1)

            _, y_slice_like = self.gaussian_conditional_hf(y_slice, scale, means=mu)
            y_high_likelihood.append(y_slice_like)

            # Apply clean STE
            y_hat_slice = ste_round(y_slice - mu) + mu

            y_hat_lh, y_hat_hl, y_hat_hh = y_hat_slice.chunk(3, dim=1)

            y_hat_lh = y_hat_lh + 0.5 * torch.tanh(
                self.lrp_transforms_lh[i](torch.cat([sup_lh, y_hat_lh], dim=1))
            )
            y_hat_hl = y_hat_hl + 0.5 * torch.tanh(
                self.lrp_transforms_hl[i](torch.cat([sup_hl, y_hat_hl], dim=1))
            )
            y_hat_hh = y_hat_hh + 0.5 * torch.tanh(
                self.lrp_transforms_hh[i](torch.cat([sup_hh, y_hat_hh], dim=1))
            )

            y_hat_slice = torch.cat([y_hat_lh, y_hat_hl, y_hat_hh], dim=1)
            lh_hats.append(y_hat_lh)
            hl_hats.append(y_hat_hl)
            hh_hats.append(y_hat_hh)
            y_high_hat_slices.append(y_hat_slice)

        y_high_hat = torch.cat(
            [
                torch.cat(lh_hats, dim=1),
                torch.cat(hl_hats, dim=1),
                torch.cat(hh_hats, dim=1),
            ],
            dim=1,
        )
        y_high_likelihoods = torch.cat(y_high_likelihood, dim=1)

        y_freq_hat = torch.cat([y_low_hat, y_high_hat], dim=1)
        y_tilde = self.idwt(y_freq_hat)
        x_hat = self.g_s(y_tilde)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y_low": y_low_likelihoods,
                "y_high": y_high_likelihoods,
                "z": z_likelihoods,
            },
            "aux_loss": self.aux_loss(),
        }

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        updated = self.gaussian_conditional_lf.update_scale_table(
            scale_table, force=force
        )
        updated |= self.gaussian_conditional_hf.update_scale_table(
            scale_table, force=force
        )
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        B, C, H, W = x.size()
        y = self.g_a(x)
        y_freq = self.dwt(y)
        H_wave, W_wave = y_freq.shape[2:]
        y_low = y_freq[:, : self.M, :, :]
        y_high = y_freq[:, self.M :, :, :]

        z = self.h_a(y_freq)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        latent_scales_lf, latent_scales_hf = (
            latent_scales[:, : self.M],
            latent_scales[:, self.M :],
        )
        latent_means_lf, latent_means_hf = (
            latent_means[:, : self.M],
            latent_means[:, self.M :],
        )

        y_low_slices = y_low.chunk(self.num_slices, 1)
        y_low_hat_slices, y_low_strings = [], []

        ctx_params_anchor_split = torch.split(
            torch.zeros(B, self.slice_ch_lf * 2 * self.num_slices, H_wave, W_wave).to(
                x.device
            ),
            [2 * self.slice_ch_lf for _ in range(self.num_slices)],
            1,
        )

        for i, y_slice in enumerate(y_low_slices):
            mean_support = torch.cat([latent_means_lf] + y_low_hat_slices, dim=1)
            mean_support = self.atten_mean_lf[i](mean_support)
            mu = self.cc_mean_transforms_lf[i](mean_support)[:, :, :H_wave, :W_wave]

            scale_support = torch.cat([latent_scales_lf] + y_low_hat_slices, dim=1)
            scale_support = self.atten_scale_lf[i](scale_support)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, :H_wave, :W_wave
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            y_anchor = y_slice.clone()
            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([ctx_params_anchor_split[i], support], dim=1)
            ).chunk(2, 1)
            means_anchor = self.anchor_atten_mean_lf[i](means_anchor)
            scales_anchor = self.anchor_atten_scale_lf[i](scales_anchor)

            B_anchor, C_anchor, H_anchor, W_anchor = y_anchor.size()
            y_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(x.device)
            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(x.device)
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(x.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                x.device
            )

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_anchor_encode
            )
            anchor_strings = self.gaussian_conditional_lf.compress(
                y_anchor_encode, indexes_anchor, means=means_anchor_encode
            )
            anchor_quantized = self.gaussian_conditional_lf.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            masked_context = self.context_prediction_lf[i](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale_lf[i](scales_non_anchor)

            y_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(x.device)
            means_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(x.device)
            scales_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(x.device)

            y_non_anchor_encode[:, :, 0::2, :] = y_slice[:, :, 0::2, 1::2]
            y_non_anchor_encode[:, :, 1::2, :] = y_slice[:, :, 1::2, 0::2]
            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = self.gaussian_conditional_lf.compress(
                y_non_anchor_encode, indexes_non_anchor, means=means_non_anchor_encode
            )
            non_anchor_quantized = self.gaussian_conditional_lf.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            y_hat_slice = y_anchor_decode + y_non_anchor_quantized
            y_hat_slice += 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )

            y_low_hat_slices.append(y_hat_slice)
            y_low_strings.append([anchor_strings, non_anchor_strings])

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY
        # ----------------------------------------------------------------------
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_lh, y_hl, y_hh = y_high.chunk(3, dim=1)
        y_lh_slices, y_hl_slices, y_hh_slices = (
            y_lh.chunk(self.num_slices, dim=1),
            y_hl.chunk(self.num_slices, dim=1),
            y_hh.chunk(self.num_slices, dim=1),
        )

        lh_hats, hl_hats, hh_hats = [], [], []
        y_high_strings = []

        for i in range(self.num_slices):
            y_slice = torch.cat([y_lh_slices[i], y_hl_slices[i], y_hh_slices[i]], dim=1)

            q_lh = torch.cat([y_low_hat, scales_lh, means_lh] + lh_hats, dim=1)
            q_hl = torch.cat([y_low_hat, scales_hl, means_hl] + hl_hats, dim=1)
            q_hh = torch.cat([y_low_hat, scales_hh, means_hh] + hh_hats, dim=1)

            dict_lh, dict_hl, dict_hh = (
                self.dt_attn_lh[i](q_lh, self.dt),
                self.dt_attn_hl[i](q_hl, self.dt),
                self.dt_attn_hh[i](q_hh, self.dt),
            )
            sup_lh, sup_hl, sup_hh = (
                torch.cat([q_lh, dict_lh], dim=1),
                torch.cat([q_hl, dict_hl], dim=1),
                torch.cat([q_hh, dict_hh], dim=1),
            )

            mu_lh, sc_lh = self.cc_transforms_lh[i](sup_lh).chunk(2, 1)
            mu_hl, sc_hl = self.cc_transforms_hl[i](sup_hl).chunk(2, 1)
            mu_hh, sc_hh = self.cc_transforms_hh[i](sup_hh).chunk(2, 1)

            mu = torch.cat([mu_lh, mu_hl, mu_hh], dim=1)
            scale = torch.cat([sc_lh, sc_hl, sc_hh], dim=1)

            indexes = self.gaussian_conditional_hf.build_indexes(scale)
            y_hf_strings = self.gaussian_conditional_hf.compress(
                y_slice, indexes, means=mu
            )
            y_hat_slice = self.gaussian_conditional_hf.decompress(
                y_hf_strings, indexes, means=mu
            )

            y_hat_lh, y_hat_hl, y_hat_hh = y_hat_slice.chunk(3, dim=1)

            y_hat_lh = y_hat_lh + 0.5 * torch.tanh(
                self.lrp_transforms_lh[i](torch.cat([sup_lh, y_hat_lh], dim=1))
            )
            y_hat_hl = y_hat_hl + 0.5 * torch.tanh(
                self.lrp_transforms_hl[i](torch.cat([sup_hl, y_hat_hl], dim=1))
            )
            y_hat_hh = y_hat_hh + 0.5 * torch.tanh(
                self.lrp_transforms_hh[i](torch.cat([sup_hh, y_hat_hh], dim=1))
            )

            lh_hats.append(y_hat_lh)
            hl_hats.append(y_hat_hl)
            hh_hats.append(y_hat_hh)
            y_high_strings.append(y_hf_strings)

        return {
            "strings": [[y_low_strings, y_high_strings], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        y_low_strings, y_high_strings = strings[0][0], strings[0][1]

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        B = z_hat.size(0)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        latent_scales_lf, latent_scales_hf = (
            latent_scales[:, : self.M],
            latent_scales[:, self.M :],
        )
        latent_means_lf, latent_means_hf = (
            latent_means[:, : self.M],
            latent_means[:, self.M :],
        )
        H_wave, W_wave = latent_scales.shape[2:]

        y_low_hat_slices = []
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, self.slice_ch_lf * 2 * self.num_slices, H_wave, W_wave).to(
                z_hat.device
            ),
            [2 * self.slice_ch_lf for _ in range(self.num_slices)],
            1,
        )

        for i in range(self.num_slices):
            mean_support = torch.cat([latent_means_lf] + y_low_hat_slices, dim=1)
            mean_support = self.atten_mean_lf[i](mean_support)
            mu = self.cc_mean_transforms_lf[i](mean_support)[:, :, :H_wave, :W_wave]

            scale_support = torch.cat([latent_scales_lf] + y_low_hat_slices, dim=1)
            scale_support = self.atten_scale_lf[i](scale_support)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, :H_wave, :W_wave
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([ctx_params_anchor_split[i], support], dim=1)
            ).chunk(2, 1)
            means_anchor = self.anchor_atten_mean_lf[i](means_anchor)
            scales_anchor = self.anchor_atten_scale_lf[i](scales_anchor)

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()
            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                z_hat.device
            )

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_anchor_encode
            )
            anchor_strings = y_low_strings[i][0]
            anchor_quantized = self.gaussian_conditional_lf.decompress(
                anchor_strings, indexes_anchor, means=means_anchor_encode
            )

            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            masked_context = self.context_prediction_lf[i](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale_lf[i](scales_non_anchor)

            means_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_non_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            indexes_non_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = y_low_strings[i][1]
            non_anchor_quantized = self.gaussian_conditional_lf.decompress(
                non_anchor_strings, indexes_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            y_hat_slice = y_anchor_decode + y_non_anchor_quantized
            y_hat_slice += 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_low_hat_slices.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)
        lh_hats, hl_hats, hh_hats = [], [], []

        for i in range(self.num_slices):
            q_lh = torch.cat([y_low_hat, scales_lh, means_lh] + lh_hats, dim=1)
            q_hl = torch.cat([y_low_hat, scales_hl, means_hl] + hl_hats, dim=1)
            q_hh = torch.cat([y_low_hat, scales_hh, means_hh] + hh_hats, dim=1)

            dict_lh, dict_hl, dict_hh = (
                self.dt_attn_lh[i](q_lh, self.dt),
                self.dt_attn_hl[i](q_hl, self.dt),
                self.dt_attn_hh[i](q_hh, self.dt),
            )
            sup_lh, sup_hl, sup_hh = (
                torch.cat([q_lh, dict_lh], dim=1),
                torch.cat([q_hl, dict_hl], dim=1),
                torch.cat([q_hh, dict_hh], dim=1),
            )

            mu_lh, sc_lh = self.cc_transforms_lh[i](sup_lh).chunk(2, 1)
            mu_hl, sc_hl = self.cc_transforms_hl[i](sup_hl).chunk(2, 1)
            mu_hh, sc_hh = self.cc_transforms_hh[i](sup_hh).chunk(2, 1)

            mu = torch.cat([mu_lh, mu_hl, mu_hh], dim=1)
            scale = torch.cat([sc_lh, sc_hl, sc_hh], dim=1)

            indexes = self.gaussian_conditional_hf.build_indexes(scale)
            y_hf_strings = y_high_strings[i]
            y_hat_slice = self.gaussian_conditional_hf.decompress(
                y_hf_strings, indexes, means=mu
            )

            y_hat_lh, y_hat_hl, y_hat_hh = y_hat_slice.chunk(3, dim=1)

            y_hat_lh = y_hat_lh + 0.5 * torch.tanh(
                self.lrp_transforms_lh[i](torch.cat([sup_lh, y_hat_lh], dim=1))
            )
            y_hat_hl = y_hat_hl + 0.5 * torch.tanh(
                self.lrp_transforms_hl[i](torch.cat([sup_hl, y_hat_hl], dim=1))
            )
            y_hat_hh = y_hat_hh + 0.5 * torch.tanh(
                self.lrp_transforms_hh[i](torch.cat([sup_hh, y_hat_hh], dim=1))
            )

            lh_hats.append(y_hat_lh)
            hl_hats.append(y_hat_hl)
            hh_hats.append(y_hat_hh)

        y_high_hat = torch.cat(
            [
                torch.cat(lh_hats, dim=1),
                torch.cat(hl_hats, dim=1),
                torch.cat(hh_hats, dim=1),
            ],
            dim=1,
        )

        y_freq_hat = torch.cat([y_low_hat, y_high_hat], dim=1)
        y_tilde = self.idwt(y_freq_hat)
        x_hat = self.g_s(y_tilde).clamp_(0, 1)

        return {"x_hat": x_hat}
