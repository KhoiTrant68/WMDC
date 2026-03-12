import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from modules.VSS_module import VSSBlock
from modules.utils import conv, deconv
from modules.wavelet_blocks import FrequencyDisentangledMamba
from modules.dictionary_blocks import MultiScaleDictionaryCrossAttentionGLU


class WMDC(CompressionModel):
    """
    Wavelet-Guided State Space Model with Dynamic Dictionary Attention
    Novelties:
    1. FD-SSM: Frequency-Disentangled Mamba (Routes LL to Mamba, HF to CNNs via DWT)
    2. HDDA: Hyper-Conditioned Dynamic Dictionary Attention (0-bit global context adaptive generation)
    3. Mamba-Assisted Linear Channel Autoregression
    """

    def __init__(
        self,
        N=192,
        M=320,
        num_slices=5,
        max_support_slices=5,
        dict_head_num=20,
        dict_num=128,
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.max_support_slices = max_support_slices
        self.slice_ch = M // num_slices

        self.dict_num = dict_num
        self.dict_dim = 32 * dict_head_num

        # ==========================================
        # 1. MAIN ENCODER & DECODER (FD-SSM)
        # ==========================================
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            conv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            conv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            deconv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            deconv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        # ==========================================
        # 2. HYPER-PRIOR AUTOENCODER
        # ==========================================
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
            conv(N, 192, kernel_size=5, stride=2),
        )

        self.h_scale_s = nn.Sequential(
            deconv(192, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
            deconv(N, M, kernel_size=5, stride=2),
        )

        self.h_mean_s = nn.Sequential(
            deconv(192, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
            deconv(N, M, kernel_size=5, stride=2),
        )

        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

        # ==========================================
        # 3. NOVELTY: HYPER-CONDITIONED DYNAMIC DICTIONARY (HDDA)
        # Generates image-specific global context tokens from z_hat (0 bits overhead)
        # ==========================================
        self.hyper_to_dict = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(192, 512),
            nn.GELU(),
            nn.Linear(512, self.dict_num * self.dict_dim),
        )

        self.dt_cross_attention = nn.ModuleList(
            MultiScaleDictionaryCrossAttentionGLU(
                input_dim=2 * M + self.slice_ch * min(i, self.max_support_slices),
                output_dim=M,
                head_num=dict_head_num,
                mlp_rate=4,
                qkv_bias=True,
            )
            for i in range(num_slices)
        )

        # ==========================================
        # 4. CHANNEL AUTOREGRESSIVE TRANSFORMS (ChARM)
        # ==========================================
        self.cc_mean_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    3 * M + self.slice_ch * min(i, self.max_support_slices),
                    224,
                    3,
                    1,
                    1,
                ),
                nn.GELU(),
                nn.Conv2d(224, 128, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(128, self.slice_ch, 3, 1, 1),
            )
            for i in range(num_slices)
        )

        self.cc_scale_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    3 * M + self.slice_ch * min(i, self.max_support_slices),
                    224,
                    3,
                    1,
                    1,
                ),
                nn.GELU(),
                nn.Conv2d(224, 128, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(128, self.slice_ch, 3, 1, 1),
            )
            for i in range(num_slices)
        )

        self.lrp_transforms = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    3 * M + self.slice_ch * min(i + 1, self.max_support_slices + 1),
                    224,
                    3,
                    1,
                    1,
                ),
                nn.GELU(),
                nn.Conv2d(224, 128, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(128, self.slice_ch, 3, 1, 1),
            )
            for i in range(num_slices)
        )

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def forward(self, x):
        b = x.size(0)

        y = self.g_a(x)
        z = self.h_a(y)

        # Quantize hyper-prior
        z_hat, z_likelihoods = self.entropy_bottleneck(z)

        # HDDA: Generate Dynamic Dictionary from z_hat
        dt_flat = self.hyper_to_dict(z_hat)
        dt = dt_flat.view(b, self.dict_num, self.dict_dim)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        for i, y_slice in enumerate(y_slices):
            # Safe Channel-wise Context Gathering
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[-self.max_support_slices :]
            )
            query = torch.cat([hyper_prior] + support_slices, dim=1)

            # Linear Dictionary Global Context
            dict_info = self.dt_cross_attention[i](query, dt)
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support)

            # Mathematically safe clamping bounds
            scale = torch.clamp(scale, min=0.11)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, means=mu)
            y_likelihood.append(y_slice_likelihood)

            # STE Quantization
            y_hat_slice = (
                (torch.round(y_slice - mu) - (y_slice - mu)).detach()
                + (y_slice - mu)
                + mu
            )

            # Latent Residual Prediction
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms[i](lrp_support))
            y_hat_slice = y_hat_slice + lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
            "aux_loss": self.aux_loss(),
        }

    def compress(self, x):
        b = x.size(0)

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        # HDDA
        dt_flat = self.hyper_to_dict(z_hat)
        dt = dt_flat.view(b, self.dict_num, self.dict_dim)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_strings = []

        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        for i, y_slice in enumerate(y_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[-self.max_support_slices :]
            )
            query = torch.cat([hyper_prior] + support_slices, dim=1)

            dict_info = self.dt_cross_attention[i](query, dt)
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support)
            scale = torch.clamp(scale, min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, index, means=mu)
            y_strings.append(y_string)

            y_hat_slice = self.gaussian_conditional.decompress(
                y_string, index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms[i](lrp_support))
            y_hat_slice = y_hat_slice + lrp

            y_hat_slices.append(y_hat_slice)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        y_strings, z_strings = strings[0], strings[1]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        b = z_hat.size(0)
        # HDDA
        dt_flat = self.hyper_to_dict(z_hat)
        dt = dt_flat.view(b, self.dict_num, self.dict_dim)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_hat_slices = []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        for i in range(self.num_slices):
            support_slices = (
                y_hat_slices
                if self.max_support_slices < 0
                else y_hat_slices[-self.max_support_slices :]
            )
            query = torch.cat([hyper_prior] + support_slices, dim=1)

            dict_info = self.dt_cross_attention[i](query, dt)
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support)
            scale = torch.clamp(scale, min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)

            y_hat_slice = self.gaussian_conditional.decompress(
                y_strings[i], index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms[i](lrp_support))
            y_hat_slice = y_hat_slice + lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
