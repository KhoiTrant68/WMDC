import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from modules.dictionary_blocks import (
    QueryDictionaryGenerator,
    SpatialDispersionEOTAttention,
)
from modules.utils import conv, deconv
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import FrequencyDisentangledMamba


class WMDC(CompressionModel):
    def __init__(
        self,
        N=192,
        M=320,
        num_slices=5,
        dict_head_num=20,
        dict_num=128,
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.slice_ch = M // num_slices
        self.dict_num = dict_num
        self.dict_dim = 32 * dict_head_num

        # --- Encoders / Decoders ---
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

        self.hyper_to_dict = QueryDictionaryGenerator(
            in_dim=192, dict_num=self.dict_num, dict_dim=self.dict_dim, num_heads=4
        )

        # =========================================================================
        # Spatially-Adaptive Sinkhorn Entropy Predictor
        # =========================================================================
        self.eps_predictor = nn.Sequential(
            nn.Conv2d(2 * M, 64, 1),
            nn.GELU(),
            nn.Conv2d(64, 1, 1),  # Predicts 1 channel for spatial epsilon
        )

        # =========================================================================
        # SHARED Markovian Architecture
        # =========================================================================
        #: Bootstrap memory state dynamically to prevent Slice 1 conditional blindness
        self.init_memory = nn.Conv2d(2 * M, self.slice_ch, kernel_size=1)

        #: Projections moved here to compute them ONCE globally instead of 5 times inside EOT
        self.k_proj = nn.Linear(self.dict_dim, self.dict_dim)
        self.v_proj = nn.Linear(self.dict_dim, self.dict_dim)

        self.eot_attention = SpatialDispersionEOTAttention(
            input_dim=2 * M
            + self.slice_ch,  # hyper_prior (2M) + memory_state (slice_ch)
            output_dim=M,
            dict_num=self.dict_num,
            dict_dim=self.dict_dim,
            tau=0.5,
            iters=3,
        )

        self.memory_updater = nn.Sequential(
            nn.Conv2d(self.slice_ch * 2, self.slice_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.slice_ch, self.slice_ch, kernel_size=3, padding=1),
        )

        shared_input_dim = 3 * M + self.slice_ch  # query + dict_info

        # Single shared transforms for all slices!
        self.cc_mean_transform = nn.Sequential(
            nn.Conv2d(shared_input_dim, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, self.slice_ch, 3, 1, 1),
        )

        self.cc_scale_transform = nn.Sequential(
            nn.Conv2d(shared_input_dim, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, self.slice_ch, 3, 1, 1),
        )

        self.lrp_transform = nn.Sequential(
            nn.Conv2d(shared_input_dim + self.slice_ch, 128, 1),
            nn.GELU(),
            nn.Conv2d(128, 224, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(224, 128, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(128, self.slice_ch, 3, 1, 1),
        )

    def _pad_for_model(self, x, p=64):
        H, W = x.size(2), x.size(3)
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h > 0 or pad_w > 0:
            x_padded = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_padded = x
        return x_padded, (H, W)

    def _crop_to_original(self, x, original_shape):
        H, W = original_shape
        if x.size(2) > H or x.size(3) > W:
            return x[:, :, :H, :W]
        return x

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = torch.exp(torch.linspace(math.log(0.11), math.log(256), 64))
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def forward(self, x):
        x_padded, original_shape = self._pad_for_model(x, p=64)

        y = self.g_a(x_padded)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        dt = self.hyper_to_dict(z_hat)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices, y_likelihood = [], []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        total_dispersion = 0.0

        # Spatial Sinkhorn epsilon from hyperprior
        spatial_epsilon = F.softplus(self.eps_predictor(hyper_prior)) + 1e-4

        # Bootstrap Memory State
        memory_state = self.init_memory(hyper_prior)

        # Compute Dictionary Projections ONCE
        k_dict = self.k_proj(dt)
        v_dict = self.v_proj(dt)

        for i, y_slice in enumerate(y_slices):
            query = torch.cat([hyper_prior, memory_state], dim=1)

            # Updated signature passing cached k_dict and v_dict
            dict_info, disp_loss = self.eot_attention(
                query, k_dict, v_dict, spatial_epsilon
            )
            total_dispersion += disp_loss

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transform(support)
            scale = torch.clamp(self.cc_scale_transform(support), min=0.11)

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, means=mu
            )
            y_likelihood.append(y_slice_likelihood)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transform(lrp_support)
            )
            y_hat_slices.append(y_hat_slice)

            # Update Markovian Memory State
            state_input = torch.cat([memory_state, y_hat_slice], dim=1)
            memory_state = memory_state + self.memory_updater(state_input)

        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1))
        x_hat = self._crop_to_original(x_hat, original_shape)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": torch.cat(y_likelihood, dim=1), "z": z_likelihoods},
            "aux_loss": self.aux_loss(),
            "dispersion_loss": total_dispersion / self.num_slices,
        }

    def compress(self, x):
        x_padded, original_shape = self._pad_for_model(x, p=64)

        y = self.g_a(x_padded)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        dt = self.hyper_to_dict(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices, y_strings = [], []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        spatial_epsilon = F.softplus(self.eps_predictor(hyper_prior)) + 1e-4

        # Bootstrap Memory State & Cache Dict Projections
        memory_state = self.init_memory(hyper_prior)
        k_dict = self.k_proj(dt)
        v_dict = self.v_proj(dt)

        for i, y_slice in enumerate(y_slices):
            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _ = self.eot_attention(query, k_dict, v_dict, spatial_epsilon)
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transform(support)
            scale = torch.clamp(self.cc_scale_transform(support), min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, index, means=mu)
            y_strings.append(y_string)

            y_hat_slice = self.gaussian_conditional.decompress(
                y_string, index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transform(lrp_support)
            )
            y_hat_slices.append(y_hat_slice)

            state_input = torch.cat([memory_state, y_hat_slice], dim=1)
            memory_state = memory_state + self.memory_updater(state_input)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "original_shape": original_shape,
        }

    def decompress(self, strings, shape, original_shape=None):
        y_strings, z_strings = strings[0], strings[1]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        dt = self.hyper_to_dict(z_hat)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_hat_slices = []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        spatial_epsilon = F.softplus(self.eps_predictor(hyper_prior)) + 1e-4

        # Bootstrap Memory State & Cache Dict Projections
        memory_state = self.init_memory(hyper_prior)
        k_dict = self.k_proj(dt)
        v_dict = self.v_proj(dt)

        for i in range(self.num_slices):
            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _ = self.eot_attention(query, k_dict, v_dict, spatial_epsilon)
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transform(support)
            scale = torch.clamp(self.cc_scale_transform(support), min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_hat_slice = self.gaussian_conditional.decompress(
                y_strings[i], index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transform(lrp_support)
            )
            y_hat_slices.append(y_hat_slice)

            state_input = torch.cat([memory_state, y_hat_slice], dim=1)
            memory_state = memory_state + self.memory_updater(state_input)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        if original_shape is not None:
            x_hat = self._crop_to_original(x_hat, original_shape)

        return {"x_hat": x_hat}
