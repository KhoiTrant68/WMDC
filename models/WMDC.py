import math

import torch
import torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from einops import rearrange

from modules.dictionary_blocks import MultiScaleDictionaryCrossAttentionGLU
from modules.utils import CheckboardMaskedConv2d, conv, deconv
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import (
    DWT_2D,
    IDWT_2D,
    ResidualBlockUpsample_wave,
    ResidualBlockWithStride_wave,
)


class WMDC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.slice_ch_lf = M // num_slices
        self.slice_ch_hf = (3 * M) // num_slices
        self.slice_ch_dir = self.slice_ch_hf // 3

        # A. MAIN ENCODER (CNN + Exact DWT)
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

        # D. PATH A: LOW FREQUENCY ENTROPY MODEL (Mamba + Spatial Context)
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

        # Mamba captures global LF structure
        self.context_vss_lf = nn.ModuleList(
            nn.Sequential(
                VSSBlock(
                    hidden_dim=(
                        2 * M + self.slice_ch_lf
                        if i == 0
                        else 2 * M + 3 * self.slice_ch_lf
                    ),
                    drop_path=0.0,
                    ssm_d_state=16,
                ),
                conv(
                    (
                        2 * M + self.slice_ch_lf
                        if i == 0
                        else 2 * M + 3 * self.slice_ch_lf
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
                self.slice_ch_lf, self.slice_ch_lf, kernel_size=5, padding=2, stride=1
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

        # E. PATH B: HIGH FREQUENCY DICTIONARY ATTENTION
        # HF relies entirely on querying the LF dictionary.
        self.fusion_lh, self.fusion_hl, self.fusion_hh = (
            nn.ModuleList(),
            nn.ModuleList(),
            nn.ModuleList(),
        )

        for i in range(self.num_slices):
            dim_sup_lh = 3 * M + self.slice_ch_dir * i
            self.fusion_lh.append(
                MultiScaleDictionaryCrossAttentionGLU(
                    input_dim=dim_sup_lh,
                    dict_input_dim=M,
                    output_dim=2 * self.slice_ch_dir,
                )
            )

            dim_sup_hl = 3 * M + self.slice_ch_dir * (2 * i + 1)
            self.fusion_hl.append(
                MultiScaleDictionaryCrossAttentionGLU(
                    input_dim=dim_sup_hl,
                    dict_input_dim=M,
                    output_dim=2 * self.slice_ch_dir,
                )
            )

            dim_sup_hh = 3 * M + self.slice_ch_dir * (3 * i + 2)
            self.fusion_hh.append(
                MultiScaleDictionaryCrossAttentionGLU(
                    input_dim=dim_sup_hh,
                    dict_input_dim=M,
                    output_dim=2 * self.slice_ch_dir,
                )
            )

        self.gaussian_conditional_hf = GaussianConditional(None)

        # F. MAIN DECODER
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def _quantize(self, y, means=None):
        """Mathematically sound quantization substitution."""
        if self.training:
            return y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            if means is not None:
                return torch.round(y - means) + means
            return torch.round(y)

    def forward(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        y_freq = self.dwt(y)
        y_low, y_high = y_freq[:, : self.M], y_freq[:, self.M :]

        z = self.h_a(y_freq)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians().detach()
        z_hat = self._quantize(z - z_offset) + z_offset

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

        anchor, non_anchor = torch.zeros_like(y_low), torch.zeros_like(y_low)
        anchor[:, :, 0::2, 0::2], anchor[:, :, 1::2, 1::2] = (
            y_low[:, :, 0::2, 0::2],
            y_low[:, :, 1::2, 1::2],
        )
        non_anchor[:, :, 0::2, 1::2], non_anchor[:, :, 1::2, 0::2] = (
            y_low[:, :, 0::2, 1::2],
            y_low[:, :, 1::2, 0::2],
        )

        y_low_slices = y_low.chunk(self.num_slices, 1)
        anchor_split, non_anchor_split = anchor.chunk(
            self.num_slices, 1
        ), non_anchor.chunk(self.num_slices, 1)

        y_low_hat_slices, y_low_likelihood = [], []
        mean_support_list, scale_support_list = [latent_means_lf], [latent_scales_lf]

        # LOW FREQUENCY (LF) Path
        for i, y_slice in enumerate(y_low_slices):
            mean_support = torch.cat(mean_support_list, dim=1)
            mu = self.cc_mean_transforms_lf[i](mean_support)[
                :, :, : y_shape[0] // 2, : y_shape[1] // 2
            ]

            scale_support = torch.cat(scale_support_list, dim=1)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, : y_shape[0] // 2, : y_shape[1] // 2
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            # Anchor
            empty_context = torch.zeros(
                support.size(0), self.slice_ch_lf, support.size(2), support.size(3),
                device=support.device, dtype=support.dtype
            )
            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([empty_context, support], dim=1)
            ).chunk(2, 1)

            scales_hat_split, means_hat_split = torch.zeros_like(
                y_slice
            ), torch.zeros_like(y_slice)
            scales_hat_split[:, :, 0::2, 0::2], scales_hat_split[:, :, 1::2, 1::2] = (
                scales_anchor[:, :, 0::2, 0::2],
                scales_anchor[:, :, 1::2, 1::2],
            )
            means_hat_split[:, :, 0::2, 0::2], means_hat_split[:, :, 1::2, 1::2] = (
                means_anchor[:, :, 0::2, 0::2],
                means_anchor[:, :, 1::2, 1::2],
            )

            y_anchor_quantized = self._quantize(anchor_split[i], means_anchor)
            y_anchor_quantized[:, :, 0::2, 1::2] = 0
            y_anchor_quantized[:, :, 1::2, 0::2] = 0

            # Non-Anchor
            masked_context = self.context_prediction_lf[i](y_anchor_quantized)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            scales_hat_split[:, :, 0::2, 1::2], scales_hat_split[:, :, 1::2, 0::2] = (
                scales_non_anchor[:, :, 0::2, 1::2],
                scales_non_anchor[:, :, 1::2, 0::2],
            )
            means_hat_split[:, :, 0::2, 1::2], means_hat_split[:, :, 1::2, 0::2] = (
                means_non_anchor[:, :, 0::2, 1::2],
                means_non_anchor[:, :, 1::2, 0::2],
            )

            y_non_anchor_quantized = self._quantize(
                non_anchor_split[i], means_non_anchor
            )
            y_non_anchor_quantized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantized[:, :, 1::2, 1::2] = 0

            _, y_slice_likelihood_val = self.gaussian_conditional_lf(
                y_slice, scales_hat_split, means=means_hat_split
            )
            y_low_likelihood.append(y_slice_likelihood_val)

            y_hat_slice = y_anchor_quantized + y_non_anchor_quantized
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )

            y_low_hat_slices.append(y_hat_slice)
            mean_support_list.append(y_hat_slice)
            scale_support_list.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)
        y_low_likelihoods = torch.cat(y_low_likelihood, dim=1)

        # HIGH FREQUENCY (HF) Path - Dictionary Cross Attention
        dt_token = rearrange(y_low_hat, "b c h w -> b (h w) c")
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_lh, y_hl, y_hh = y_high.chunk(3, dim=1)
        y_lh_slices, y_hl_slices, y_hh_slices = (
            y_lh.chunk(self.num_slices, 1),
            y_hl.chunk(self.num_slices, 1),
            y_hh.chunk(self.num_slices, 1),
        )
        lh_hats, hl_hats, hh_hats, y_high_likelihood = [], [], [], []

        for i in range(self.num_slices):
            # 1. LH
            sup_lh = torch.cat([y_low_hat, scales_lh, means_lh] + lh_hats, dim=1)
            mu_lh, sc_lh = self.fusion_lh[i](sup_lh, dt_token).chunk(2, 1)
            _, like_lh = self.gaussian_conditional_hf(
                y_lh_slices[i], sc_lh, means=mu_lh
            )
            lh_hats.append(self._quantize(y_lh_slices[i], mu_lh))

            # 2. HL
            sup_hl = torch.cat(
                [y_low_hat, scales_hl, means_hl] + hl_hats + lh_hats, dim=1
            )
            mu_hl, sc_hl = self.fusion_hl[i](sup_hl, dt_token).chunk(2, 1)
            _, like_hl = self.gaussian_conditional_hf(
                y_hl_slices[i], sc_hl, means=mu_hl
            )
            hl_hats.append(self._quantize(y_hl_slices[i], mu_hl))

            # 3. HH
            sup_hh = torch.cat(
                [y_low_hat, scales_hh, means_hh] + hh_hats + lh_hats + hl_hats, dim=1
            )
            mu_hh, sc_hh = self.fusion_hh[i](sup_hh, dt_token).chunk(2, 1)
            _, like_hh = self.gaussian_conditional_hf(
                y_hh_slices[i], sc_hh, means=mu_hh
            )
            hh_hats.append(self._quantize(y_hh_slices[i], mu_hh))

            y_high_likelihood.append(torch.cat([like_lh, like_hl, like_hh], dim=1))

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
        y = self.g_a(x)
        y_freq = self.dwt(y)
        H_wave, W_wave = y_freq.shape[2:]
        y_low, y_high = y_freq[:, : self.M], y_freq[:, self.M :]

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
        mean_support_list, scale_support_list = [latent_means_lf], [latent_scales_lf]

        for i, y_slice in enumerate(y_low_slices):
            mean_support = torch.cat(mean_support_list, dim=1)
            mu = self.cc_mean_transforms_lf[i](mean_support)[:, :, :H_wave, :W_wave]

            scale_support = torch.cat(scale_support_list, dim=1)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, :H_wave, :W_wave
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            # Anchor Compression
            empty_context = torch.zeros(
                support.size(0), self.slice_ch_lf, support.size(2), support.size(3),
                device=support.device, dtype=support.dtype
            )
            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([empty_context, support], dim=1)
            ).chunk(2, 1)

            B_slice, C_slice, H_slice, W_slice = y_slice.size()
            y_anchor_encode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice // 2, device=x.device
            )
            means_anchor_encode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice // 2, device=x.device
            )
            scales_anchor_encode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice // 2, device=x.device
            )

            y_anchor_encode[:, :, 0::2, :] = y_slice[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_slice[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            idx_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_anchor_encode
            )
            anchor_strings = self.gaussian_conditional_lf.compress(
                y_anchor_encode, idx_anchor, means=means_anchor_encode
            )
            anchor_quantized = self.gaussian_conditional_lf.decompress(
                anchor_strings, idx_anchor, means=means_anchor_encode
            )

            y_anchor_decode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice, device=x.device
            )
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # Non-Anchor Compression
            masked_context = self.context_prediction_lf[i](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            y_non_anchor_encode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice // 2, device=x.device
            )
            means_non_anchor_encode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice // 2, device=x.device
            )
            scales_non_anchor_encode = torch.zeros(
                B_slice, C_slice, H_slice, W_slice // 2, device=x.device
            )

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

            idx_non_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = self.gaussian_conditional_lf.compress(
                y_non_anchor_encode, idx_non_anchor, means=means_non_anchor_encode
            )
            non_anchor_quantized = self.gaussian_conditional_lf.decompress(
                non_anchor_strings, idx_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_decode = torch.zeros_like(y_anchor_decode)
            y_non_anchor_decode[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_decode[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_hat_slice = y_anchor_decode + y_non_anchor_decode
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )

            y_low_hat_slices.append(y_hat_slice)
            mean_support_list.append(y_hat_slice)
            scale_support_list.append(y_hat_slice)
            y_low_strings.append([anchor_strings, non_anchor_strings])

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # HIGH FREQUENCY (HF) Compression
        dt_token = rearrange(y_low_hat, "b c h w -> b (h w) c")
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_lh, y_hl, y_hh = y_high.chunk(3, dim=1)
        y_lh_slices, y_hl_slices, y_hh_slices = (
            y_lh.chunk(self.num_slices, 1),
            y_hl.chunk(self.num_slices, 1),
            y_hh.chunk(self.num_slices, 1),
        )

        lh_hats, hl_hats, hh_hats, y_high_strings = [], [], [], []

        for i in range(self.num_slices):
            # 1. LH
            sup_lh = torch.cat([y_low_hat, scales_lh, means_lh] + lh_hats, dim=1)
            mu_lh, sc_lh = self.fusion_lh[i](sup_lh, dt_token).chunk(2, 1)
            idx_lh = self.gaussian_conditional_hf.build_indexes(sc_lh)
            strings_lh = self.gaussian_conditional_hf.compress(
                y_lh_slices[i], idx_lh, means=mu_lh
            )
            lh_hats.append(
                self.gaussian_conditional_hf.decompress(strings_lh, idx_lh, means=mu_lh)
            )

            # 2. HL
            sup_hl = torch.cat(
                [y_low_hat, scales_hl, means_hl] + hl_hats + lh_hats, dim=1
            )
            mu_hl, sc_hl = self.fusion_hl[i](sup_hl, dt_token).chunk(2, 1)
            idx_hl = self.gaussian_conditional_hf.build_indexes(sc_hl)
            strings_hl = self.gaussian_conditional_hf.compress(
                y_hl_slices[i], idx_hl, means=mu_hl
            )
            hl_hats.append(
                self.gaussian_conditional_hf.decompress(strings_hl, idx_hl, means=mu_hl)
            )

            # 3. HH
            sup_hh = torch.cat(
                [y_low_hat, scales_hh, means_hh] + hh_hats + lh_hats + hl_hats, dim=1
            )
            mu_hh, sc_hh = self.fusion_hh[i](sup_hh, dt_token).chunk(2, 1)
            idx_hh = self.gaussian_conditional_hf.build_indexes(sc_hh)
            strings_hh = self.gaussian_conditional_hf.compress(
                y_hh_slices[i], idx_hh, means=mu_hh
            )
            hh_hats.append(
                self.gaussian_conditional_hf.decompress(strings_hh, idx_hh, means=mu_hh)
            )

            y_high_strings.append([strings_lh, strings_hl, strings_hh])

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
        mean_support_list, scale_support_list = [latent_means_lf], [latent_scales_lf]

        # LOW FREQUENCY (LF) Decompression
        for i in range(self.num_slices):
            mean_support = torch.cat(mean_support_list, dim=1)
            mu = self.cc_mean_transforms_lf[i](mean_support)[:, :, :H_wave, :W_wave]

            scale_support = torch.cat(scale_support_list, dim=1)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, :H_wave, :W_wave
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            # Anchor Decompression
            empty_context = torch.zeros(
                support.size(0), self.slice_ch_lf, support.size(2), support.size(3),
                device=support.device, dtype=support.dtype
            )
            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([empty_context, support], dim=1)
            ).chunk(2, 1)
            C_slice = means_anchor.size(1)
            means_anchor_encode = torch.zeros(
                B, C_slice, H_wave, W_wave // 2, device=z_hat.device
            )
            scales_anchor_encode = torch.zeros(
                B, C_slice, H_wave, W_wave // 2, device=z_hat.device
            )

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            idx_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_anchor_encode
            )
            anchor_strings = y_low_strings[i][0]
            anchor_quantized = self.gaussian_conditional_lf.decompress(
                anchor_strings, idx_anchor, means=means_anchor_encode
            )

            y_anchor_decode = torch.zeros(
                B, C_slice, H_wave, W_wave, device=z_hat.device
            )
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # Non-Anchor Decompression
            masked_context = self.context_prediction_lf[i](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)

            means_non_anchor_encode = torch.zeros(
                B, C_slice, H_wave, W_wave // 2, device=z_hat.device
            )
            scales_non_anchor_encode = torch.zeros(
                B, C_slice, H_wave, W_wave // 2, device=z_hat.device
            )

            means_non_anchor_encode[:, :, 0::2, :] = means_non_anchor[:, :, 0::2, 1::2]
            means_non_anchor_encode[:, :, 1::2, :] = means_non_anchor[:, :, 1::2, 0::2]
            scales_non_anchor_encode[:, :, 0::2, :] = scales_non_anchor[
                :, :, 0::2, 1::2
            ]
            scales_non_anchor_encode[:, :, 1::2, :] = scales_non_anchor[
                :, :, 1::2, 0::2
            ]

            idx_non_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_non_anchor_encode
            )
            non_anchor_strings = y_low_strings[i][1]
            non_anchor_quantized = self.gaussian_conditional_lf.decompress(
                non_anchor_strings, idx_non_anchor, means=means_non_anchor_encode
            )

            y_non_anchor_decode = torch.zeros_like(y_anchor_decode)
            y_non_anchor_decode[:, :, 0::2, 1::2] = non_anchor_quantized[:, :, 0::2, :]
            y_non_anchor_decode[:, :, 1::2, 0::2] = non_anchor_quantized[:, :, 1::2, :]

            y_hat_slice = y_anchor_decode + y_non_anchor_decode
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )

            y_low_hat_slices.append(y_hat_slice)
            mean_support_list.append(y_hat_slice)
            scale_support_list.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # HIGH FREQUENCY (HF) Decompression
        dt_token = rearrange(y_low_hat, "b c h w -> b (h w) c")
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        lh_hats, hl_hats, hh_hats = [], [], []

        for i in range(self.num_slices):
            strings_lh, strings_hl, strings_hh = y_high_strings[i]

            # 1. LH
            sup_lh = torch.cat([y_low_hat, scales_lh, means_lh] + lh_hats, dim=1)
            mu_lh, sc_lh = self.fusion_lh[i](sup_lh, dt_token).chunk(2, 1)
            idx_lh = self.gaussian_conditional_hf.build_indexes(sc_lh)
            lh_hats.append(
                self.gaussian_conditional_hf.decompress(strings_lh, idx_lh, means=mu_lh)
            )

            # 2. HL
            sup_hl = torch.cat(
                [y_low_hat, scales_hl, means_hl] + hl_hats + lh_hats, dim=1
            )
            mu_hl, sc_hl = self.fusion_hl[i](sup_hl, dt_token).chunk(2, 1)
            idx_hl = self.gaussian_conditional_hf.build_indexes(sc_hl)
            hl_hats.append(
                self.gaussian_conditional_hf.decompress(strings_hl, idx_hl, means=mu_hl)
            )

            # 3. HH
            sup_hh = torch.cat(
                [y_low_hat, scales_hh, means_hh] + hh_hats + lh_hats + hl_hats, dim=1
            )
            mu_hh, sc_hh = self.fusion_hh[i](sup_hh, dt_token).chunk(2, 1)
            idx_hh = self.gaussian_conditional_hf.build_indexes(sc_hh)
            hh_hats.append(
                self.gaussian_conditional_hf.decompress(strings_hh, idx_hh, means=mu_hh)
            )

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
