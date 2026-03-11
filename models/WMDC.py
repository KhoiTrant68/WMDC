import math

import torch
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from torch import nn

from modules.dictionary_blocks import MultiScaleDictionaryCrossAttentionGLU
from modules.utils import CheckboardMaskedConv2d, conv, deconv, ste_round
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import (
    DWT_2D,
    IDWT_2D,
    ResidualBlockUpsample_wave,
    ResidualBlockWithStride_wave,
)


class SoftQuantizer(nn.Module):
    """
    Transitions smoothly from uniform noise (tau=0) to STE (tau=1) over training,
    stabilizing gradients for highly non-linear architectures (Mamba + Cross-Attention).
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("tau", torch.tensor(0.0, dtype=torch.float32))

    def update_tau(self, tau_value):
        self.tau.fill_(tau_value)

    def forward(self, x, means=None, noise=None):
        if self.training:
            x_centered = x - means if means is not None else x
            ste_out = ste_round(x_centered)
            ste_out = ste_out + means if means is not None else ste_out

            if noise is None:
                noise = torch.rand_like(x) - 0.5
            noise_out = x + noise

            return self.tau * ste_out + (1.0 - self.tau) * noise_out
        else:
            if means is not None:
                return torch.round(x - means) + means
            return torch.round(x)


class MambaContextLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.vss = VSSBlock(hidden_dim=dim, drop_path=0.0, ssm_d_state=16)

    def forward(self, x):
        return self.vss(x)


class WMDC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, dict_num=128, dict_head_num=20):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = N
        self.M = M
        self.num_slices = num_slices

        self.slice_ch_lf = M // num_slices
        self.slice_ch_hf = (3 * M) // num_slices
        self.slice_ch_dir = self.slice_ch_hf // 3  # Channels per sub-band per slice

        self.quantizer = SoftQuantizer()

        # A. MAIN ENCODER (Spatial Wavelet CNN)
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            conv(N, M, kernel_size=5, stride=2),
        )

        # B. SPECTRAL SPLIT (Wavelet Transform)
        self.dwt = DWT_2D(wave="haar")
        self.idwt = IDWT_2D(wave="haar")

        # C. HYPER-PRIOR AUTOENCODER (State Space Prior)
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

        # D. PATH A: LOW FREQUENCY ENTROPY MODEL (Mamba Checkerboard Context)
        self.atten_mean_lf = nn.ModuleList(
            MambaContextLayer(M + self.slice_ch_lf * min(i, 5))
            for i in range(self.num_slices)
        )
        self.atten_scale_lf = nn.ModuleList(
            MambaContextLayer(M + self.slice_ch_lf * min(i, 5))
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
            MambaContextLayer(self.slice_ch_lf) for _ in range(self.num_slices)
        )
        self.anchor_atten_scale_lf = nn.ModuleList(
            MambaContextLayer(self.slice_ch_lf) for _ in range(self.num_slices)
        )

        self.context_vss_lf = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(
                    (
                        2 * M + 2 * self.slice_ch_lf
                        if i == 0
                        else 2 * M + 4 * self.slice_ch_lf
                    ),
                    128,
                    kernel_size=1,
                ),
                nn.GELU(),
                nn.Conv2d(128, 2 * self.slice_ch_lf, kernel_size=3, padding=1),
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

        # E. PATH B: DIRECTION-DECOUPLED HIGH FREQUENCY ENTROPY MODEL
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
        z_hat_noise, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians().detach()
        if self.training:
            z_ste = ste_round(z - z_offset) + z_offset
            z_hat = (
                self.quantizer.tau * z_ste + (1.0 - self.quantizer.tau) * z_hat_noise
            )
        else:
            z_hat = z_hat_noise

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
        # PATH A: LOW FREQUENCY (Mamba Checkerboard Context)
        # ----------------------------------------------------------------------
        anchor = torch.zeros_like(y_low).to(x.device)
        non_anchor = torch.zeros_like(y_low).to(x.device)
        anchor[:, :, 0::2, 0::2] = y_low[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y_low[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y_low[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y_low[:, :, 1::2, 0::2]

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
            scales_anchor = (
                torch.nn.functional.softplus(
                    self.anchor_atten_scale_lf[i](scales_anchor)
                )
                + 1e-6
            )

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

            y_anchor_quantized = self.quantizer(y_anchor, means=means_hat_split)
            y_anchor_quantized[:, :, 0::2, 1::2] = 0
            y_anchor_quantized[:, :, 1::2, 0::2] = 0

            masked_context = self.context_prediction_lf[i](y_anchor_quantized)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = (
                torch.nn.functional.softplus(
                    self.anchor_atten_scale_lf[i](scales_non_anchor)
                )
                + 1e-6
            )

            scales_hat_split[:, :, 0::2, 1::2], scales_hat_split[:, :, 1::2, 0::2] = (
                scales_non_anchor[:, :, 0::2, 1::2],
                scales_non_anchor[:, :, 1::2, 0::2],
            )
            means_hat_split[:, :, 0::2, 1::2], means_hat_split[:, :, 1::2, 0::2] = (
                means_non_anchor[:, :, 0::2, 1::2],
                means_non_anchor[:, :, 1::2, 0::2],
            )

            y_non_anchor = non_anchor_split[i]
            y_non_anchor_quantized = self.quantizer(
                y_non_anchor, means=means_non_anchor
            )
            y_non_anchor_quantized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantized[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantized + y_non_anchor_quantized

            y_slice_likelihood = self.gaussian_conditional_lf._likelihood(
                y_hat_slice, scales_hat_split, means=means_hat_split
            )
            y_slice_likelihood = self.gaussian_conditional_lf.likelihood_lower_bound(
                y_slice_likelihood
            )
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_low_hat_slices.append(y_hat_slice)
            y_low_likelihood.append(y_slice_likelihood)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)
        y_low_likelihoods = torch.cat(y_low_likelihood, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY (Proper Directional Splitting)
        # ----------------------------------------------------------------------
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_lh, y_hl, y_hh = y_high.chunk(3, dim=1)
        y_lh_slices = y_lh.chunk(self.num_slices, dim=1)
        y_hl_slices = y_hl.chunk(self.num_slices, dim=1)
        y_hh_slices = y_hh.chunk(self.num_slices, dim=1)

        y_high_hat_slices, y_high_likelihood = [], []
        lh_hats, hl_hats, hh_hats = [], [], []

        for i in range(self.num_slices):
            # Form accurate directional slice subset
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
            scale = nn.functional.softplus(scale) + 1e-6

            y_hat_slice = self.quantizer(y_slice, means=mu)
            y_slice_like = self.gaussian_conditional_hf._likelihood(
                y_hat_slice, scale, means=mu
            )
            y_slice_like = self.gaussian_conditional_hf.likelihood_lower_bound(
                y_slice_like
            )
            y_high_likelihood.append(y_slice_like)

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

        # y_high_hat = torch.cat(y_high_hat_slices, dim=1)
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

        # Return dict natively includes aux_loss for DDP stability
        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y_low": y_low_likelihoods,
                "y_high": y_high_likelihoods,
                "z": z_likelihoods,
            },
            "aux_loss": self.aux_loss(),  # Triggers self.entropy_bottleneck.loss() correctly
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
        # [Truncated logic identical to previous compress, ensure y_high is split correctly]
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

        # ----------------------------------------------------------------------
        # PATH A: LOW FREQUENCY (Identical to forward pass)
        # ----------------------------------------------------------------------
        y_low_slices = y_low.chunk(self.num_slices, 1)
        y_low_hat_slices = []

        ctx_params_anchor_split = torch.split(
            torch.zeros(B, self.slice_ch_lf * 2 * self.num_slices, H_wave, W_wave).to(
                x.device
            ),
            [2 * self.slice_ch_lf for _ in range(self.num_slices)],
            1,
        )

        cdf_lf = self.gaussian_conditional_lf.quantized_cdf.tolist()
        cdf_lengths_lf = (
            self.gaussian_conditional_lf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_lf = self.gaussian_conditional_lf.offset.reshape(-1).int().tolist()

        encoder_lf = BufferedRansEncoder()
        symbols_tensors_lf, indexes_tensors_lf = [], []

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
            scales_anchor = (
                torch.nn.functional.softplus(
                    self.anchor_atten_scale_lf[i](scales_anchor)
                )
                + 1e-6
            )

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

            y_anchor_encode[:, :, 0::2, :] = y_anchor[:, :, 0::2, 0::2]
            y_anchor_encode[:, :, 1::2, :] = y_anchor[:, :, 1::2, 1::2]
            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_anchor_encode
            )
            y_anchor_symbols = self.gaussian_conditional_lf.quantize(
                y_anchor_encode, "symbols", means_anchor_encode
            )

            offset_anchor = self.gaussian_conditional_lf.offset.view(-1)[indexes_anchor]
            y_anchor_symbols_shifted = (y_anchor_symbols - offset_anchor).int()

            symbols_tensors_lf.append(y_anchor_symbols_shifted.view(-1))
            indexes_tensors_lf.append(indexes_anchor.view(-1))

            anchor_quantized = self.gaussian_conditional_lf.dequantize(
                y_anchor_symbols, means_anchor_encode
            )
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                x.device
            )
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            masked_context = self.context_prediction_lf[i](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = (
                torch.nn.functional.softplus(
                    self.anchor_atten_scale_lf[i](scales_non_anchor)
                )
                + 1e-6
            )

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
            y_non_anchor_symbols = self.gaussian_conditional_lf.quantize(
                y_non_anchor_encode, "symbols", means_non_anchor_encode
            )

            offset_non_anchor = self.gaussian_conditional_lf.offset.view(-1)[
                indexes_non_anchor
            ]
            y_non_anchor_symbols_shifted = (
                y_non_anchor_symbols - offset_non_anchor
            ).int()

            symbols_tensors_lf.append(y_non_anchor_symbols_shifted.view(-1))
            indexes_tensors_lf.append(indexes_non_anchor.view(-1))

            non_anchor_quantized = self.gaussian_conditional_lf.dequantize(
                y_non_anchor_symbols, means_non_anchor_encode
            )
            y_non_anchor_quantized = torch.zeros_like(y_anchor)
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

        symbols_list_lf = (
            torch.cat(symbols_tensors_lf).cpu().tolist() if symbols_tensors_lf else []
        )
        indexes_list_lf = (
            torch.cat(indexes_tensors_lf).cpu().tolist() if indexes_tensors_lf else []
        )

        encoder_lf.encode_with_indexes(
            symbols_list_lf, indexes_list_lf, cdf_lf, cdf_lengths_lf, offsets_lf
        )
        y_low_string = encoder_lf.flush()
        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY (Proper Directional Splitting)
        # ----------------------------------------------------------------------
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_lh, y_hl, y_hh = y_high.chunk(3, dim=1)
        y_lh_slices = y_lh.chunk(self.num_slices, dim=1)
        y_hl_slices = y_hl.chunk(self.num_slices, dim=1)
        y_hh_slices = y_hh.chunk(self.num_slices, dim=1)

        y_high_hat_slices = []
        lh_hats, hl_hats, hh_hats = [], [], []

        cdf_hf = self.gaussian_conditional_hf.quantized_cdf.tolist()
        cdf_lengths_hf = (
            self.gaussian_conditional_hf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_hf = self.gaussian_conditional_hf.offset.reshape(-1).int().tolist()

        encoder_hf = BufferedRansEncoder()
        symbols_tensors_hf, indexes_tensors_hf = [], []

        for i in range(self.num_slices):
            # Form accurate directional slice subset
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
            scale = nn.functional.softplus(scale) + 1e-6

            indexes = self.gaussian_conditional_hf.build_indexes(scale)
            y_hf_symbols = self.gaussian_conditional_hf.quantize(y_slice, "symbols", mu)
            offset_hf = self.gaussian_conditional_hf.offset.view(-1)[indexes]
            y_hf_symbols_shifted = (y_hf_symbols - offset_hf).int()

            symbols_tensors_hf.append(y_hf_symbols_shifted.view(-1))
            indexes_tensors_hf.append(indexes.view(-1))

            y_hat_slice = self.gaussian_conditional_hf.dequantize(y_hf_symbols, mu)
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

        # Batch transfer optimization
        symbols_list_hf = (
            torch.cat(symbols_tensors_hf).cpu().tolist() if symbols_tensors_hf else []
        )
        indexes_list_hf = (
            torch.cat(indexes_tensors_hf).cpu().tolist() if indexes_tensors_hf else []
        )

        encoder_hf.encode_with_indexes(
            symbols_list_hf, indexes_list_hf, cdf_hf, cdf_lengths_hf, offsets_hf
        )
        y_high_string = encoder_hf.flush()

        return {
            "strings": [[y_low_string, y_high_string], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        y_strings, z_strings = strings[0], strings[1]
        y_low_string, y_high_string = y_strings[0], y_strings[1]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
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

        # ----------------------------------------------------------------------
        # PATH A: LOW FREQUENCY (Identical to compress pass)
        # ----------------------------------------------------------------------
        y_low_hat_slices = []
        ctx_params_anchor_split = torch.split(
            torch.zeros(B, self.slice_ch_lf * 2 * self.num_slices, H_wave, W_wave).to(
                z_hat.device
            ),
            [2 * self.slice_ch_lf for _ in range(self.num_slices)],
            1,
        )

        cdf_lf = self.gaussian_conditional_lf.quantized_cdf.tolist()
        cdf_lengths_lf = (
            self.gaussian_conditional_lf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_lf = self.gaussian_conditional_lf.offset.reshape(-1).int().tolist()

        decoder_lf = RansDecoder()
        decoder_lf.set_stream(y_low_string)

        for i in range(self.num_slices):
            # [Identical decompression block exactly as in the original code for PATH A...]
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
            scales_anchor = (
                torch.nn.functional.softplus(
                    self.anchor_atten_scale_lf[i](scales_anchor)
                )
                + 1e-6
            )

            B_anchor, C_anchor, H_anchor, W_anchor = means_anchor.size()
            means_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)
            scales_anchor_encode = torch.zeros(
                B_anchor, C_anchor, H_anchor, W_anchor // 2
            ).to(z_hat.device)

            means_anchor_encode[:, :, 0::2, :] = means_anchor[:, :, 0::2, 0::2]
            means_anchor_encode[:, :, 1::2, :] = means_anchor[:, :, 1::2, 1::2]
            scales_anchor_encode[:, :, 0::2, :] = scales_anchor[:, :, 0::2, 0::2]
            scales_anchor_encode[:, :, 1::2, :] = scales_anchor[:, :, 1::2, 1::2]

            indexes_anchor = self.gaussian_conditional_lf.build_indexes(
                scales_anchor_encode
            )

            rv_anchor = decoder_lf.decode_stream(
                indexes_anchor.reshape(-1).tolist(), cdf_lf, cdf_lengths_lf, offsets_lf
            )
            rv_anchor = (
                torch.Tensor(rv_anchor).to(z_hat.device).reshape(indexes_anchor.size())
            )

            offset_anchor = self.gaussian_conditional_lf.offset.view(-1)[indexes_anchor]
            rv_anchor = rv_anchor + offset_anchor
            anchor_quantized = rv_anchor + means_anchor_encode

            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                z_hat.device
            )
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            masked_context = self.context_prediction_lf[i](y_anchor_decode)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = (
                torch.nn.functional.softplus(
                    self.anchor_atten_scale_lf[i](scales_non_anchor)
                )
                + 1e-6
            )

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

            rv_non_anchor = decoder_lf.decode_stream(
                indexes_non_anchor.reshape(-1).tolist(),
                cdf_lf,
                cdf_lengths_lf,
                offsets_lf,
            )
            rv_non_anchor = (
                torch.Tensor(rv_non_anchor)
                .to(z_hat.device)
                .reshape(indexes_non_anchor.size())
            )

            offset_non_anchor = self.gaussian_conditional_lf.offset.view(-1)[
                indexes_non_anchor
            ]
            rv_non_anchor = rv_non_anchor + offset_non_anchor
            non_anchor_quantized = rv_non_anchor + means_non_anchor_encode

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

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY (Proper Directional Splitting matching compress layout)
        # ----------------------------------------------------------------------
        scales_lh, scales_hl, scales_hh = latent_scales_hf.chunk(3, dim=1)
        means_lh, means_hl, means_hh = latent_means_hf.chunk(3, dim=1)

        y_high_hat_slices = []
        lh_hats, hl_hats, hh_hats = [], [], []

        cdf_hf = self.gaussian_conditional_hf.quantized_cdf.tolist()
        cdf_lengths_hf = (
            self.gaussian_conditional_hf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_hf = self.gaussian_conditional_hf.offset.reshape(-1).int().tolist()

        decoder_hf = RansDecoder()
        decoder_hf.set_stream(y_high_string)

        for i in range(self.num_slices):
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
            scale = nn.functional.softplus(scale) + 1e-6

            indexes = self.gaussian_conditional_hf.build_indexes(scale)

            rv_hf = decoder_hf.decode_stream(
                indexes.reshape(-1).tolist(), cdf_hf, cdf_lengths_hf, offsets_hf
            )
            rv_hf = torch.Tensor(rv_hf).to(z_hat.device).reshape(indexes.size())

            offset_hf = self.gaussian_conditional_hf.offset.view(-1)[indexes]

            y_hat_slice = rv_hf + offset_hf + mu
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

        # y_high_hat = torch.cat(y_high_hat_slices, dim=1)
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
