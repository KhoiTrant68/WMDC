import math
import torch
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel
from torch import nn
from torch_frft.frft_module import frft

from modules.dictionary_blocks import MultiScaleDictionaryCrossAttentionGLU
from modules.utils import CheckboardMaskedConv2d, conv, conv1x1, deconv
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import (
    ResidualBlockUpsample_wave,
    ResidualBlockWithStride_wave,
)


class LearnableFRFT_Split(nn.Module):
    """
    Helper module to apply learnable Fractional Fourier Transform (FRFT) on 2D feature maps.
    """

    def __init__(self, init_a=0.5):
        super().__init__()
        self.raw_a_h = nn.Parameter(torch.tensor(init_a))
        self.raw_a_w = nn.Parameter(torch.tensor(init_a))

    def forward(self, x):
        # Apply FRFT on Height (dim=2) and Width (dim=3)
        x = frft(x, self.raw_a_h, dim=2)
        x = frft(x, self.raw_a_w, dim=3)
        return x

    def inverse(self, x):
        # Inverse Apply (-a)
        x = frft(x, -self.raw_a_w, dim=3)
        x = frft(x, -self.raw_a_h, dim=2)
        return x


class LearnableFrftForward(nn.Module):
    """
    Generalized Frequency Decoupling using Fractional Fourier Transform (FRFT).
    Replaces the traditional Discrete Wavelet Transform (DWT). The pixel_unshuffle
    operation merely rearranges spatial features into the channel dimension before
    applying the continuous fractional domain transformation.
    """

    def __init__(self, channel_m, frft_module):
        super().__init__()
        self.frft = frft_module
        # Projection: Real+Imag (8M) -> Real (4M)
        self.c2r_conv = nn.Conv2d(channel_m * 8, channel_m * 4, kernel_size=1)

    def forward(self, x):
        x = nn.functional.pixel_unshuffle(x, 2)  # Spatial-to-channel rearrangement
        with torch.amp.autocast(
            device_type="cuda" if x.is_cuda else "cpu", dtype=torch.float32
        ):
            x_freq = self.frft(x.float())
            x_cat = torch.cat([x_freq.real, x_freq.imag], dim=1)  # [B, 8M, H/2, W/2]
        x_real = self.c2r_conv(x_cat)
        return x_real


class LearnableFrftInverse(nn.Module):
    """
    Inverse FRFT processing. Merges Frequency Components back to the spatial domain.
    Ensures invertibility by using the shared learnable parameters.
    """

    def __init__(self, channel_m, frft_module):
        super().__init__()
        self.frft = frft_module
        # Projection: Real (4M) -> Real+Imag (8M)
        self.r2c_conv = nn.Conv2d(channel_m * 4, channel_m * 8, kernel_size=1)

    def forward(self, x):
        x_cat = self.r2c_conv(x)  # [B, 8M, H/2, W/2]
        chunks = torch.chunk(x_cat, 2, dim=1)  # Tuple of[B, 4M, ...]
        with torch.amp.autocast(
            device_type="cuda" if x.is_cuda else "cpu", dtype=torch.float32
        ):
            x_complex = torch.complex(chunks[0].float(), chunks[1].float())
            x_spatial = self.frft.inverse(x_complex)
            x_spatial = x_spatial.real
        x_out = torch.nn.functional.pixel_shuffle(x_spatial, 2)  # [B, M, H, W]
        return x_out


class SoftQuantizer(nn.Module):
    """
    Annealed Quantizer: Switches from Additive Uniform Noise to STE Rounding.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("quant_mode", torch.tensor(0, dtype=torch.int))

    def set_mode(self, mode="noise"):
        self.quant_mode.fill_(0 if mode == "noise" else 1)

    def forward(self, x, means=None):
        if self.training and self.quant_mode == 0:
            noise = torch.rand_like(x) - 0.5
            return x + noise
        else:
            if means is not None:
                x_centered = x - means
                x_quant = torch.round(x_centered)
                return x_quant.detach() - x_centered.detach() + x_centered + means
            else:
                x_quant = torch.round(x)
                return x_quant.detach() - x.detach() + x


class MambaContextLayer(nn.Module):
    """
    Replaces redundant Swin-Attention with a pure Visual State Space (Mamba) block.
    This ensures the model relies exclusively on SSMs for spatial context extraction,
    sharpening the architectural narrative and reducing computational overhead.
    """

    def __init__(self, dim):
        super().__init__()
        # Utilizing the core VSSBlock to establish global receptive fields efficiently
        self.vss = VSSBlock(hidden_dim=dim, drop_path=0.0, ssm_d_state=16)

    def forward(self, x):
        return self.vss(x)


class WMDC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, dict_num=128, dict_head_num=20):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.window_size = 8

        self.slice_ch_lf = M // num_slices
        self.slice_ch_hf = (3 * M) // num_slices

        self.quantizer = SoftQuantizer()

        # A. MAIN ENCODER
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            conv(N, M, kernel_size=5, stride=2),
        )

        # B. SPECTRAL SPLIT (Learnable FRFT Domain Transformation)
        self.shared_frft = LearnableFRFT_Split(init_a=0.5)
        self.frft_forward = LearnableFrftForward(M, self.shared_frft)
        self.frft_inverse = LearnableFrftInverse(M, self.shared_frft)

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

        # D. PATH A: LOW FREQUENCY ENTROPY MODEL (Pure Mamba Context)
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
                VSSBlock(
                    hidden_dim=(
                        2 * M + 2 * self.slice_ch_lf
                        if i == 0
                        else 2 * M + 4 * self.slice_ch_lf
                    ),
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

        # E. PATH B: HIGH FREQUENCY ENTROPY MODEL (Dictionary & Cross-Attention)
        self.dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(
            torch.randn([dict_num, self.dict_dim]), requires_grad=True
        )

        # BOTTLENECK: Compress massive channel dimensions down to 512 to prevent FLOPs explosion
        # Base input = M (y_low_hat) + 6*M (params_hf) = 7*M. Plus decoded slices = slice_ch_hf * i
        self.hf_bottlenecks = nn.ModuleList(
            conv1x1(7 * M + self.slice_ch_hf * i, 512) for i in range(self.num_slices)
        )

        self.dt_cross_attention_hf = nn.ModuleList(
            MultiScaleDictionaryCrossAttentionGLU(
                input_dim=512,  # Uses the bottleneck output
                output_dim=3 * M,
                head_num=dict_head_num,
                mlp_rate=4,
            )
            for i in range(self.num_slices)
        )

        gaussian_out_dim = 2 * self.slice_ch_hf

        self.cc_transforms_hf = nn.ModuleList(
            nn.Sequential(
                conv(
                    512 + 3 * M, 512, stride=1, kernel_size=3
                ),  # Query (512) + Dict Output (3M)
                nn.GELU(),
                conv(512, 256, stride=1, kernel_size=3),
                nn.GELU(),
                conv(256, gaussian_out_dim, stride=1, kernel_size=3),
            )
            for i in range(self.num_slices)
        )

        self.lrp_transforms_hf = nn.ModuleList(
            nn.Sequential(
                conv(512 + 3 * M + self.slice_ch_hf, 512, stride=1, kernel_size=3),
                nn.GELU(),
                conv(512, 256, stride=1, kernel_size=3),
                nn.GELU(),
                conv(256, self.slice_ch_hf, stride=1, kernel_size=3),
            )
            for i in range(self.num_slices)
        )
        self.gaussian_conditional_hf = GaussianConditional(None)

        # F. MAIN DECODER
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    def set_quantization_stage(self, mode="noise"):
        self.quantizer.set_mode(mode)

    # ==========================================================================
    # FORWARD
    # ==========================================================================
    def forward(self, x):
        B, C, H, W = x.size()

        y = self.g_a(x)
        y_shape = y.shape[2:]

        # Fractional Domain Decoupling
        y_freq = self.frft_forward(y)
        y_low = y_freq[:, : self.M, :, :]
        y_high = y_freq[:, self.M :, :, :]

        z = self.h_a(y_freq)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = self.quantizer(z - z_offset) + z_offset

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
        params_hf = torch.cat([latent_scales_hf, latent_means_hf], dim=1)

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

        y_low_hat_slices = []
        y_low_likelihood = []

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
            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

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

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

            _, y_slice_likelihood = self.gaussian_conditional_lf(
                y_slice, scales_hat_split, means=means_hat_split
            )

            y_non_anchor = non_anchor_split[i]
            y_non_anchor_quantized = self.quantizer(
                y_non_anchor, means=means_non_anchor
            )
            y_non_anchor_quantized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantized[:, :, 1::2, 1::2] = 0

            y_hat_slice = y_anchor_quantized + y_non_anchor_quantized
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_low_hat_slices.append(y_hat_slice)
            y_low_likelihood.append(y_slice_likelihood)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)
        y_low_likelihoods = torch.cat(y_low_likelihood, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY (Bottleneck + Dictionary Cross-Attention)
        # ----------------------------------------------------------------------
        y_high_slices = y_high.chunk(self.num_slices, 1)
        y_high_hat_slices = []
        y_high_likelihood = []
        dt_batch = self.dt.repeat([B, 1, 1])

        for i, y_slice in enumerate(y_high_slices):
            query_raw = torch.cat([y_low_hat, params_hf] + y_high_hat_slices, dim=1)

            # Bottleneck to resolve channel explosion
            query = self.hf_bottlenecks[i](query_raw)

            dict_info = self.dt_cross_attention_hf[i](query, dt_batch)

            support = torch.cat([query, dict_info], dim=1)
            gaussian_params = self.cc_transforms_hf[i](support)
            mu, scale = gaussian_params.chunk(2, 1)

            # Ensure strictly positive scales
            scale = nn.functional.softplus(scale) + 1e-6

            _, y_slice_likelihood = self.gaussian_conditional_hf(
                y_slice, scale, means=mu
            )
            y_high_likelihood.append(y_slice_likelihood)

            y_hat_slice = self.quantizer(y_slice, means=mu)

            # LRP
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms_hf[i](lrp_support))
            y_hat_slice += lrp

            y_high_hat_slices.append(y_hat_slice)

        y_high_hat = torch.cat(y_high_hat_slices, dim=1)
        y_high_likelihoods = torch.cat(y_high_likelihood, dim=1)

        # ----------------------------------------------------------------------
        # MERGE & DECODE
        # ----------------------------------------------------------------------
        y_freq_hat = torch.cat([y_low_hat, y_high_hat], dim=1)
        y_tilde = self.frft_inverse(y_freq_hat)
        x_hat = self.g_s(y_tilde)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y_low": y_low_likelihoods,
                "y_high": y_high_likelihoods,
                "z": z_likelihoods,
            },
        }

    # ==========================================================================
    # UPDATE CDFs
    # ==========================================================================
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
        y_freq = self.frft_forward(y)
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
        params_hf = torch.cat([latent_scales_hf, latent_means_hf], dim=1)

        # ----------------------------------------------------------------------
        # PATH A: LOW FREQUENCY
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
        symbols_list_lf = []
        indexes_list_lf = []

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

            symbols_list_lf.extend(y_anchor_symbols_shifted.reshape(-1).tolist())
            indexes_list_lf.extend(indexes_anchor.reshape(-1).tolist())

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
            y_non_anchor_symbols = self.gaussian_conditional_lf.quantize(
                y_non_anchor_encode, "symbols", means_non_anchor_encode
            )

            offset_non_anchor = self.gaussian_conditional_lf.offset.view(-1)[
                indexes_non_anchor
            ]
            y_non_anchor_symbols_shifted = (
                y_non_anchor_symbols - offset_non_anchor
            ).int()

            symbols_list_lf.extend(y_non_anchor_symbols_shifted.reshape(-1).tolist())
            indexes_list_lf.extend(indexes_non_anchor.reshape(-1).tolist())

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
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_low_hat_slices.append(y_hat_slice)

        encoder_lf.encode_with_indexes(
            symbols_list_lf, indexes_list_lf, cdf_lf, cdf_lengths_lf, offsets_lf
        )
        y_low_string = encoder_lf.flush()
        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY
        # ----------------------------------------------------------------------
        y_high_slices = y_high.chunk(self.num_slices, 1)
        y_high_hat_slices = []
        dt_batch = self.dt.repeat([B, 1, 1])

        cdf_hf = self.gaussian_conditional_hf.quantized_cdf.tolist()
        cdf_lengths_hf = (
            self.gaussian_conditional_hf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_hf = self.gaussian_conditional_hf.offset.reshape(-1).int().tolist()

        encoder_hf = BufferedRansEncoder()
        symbols_list_hf = []
        indexes_list_hf = []

        for i, y_slice in enumerate(y_high_slices):
            query_raw = torch.cat([y_low_hat, params_hf] + y_high_hat_slices, dim=1)
            query = self.hf_bottlenecks[i](query_raw)
            dict_info = self.dt_cross_attention_hf[i](query, dt_batch)

            support = torch.cat([query, dict_info], dim=1)
            gaussian_params = self.cc_transforms_hf[i](support)
            mu, scale = gaussian_params.chunk(2, 1)
            scale = nn.functional.softplus(scale) + 1e-6

            indexes = self.gaussian_conditional_hf.build_indexes(scale)
            y_hf_symbols = self.gaussian_conditional_hf.quantize(y_slice, "symbols", mu)

            offset_hf = self.gaussian_conditional_hf.offset.view(-1)[indexes]
            y_hf_symbols_shifted = (y_hf_symbols - offset_hf).int()
            symbols_list_hf.extend(y_hf_symbols_shifted.reshape(-1).tolist())
            indexes_list_hf.extend(indexes.reshape(-1).tolist())

            y_hat_slice = self.gaussian_conditional_hf.dequantize(y_hf_symbols, mu)

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms_hf[i](lrp_support))
            y_hat_slice += lrp

            y_high_hat_slices.append(y_hat_slice)

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
        params_hf = torch.cat([latent_scales_hf, latent_means_hf], dim=1)
        H_wave, W_wave = latent_scales.shape[2:]

        # ----------------------------------------------------------------------
        # PATH A: LOW FREQUENCY
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
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_low_hat_slices.append(y_hat_slice)

        y_low_hat = torch.cat(y_low_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY
        # ----------------------------------------------------------------------
        y_high_hat_slices = []
        dt_batch = self.dt.repeat([B, 1, 1])

        cdf_hf = self.gaussian_conditional_hf.quantized_cdf.tolist()
        cdf_lengths_hf = (
            self.gaussian_conditional_hf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_hf = self.gaussian_conditional_hf.offset.reshape(-1).int().tolist()

        decoder_hf = RansDecoder()
        decoder_hf.set_stream(y_high_string)

        for i in range(self.num_slices):
            query_raw = torch.cat([y_low_hat, params_hf] + y_high_hat_slices, dim=1)
            query = self.hf_bottlenecks[i](query_raw)
            dict_info = self.dt_cross_attention_hf[i](query, dt_batch)

            support = torch.cat([query, dict_info], dim=1)
            gaussian_params = self.cc_transforms_hf[i](support)
            mu, scale = gaussian_params.chunk(2, 1)
            scale = nn.functional.softplus(scale) + 1e-6

            indexes = self.gaussian_conditional_hf.build_indexes(scale)

            rv_hf = decoder_hf.decode_stream(
                indexes.reshape(-1).tolist(), cdf_hf, cdf_lengths_hf, offsets_hf
            )
            rv_hf = torch.Tensor(rv_hf).to(z_hat.device).reshape(indexes.size())

            offset_hf = self.gaussian_conditional_hf.offset.view(-1)[indexes]
            y_hat_slice = rv_hf + offset_hf + mu

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms_hf[i](lrp_support))
            y_hat_slice += lrp

            y_high_hat_slices.append(y_hat_slice)

        y_high_hat = torch.cat(y_high_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # MERGE & DECODE
        # ----------------------------------------------------------------------
        y_freq_hat = torch.cat([y_low_hat, y_high_hat], dim=1)
        y_tilde = self.frft_inverse(y_freq_hat)
        x_hat = self.g_s(y_tilde).clamp_(0, 1)

        return {"x_hat": x_hat}
