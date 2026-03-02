import math
import torch
from torch import nn
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import AttentionBlock
from compressai.models import CompressionModel

from modules.dictionary_blocks import MultiScaleDictionaryCrossAttentionGLU
from modules.utils import CheckboardMaskedConv2d, conv, conv1x1, deconv, ste_round
from modules.VSS_module import SwinBlock, VSSBlock
from modules.wavelet_blocks import (
    ResidualBlockUpsample_wave,
    ResidualBlockWithStride_wave,
)

from torch_frft.frft_module import frft


class LearnableFRFT_Split(nn.Module):
    """
    Helper module to apply learnable FRFT on 2D feature maps.
    """
    def __init__(self, init_a=0.5):
        super().__init__()
        self.raw_a_h = nn.Parameter(torch.tensor(init_a))
        self.raw_a_w = nn.Parameter(torch.tensor(init_a))

    def forward(self, x):
        # x: [B, C, H, W] (Complex or Real)
        # Apply FRFT on Height (dim=2) and Width (dim=3)
        x = frft(x, self.raw_a_h, dim=2)
        x = frft(x, self.raw_a_w, dim=3)
        return x

    def inverse(self, x):
        # Inverse Apply (-a)
        x = frft(x, -self.raw_a_w, dim=3)
        x = frft(x, -self.raw_a_h, dim=2)
        return x


class FRFTSplitter(nn.Module):
    """
    Replaces DWT_2D.
    Splits latent y [B, M, H, W] into Frequency Components [B, 4M, H/2, W/2].
    
    Pipeline:
    1. Pixel Unshuffle (Space-to-Depth): [B, M, H, W] -> [B, 4M, H/2, W/2]
    2. Learnable FRFT: Rotates into optimal fractional frequency domain.
    3. Complex Projection: Learns to map Complex FRFT coefficients to Real-valued latents.
    """
    def __init__(self, channel_m):
        super().__init__()
        self.frft = LearnableFRFT_Split(init_a=0.5)
        # Projection: Real+Imag (8M) -> Real (4M)
        self.c2r_conv = nn.Conv2d(channel_m * 8, channel_m * 4, kernel_size=1)

    def forward(self, x):
        # 1. Space to Depth
        x = nn.functional.pixel_unshuffle(x, 2) # [B, 4M, H/2, W/2]
        
        # 2. FRFT (Returns Complex)
        x_freq = self.frft(x) 
        
        # 3. Projection to Real (for Entropy Coding compatibility)
        # Cat Real and Imag components along channel dimension
        x_cat = torch.cat([x_freq.real, x_freq.imag], dim=1) # [B, 8M, H/2, W/2]
        
        # Mix them back to 4M channels
        x_real = self.c2r_conv(x_cat)
        
        return x_real


class FRFTMerger(nn.Module):
    """
    Replaces IDWT_2D.
    Merges Frequency Components [B, 4M, H/2, W/2] back to [B, M, H, W].
    """
    def __init__(self, channel_m):
        super().__init__()
        self.frft = LearnableFRFT_Split(init_a=0.5)
        # Projection: Real (4M) -> Real+Imag (8M)
        self.r2c_conv = nn.Conv2d(channel_m * 4, channel_m * 8, kernel_size=1)

    def forward(self, x):
        # x: [B, 4M, H/2, W/2] (Real, from entropy decoder)
        
        # 1. Project back to Complex Space components
        x_cat = self.r2c_conv(x) # [B, 8M, H/2, W/2]
        
        # Split back into Real and Imag parts
        chunks = torch.chunk(x_cat, 2, dim=1) # Tuple of [B, 4M, ...]
        x_complex = torch.complex(chunks[0], chunks[1])
        
        # 2. Inverse FRFT
        x_spatial = self.frft.inverse(x_complex)
        
        # 3. Take Real part (Energy compaction usually preserves info in Real)
        # Note: Ideally, we take magnitude, but keeping Real allows signed values 
        # which acts better with PixelShuffle reconstruction.
        x_spatial = x_spatial.real 
        
        # 4. Depth to Space
        x_out = torch.nn.functional.pixel_shuffle(x_spatial, 2) # [B, M, H, W]
        
        return x_out


class SWAtten(AttentionBlock):
    def __init__(
        self, input_dim, output_dim, head_dim, window_size, drop_path, inter_dim=192
    ) -> None:
        if inter_dim is not None:
            super().__init__(N=inter_dim)
            self.non_local_block = SwinBlock(
                inter_dim, inter_dim, head_dim, window_size, drop_path
            )
            self.in_conv = conv1x1(input_dim, inter_dim)
            self.out_conv = conv1x1(inter_dim, output_dim)
        else:
            super().__init__(N=input_dim)
            self.non_local_block = SwinBlock(
                input_dim, input_dim, head_dim, window_size, drop_path
            )

        self.window_size = window_size

    def forward(self, x):
        if hasattr(self, "in_conv"):
            x = self.in_conv(x)

        identity = x

        # --- DYNAMIC REFLECTION PADDING  ---
        B, C, H, W = x.shape
        pad_r = (self.window_size - x.size(-1) % self.window_size) % self.window_size
        pad_b = (self.window_size - x.size(-2) % self.window_size) % self.window_size

        if pad_r > 0 or pad_b > 0:
            x_padded = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b), mode="reflect")
        else:
            x_padded = x

        z = self.non_local_block(x_padded)

        if pad_r > 0 or pad_b > 0:
            z = z[:, :, :H, :W]
        # --------------------------------------

        a = self.conv_a(x)
        b = self.conv_b(z)
        out = a * torch.sigmoid(b)
        out += identity

        if hasattr(self, "out_conv"):
            out = self.out_conv(out)

        return out


class WMDC(CompressionModel):
    def __init__(self, N=192, M=320, num_slices=5, dict_num=128, dict_head_num=20):
        super().__init__(entropy_bottleneck_channels=N)
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.window_size = 8

        self.slice_ch_lf = M // num_slices  # 320 // 5 = 64
        self.slice_ch_hf = (3 * M) // num_slices  # 960 // 5 = 192

        # ----------------------------------------------------------------------
        # A. MAIN ENCODER (FRFT-Integrated)
        # ----------------------------------------------------------------------
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            ResidualBlockWithStride_wave(N, N, stride=2, wavelet="haar"),
            conv(N, M, kernel_size=5, stride=2),
        )

        # ----------------------------------------------------------------------
        # B. SPECTRAL SPLIT (Learned FRFT Splitter)
        # ----------------------------------------------------------------------
        self.dwt = FRFTSplitter(M)
        self.idwt = FRFTMerger(M)

        # ----------------------------------------------------------------------
        # C. HYPER-PRIOR AUTOENCODER (Mamba-Enhanced)
        # ----------------------------------------------------------------------
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

        # ----------------------------------------------------------------------
        # D. PATH A: LOW FREQUENCY ENTROPY MODEL (MambaIC Checkerboard)
        # ----------------------------------------------------------------------
        self.atten_mean_lf = nn.ModuleList(
            SWAtten(
                (M + self.slice_ch_lf * min(i, 5)),
                (M + self.slice_ch_lf * min(i, 5)),
                16,
                self.window_size,
                0,
                inter_dim=128,
            )
            for i in range(self.num_slices)
        )
        self.atten_scale_lf = nn.ModuleList(
            SWAtten(
                (M + self.slice_ch_lf * min(i, 5)),
                (M + self.slice_ch_lf * min(i, 5)),
                16,
                self.window_size,
                0,
                inter_dim=128,
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

        # Mamba Checkerboard Core
        self.anchor_atten_mean_lf = nn.ModuleList(
            SWAtten(
                self.slice_ch_lf,
                self.slice_ch_lf,
                16,
                self.window_size,
                0,
                inter_dim=128,
            )
            for _ in range(self.num_slices)
        )
        self.anchor_atten_scale_lf = nn.ModuleList(
            SWAtten(
                self.slice_ch_lf,
                self.slice_ch_lf,
                16,
                self.window_size,
                0,
                inter_dim=128,
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

        # ----------------------------------------------------------------------
        # E. PATH B: HIGH FREQUENCY ENTROPY MODEL (Dictionary Attention)
        # ----------------------------------------------------------------------
        self.dict_dim = 32 * dict_head_num
        self.dt = nn.Parameter(
            torch.randn([dict_num, self.dict_dim]), requires_grad=True
        )

        self.dt_cross_attention_hf = nn.ModuleList(
            MultiScaleDictionaryCrossAttentionGLU(
                input_dim=M
                + 6 * M
                + self.slice_ch_hf * i,  # LF_hat + HF_Params(6M) + prev_slices
                output_dim=3 * M,
                head_num=dict_head_num,
                mlp_rate=4,
            )
            for i in range(self.num_slices)
        )

        self.cc_transforms_hf = nn.ModuleList(
            nn.Sequential(
                conv(
                    M + 6 * M + self.slice_ch_hf * i + 3 * M,
                    512,
                    stride=1,
                    kernel_size=3,
                ),  # Support + Dict Context
                nn.GELU(),
                conv(512, 256, stride=1, kernel_size=3),
                nn.GELU(),
                conv(
                    256, 2 * self.slice_ch_hf, stride=1, kernel_size=3
                ),  # Outputs: mu + scale
            )
            for i in range(self.num_slices)
        )

        self.lrp_transforms_hf = nn.ModuleList(
            nn.Sequential(
                conv(
                    M + 6 * M + self.slice_ch_hf * i + 3 * M + self.slice_ch_hf,
                    512,
                    stride=1,
                    kernel_size=3,
                ),
                nn.GELU(),
                conv(512, 256, stride=1, kernel_size=3),
                nn.GELU(),
                conv(256, self.slice_ch_hf, stride=1, kernel_size=3),
            )
            for i in range(self.num_slices)
        )
        self.gaussian_conditional_hf = GaussianConditional(None)

        # ----------------------------------------------------------------------
        # F. MAIN DECODER
        # ----------------------------------------------------------------------
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            ResidualBlockUpsample_wave(N, N, upsample=2, wavelet="haar"),
            deconv(N, 3, kernel_size=5, stride=2),
        )

    # ==========================================================================
    # FORWARD
    # ==========================================================================
    def forward(self, x):
        B, C, H, W = x.size()

        # 1. Encode & Split
        y = self.g_a(x)
        y_shape = y.shape[2:]

        # Output is [B, 4M, H/2, W/2]
        y_wave = self.dwt(y)
        
        # Split into Low and High fractional frequency components
        # y_lf: [B, M, H/2, W/2]
        # y_hf: [B, 3M, H/2, W/2]
        y_lf = y_wave[:, : self.M, :, :]
        y_hf = y_wave[:, self.M :, :, :]

        # 2. Hyper-prior
        z = self.h_a(y_wave)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
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
        params_hf = torch.cat([latent_scales_hf, latent_means_hf], dim=1)  # 6M channels

        # ----------------------------------------------------------------------
        # 3. PATH A: LOW FREQUENCY (MAMBAIC CHECKERBOARD)
        # ----------------------------------------------------------------------
        anchor = torch.zeros_like(y_lf).to(x.device)
        non_anchor = torch.zeros_like(y_lf).to(x.device)
        anchor[:, :, 0::2, 0::2] = y_lf[:, :, 0::2, 0::2]
        anchor[:, :, 1::2, 1::2] = y_lf[:, :, 1::2, 1::2]
        non_anchor[:, :, 0::2, 1::2] = y_lf[:, :, 0::2, 1::2]
        non_anchor[:, :, 1::2, 0::2] = y_lf[:, :, 1::2, 0::2]

        y_lf_slices = y_lf.chunk(self.num_slices, 1)
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

        y_lf_hat_slices = []
        y_lf_likelihood = []

        for i, y_slice in enumerate(y_lf_slices):
            # Prior refinement
            mean_support = torch.cat([latent_means_lf] + y_lf_hat_slices, dim=1)
            mean_support = self.atten_mean_lf[i](mean_support)
            mu = self.cc_mean_transforms_lf[i](mean_support)[
                :, :, : y_shape[0] // 2, : y_shape[1] // 2
            ]

            scale_support = torch.cat([latent_scales_lf] + y_lf_hat_slices, dim=1)
            scale_support = self.atten_scale_lf[i](scale_support)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, : y_shape[0] // 2, : y_shape[1] // 2
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            # Anchor Decoding
            y_anchor = anchor_split[i]
            means_anchor, scales_anchor = self.context_vss_lf[i](
                torch.cat([ctx_params_anchor_split[i], support], dim=1)
            ).chunk(2, 1)
            means_anchor = self.anchor_atten_mean_lf[i](means_anchor)
            scales_anchor = self.anchor_atten_scale_lf[i](scales_anchor)

            scales_hat_split = torch.zeros_like(y_anchor)
            means_hat_split = torch.zeros_like(y_anchor)
            scales_hat_split[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
            scales_hat_split[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
            means_hat_split[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
            means_hat_split[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]

            y_anchor_quantized = ste_round(y_anchor - means_anchor) + means_anchor
            y_anchor_quantized[:, :, 0::2, 1::2] = 0
            y_anchor_quantized[:, :, 1::2, 0::2] = 0

            # Non-Anchor Decoding
            masked_context = self.context_prediction_lf[i](y_anchor_quantized)
            means_non_anchor, scales_non_anchor = self.context_vss_lf[i](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, 1)
            means_non_anchor = self.anchor_atten_mean_lf[i](means_non_anchor)
            scales_non_anchor = self.anchor_atten_scale_lf[i](scales_non_anchor)

            scales_hat_split[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
            scales_hat_split[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
            means_hat_split[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
            means_hat_split[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]

            _, y_slice_likelihood = self.gaussian_conditional_lf(
                y_slice, scales_hat_split, means=means_hat_split
            )

            y_non_anchor = non_anchor_split[i]
            y_non_anchor_quantized = (
                ste_round(y_non_anchor - means_non_anchor) + means_non_anchor
            )
            y_non_anchor_quantized[:, :, 0::2, 0::2] = 0
            y_non_anchor_quantized[:, :, 1::2, 1::2] = 0

            # LRP
            y_hat_slice = y_anchor_quantized + y_non_anchor_quantized
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_lf_hat_slices.append(y_hat_slice)
            y_lf_likelihood.append(y_slice_likelihood)

        y_lf_hat = torch.cat(y_lf_hat_slices, dim=1)
        y_lf_likelihoods = torch.cat(y_lf_likelihood, dim=1)

        # ----------------------------------------------------------------------
        # 4. PATH B: HIGH FREQUENCY (DCAE DICTIONARY)
        # ----------------------------------------------------------------------
        y_hf_slices = y_hf.chunk(self.num_slices, 1)
        y_hf_hat_slices = []
        y_hf_likelihood = []
        dt_batch = self.dt.repeat([B, 1, 1])

        for i, y_slice in enumerate(y_hf_slices):
            query = torch.cat([y_lf_hat, params_hf] + y_hf_hat_slices, dim=1)
            dict_info = self.dt_cross_attention_hf[i](query, dt_batch)

            support = torch.cat([query, dict_info], dim=1)
            mu_scale = self.cc_transforms_hf[i](support)
            mu, scale = mu_scale.chunk(2, 1)

            _, y_slice_likelihood = self.gaussian_conditional_hf(
                y_slice, scale, means=mu
            )
            y_hat_slice = ste_round(y_slice - mu) + mu

            # LRP
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms_hf[i](lrp_support))
            y_hat_slice += lrp

            y_hf_hat_slices.append(y_hat_slice)
            y_hf_likelihood.append(y_slice_likelihood)

        y_hf_hat = torch.cat(y_hf_hat_slices, dim=1)
        y_hf_likelihoods = torch.cat(y_hf_likelihood, dim=1)

        # ----------------------------------------------------------------------
        # 5. MERGE & DECODE
        # ----------------------------------------------------------------------
        y_wave_hat = torch.cat([y_lf_hat, y_hf_hat], dim=1)
        y_tilde = self.idwt(y_wave_hat)
        x_hat = self.g_s(y_tilde)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y_lf": y_lf_likelihoods,
                "y_hf": y_hf_likelihoods,
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

        y_wave = self.dwt(y)
        H_wave, W_wave = y_wave.shape[2:]
        y_lf = y_wave[:, : self.M, :, :]
        y_hf = y_wave[:, self.M :, :, :]

        # Hyper-prior
        z = self.h_a(y_wave)
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
        # PATH A: LOW FREQUENCY (MAMBAIC CHECKERBOARD)
        # ----------------------------------------------------------------------
        y_lf_slices = y_lf.chunk(self.num_slices, 1)
        y_lf_hat_slices = []

        ctx_params_anchor_split = torch.split(
            torch.zeros(B, self.slice_ch_lf * 2 * self.num_slices, H_wave, W_wave).to(
                x.device
            ),
            [2 * self.slice_ch_lf for _ in range(self.num_slices)],
            1,
        )

        # 1. Setup LF Encoder
        cdf_lf = self.gaussian_conditional_lf.quantized_cdf.tolist()
        cdf_lengths_lf = (
            self.gaussian_conditional_lf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_lf = self.gaussian_conditional_lf.offset.reshape(-1).int().tolist()

        encoder_lf = BufferedRansEncoder()
        symbols_list_lf = []
        indexes_list_lf = []

        for i, y_slice in enumerate(y_lf_slices):
            # Prior refinement
            mean_support = torch.cat([latent_means_lf] + y_lf_hat_slices, dim=1)
            mean_support = self.atten_mean_lf[i](mean_support)
            mu = self.cc_mean_transforms_lf[i](mean_support)[:, :, :H_wave, :W_wave]

            scale_support = torch.cat([latent_scales_lf] + y_lf_hat_slices, dim=1)
            scale_support = self.atten_scale_lf[i](scale_support)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, :H_wave, :W_wave
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            # Anchor Encoding/Decoding
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

            symbols_list_lf.extend(y_anchor_symbols.reshape(-1).tolist())
            indexes_list_lf.extend(indexes_anchor.reshape(-1).tolist())

            # Local reconstruction for non-anchor masking context
            anchor_quantized = (
                ste_round(y_anchor_encode - means_anchor_encode) + means_anchor_encode
            )
            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                x.device
            )
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # Non-Anchor Encoding/Decoding
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

            symbols_list_lf.extend(y_non_anchor_symbols.reshape(-1).tolist())
            indexes_list_lf.extend(indexes_non_anchor.reshape(-1).tolist())

            non_anchor_quantized = (
                ste_round(y_non_anchor_encode - means_non_anchor_encode)
                + means_non_anchor_encode
            )

            y_non_anchor_quantized = torch.zeros_like(y_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            # Merge and LRP
            y_hat_slice = y_anchor_decode + y_non_anchor_quantized
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_lf_hat_slices.append(y_hat_slice)

        # Flush LF stream
        encoder_lf.encode_with_indexes(
            symbols_list_lf, indexes_list_lf, cdf_lf, cdf_lengths_lf, offsets_lf
        )
        y_lf_string = encoder_lf.flush()
        y_lf_hat = torch.cat(y_lf_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY (DCAE DICTIONARY)
        # ----------------------------------------------------------------------
        y_hf_slices = y_hf.chunk(self.num_slices, 1)
        y_hf_hat_slices = []
        dt_batch = self.dt.repeat([B, 1, 1])

        # 2. Setup HF Encoder
        cdf_hf = self.gaussian_conditional_hf.quantized_cdf.tolist()
        cdf_lengths_hf = (
            self.gaussian_conditional_hf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_hf = self.gaussian_conditional_hf.offset.reshape(-1).int().tolist()

        encoder_hf = BufferedRansEncoder()
        symbols_list_hf = []
        indexes_list_hf = []

        for i, y_slice in enumerate(y_hf_slices):
            query = torch.cat([y_lf_hat, params_hf] + y_hf_hat_slices, dim=1)
            dict_info = self.dt_cross_attention_hf[i](query, dt_batch)

            support = torch.cat([query, dict_info], dim=1)
            mu_scale = self.cc_transforms_hf[i](support)
            mu, scale = mu_scale.chunk(2, 1)

            indexes = self.gaussian_conditional_hf.build_indexes(scale)
            y_hf_symbols = self.gaussian_conditional_hf.quantize(y_slice, "symbols", mu)

            symbols_list_hf.extend(y_hf_symbols.reshape(-1).tolist())
            indexes_list_hf.extend(indexes.reshape(-1).tolist())

            y_hat_slice = ste_round(y_slice - mu) + mu

            # LRP
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms_hf[i](lrp_support))
            y_hat_slice += lrp

            y_hf_hat_slices.append(y_hat_slice)

        # Flush HF stream
        encoder_hf.encode_with_indexes(
            symbols_list_hf, indexes_list_hf, cdf_hf, cdf_lengths_hf, offsets_hf
        )
        y_hf_string = encoder_hf.flush()

        return {
            "strings": [[y_lf_string, y_hf_string], z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        # We packaged it as [[y_lf, y_hf], z]
        assert isinstance(strings, list) and len(strings) == 2
        y_strings, z_strings = strings[0], strings[1]
        y_lf_string, y_hf_string = y_strings[0], y_strings[1]

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
        # PATH A: LOW FREQUENCY (MAMBAIC CHECKERBOARD)
        # ----------------------------------------------------------------------
        y_lf_hat_slices = []

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
        decoder_lf.set_stream(y_lf_string)

        for i in range(self.num_slices):
            # Prior refinement
            mean_support = torch.cat([latent_means_lf] + y_lf_hat_slices, dim=1)
            mean_support = self.atten_mean_lf[i](mean_support)
            mu = self.cc_mean_transforms_lf[i](mean_support)[:, :, :H_wave, :W_wave]

            scale_support = torch.cat([latent_scales_lf] + y_lf_hat_slices, dim=1)
            scale_support = self.atten_scale_lf[i](scale_support)
            scale = self.cc_scale_transforms_lf[i](scale_support)[
                :, :, :H_wave, :W_wave
            ]

            support = (
                torch.cat([latent_means_lf, latent_scales_lf], dim=1)
                if i == 0
                else torch.cat([mu, scale, latent_means_lf, latent_scales_lf], dim=1)
            )

            # Anchor Decoding
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

            # Decoder returns zero-centered residues when paired with these offsets
            rv_anchor = decoder_lf.decode_stream(
                indexes_anchor.reshape(-1).tolist(), cdf_lf, cdf_lengths_lf, offsets_lf
            )
            rv_anchor = (
                torch.Tensor(rv_anchor).to(z_hat.device).reshape(indexes_anchor.size())
            )

            anchor_quantized = rv_anchor + means_anchor_encode

            y_anchor_decode = torch.zeros(B_anchor, C_anchor, H_anchor, W_anchor).to(
                z_hat.device
            )
            y_anchor_decode[:, :, 0::2, 0::2] = anchor_quantized[:, :, 0::2, :]
            y_anchor_decode[:, :, 1::2, 1::2] = anchor_quantized[:, :, 1::2, :]

            # Non-Anchor Decoding
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

            non_anchor_quantized = rv_non_anchor + means_non_anchor_encode

            y_non_anchor_quantized = torch.zeros_like(means_anchor)
            y_non_anchor_quantized[:, :, 0::2, 1::2] = non_anchor_quantized[
                :, :, 0::2, :
            ]
            y_non_anchor_quantized[:, :, 1::2, 0::2] = non_anchor_quantized[
                :, :, 1::2, :
            ]

            # Merge and LRP
            y_hat_slice = y_anchor_decode + y_non_anchor_quantized
            lrp = 0.5 * torch.tanh(
                self.lrp_transforms_lf[i](torch.cat([mean_support, y_hat_slice], dim=1))
            )
            y_hat_slice += lrp

            y_lf_hat_slices.append(y_hat_slice)

        y_lf_hat = torch.cat(y_lf_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # PATH B: HIGH FREQUENCY (DCAE DICTIONARY)
        # ----------------------------------------------------------------------
        y_hf_hat_slices = []
        dt_batch = self.dt.repeat([B, 1, 1])

        cdf_hf = self.gaussian_conditional_hf.quantized_cdf.tolist()
        cdf_lengths_hf = (
            self.gaussian_conditional_hf.cdf_length.reshape(-1).int().tolist()
        )
        offsets_hf = self.gaussian_conditional_hf.offset.reshape(-1).int().tolist()

        decoder_hf = RansDecoder()
        decoder_hf.set_stream(y_hf_string)

        for i in range(self.num_slices):
            query = torch.cat([y_lf_hat, params_hf] + y_hf_hat_slices, dim=1)
            dict_info = self.dt_cross_attention_hf[i](query, dt_batch)

            support = torch.cat([query, dict_info], dim=1)
            mu_scale = self.cc_transforms_hf[i](support)
            mu, scale = mu_scale.chunk(2, 1)

            indexes = self.gaussian_conditional_hf.build_indexes(scale)

            rv_hf = decoder_hf.decode_stream(
                indexes.reshape(-1).tolist(), cdf_hf, cdf_lengths_hf, offsets_hf
            )
            rv_hf = torch.Tensor(rv_hf).to(z_hat.device).reshape(indexes.size())

            y_hat_slice = rv_hf + mu

            # LRP
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            lrp = 0.5 * torch.tanh(self.lrp_transforms_hf[i](lrp_support))
            y_hat_slice += lrp

            y_hf_hat_slices.append(y_hat_slice)

        y_hf_hat = torch.cat(y_hf_hat_slices, dim=1)

        # ----------------------------------------------------------------------
        # MERGE & DECODE
        # ----------------------------------------------------------------------
        y_wave_hat = torch.cat([y_lf_hat, y_hf_hat], dim=1)
        y_tilde = self.idwt(y_wave_hat)
        x_hat = self.g_s(y_tilde).clamp_(0, 1)

        return {"x_hat": x_hat}