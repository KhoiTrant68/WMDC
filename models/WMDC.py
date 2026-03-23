import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models import CompressionModel

from modules.dictionary_blocks import (
    QueryDictionaryGenerator,
    UnifiedDictionaryAttention,
)
from modules.utils import conv, deconv
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import FrequencyDisentangledMamba


# Helper function to solve the train/test quantization gap
def ste_round(x):
    """Straight-Through Estimator rounding."""
    return torch.round(x) - x.detach() + x


class WMDC(CompressionModel):
    def __init__(
        self,
        N=192,
        M=320,
        num_slices=5,
        dict_head_num=20,
        dict_num=128,
        routing_mode="unbalanced_eot",
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
        # Spatially-Adaptive Sinkhorn Epsilon Predictor
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

        #: Projections moved here to compute them ONCE globally
        self.k_proj = nn.Linear(self.dict_dim, self.dict_dim)
        self.v_proj = nn.Linear(self.dict_dim, self.dict_dim)

        self.eot_attention = UnifiedDictionaryAttention(
            input_dim=2 * M
            + self.slice_ch,  # hyper_prior (2M) + memory_state (slice_ch)
            output_dim=M,
            dict_num=self.dict_num,
            dict_dim=self.dict_dim,
            tau=0.5,
            iters=3,
            routing_mode=routing_mode,
        )

        # Use separate memory updaters per slice to increase capacity
        # and allow tracking unique residual evolutions across steps.
        # We only need (num_slices - 1) updaters.
        # The final slice does not have a subsequent slice to pass memory to!
        self.memory_updaters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        self.slice_ch * 2, self.slice_ch, kernel_size=3, padding=1
                    ),
                    nn.GELU(),
                    nn.Conv2d(self.slice_ch, self.slice_ch, kernel_size=3, padding=1),
                )
                for _ in range(num_slices - 1)
            ]
        )

        shared_input_dim = 3 * M + self.slice_ch  # query + dict_info

        # Unshared transforms per slice
        self.cc_mean_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(shared_input_dim, 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, self.slice_ch, 3, 1, 1),
                )
                for _ in range(num_slices)
            ]
        )

        self.cc_scale_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(shared_input_dim, 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, self.slice_ch, 3, 1, 1),
                )
                for _ in range(num_slices)
            ]
        )

        self.lrp_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(shared_input_dim + self.slice_ch, 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, self.slice_ch, 3, 1, 1),
                )
                for _ in range(num_slices)
            ]
        )

        # Zero-initialize the final convolution of LRP transforms to start as identity
        for transform in self.lrp_transforms:
            nn.init.zeros_(transform[-1].weight)
            nn.init.zeros_(transform[-1].bias)

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            # Explicit float32 prevents dtype mismatch if the default dtype
            # has been changed externally (e.g. torch.set_default_dtype(float64)).
            scale_table = torch.exp(
                torch.linspace(math.log(0.11), math.log(256), 64, dtype=torch.float32)
            )
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def aux_loss(self):
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _compute_spatial_epsilon(self, hyper_prior):
        """
        Compute spatially-adaptive Sinkhorn regularisation strength.

        clamp(min=0.01) prevents near-zero ε that makes Sinkhorn numerically
        degenerate (effectively zero-temperature hard assignment in 3 iterations).
        The +1e-4 offset guarantees strict positivity after the clamp.
        """
        return F.softplus(self.eps_predictor(hyper_prior)).clamp(min=0.01) + 1e-4

    def forward(self, x):
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(
                f"Model expects input dimensions divisible by 64. Got {x.shape}"
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        dt = self.hyper_to_dict(z_hat)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices, y_likelihood = [], []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        # Initialise as a zero tensor (not a Python float) so the returned
        # value is always a Tensor regardless of which branch executes.
        total_dispersion = torch.zeros(1, device=x.device)

        # Shared epsilon across slices (spatially adaptive, image-level).
        # clamp(min=0.01) added to prevent degenerate near-zero temperature.
        spatial_epsilon = self._compute_spatial_epsilon(hyper_prior)

        memory_state = self.init_memory(hyper_prior)

        k_dict = self.k_proj(dt)
        v_dict = self.v_proj(dt)

        for i, y_slice in enumerate(y_slices):
            query = torch.cat([hyper_prior, memory_state], dim=1)

            # Dispersion loss computed on Slice 0 only to avoid opposing
            # gradients from high-frequency slices diluting the dictionary.
            dict_info, disp_loss = self.eot_attention(
                query, k_dict, v_dict, spatial_epsilon, calc_disp=(i == 0)
            )

            if i == 0:
                total_dispersion = disp_loss

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = torch.clamp(self.cc_scale_transforms[i](support), min=0.11)

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, means=mu
            )
            y_likelihood.append(y_slice_likelihood)

            # Output-bound update (receives uniform noise for loss stability)
            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms[i](lrp_support)
            )
            y_hat_slices.append(y_hat_slice)

            # Memory-bound update uses strictly discrete variables via STE rounding
            # This perfectly bridges the train-test quantization gap.
            if i < self.num_slices - 1:
                # 1. Strip continuous noise, mimic exactly what happens in decompress()
                y_hat_discrete = ste_round(y_slice - mu) + mu

                # 2. Run the exact same LRP post-processing
                discrete_lrp_support = torch.cat([support, y_hat_discrete], dim=1)
                y_hat_discrete = y_hat_discrete + 0.5 * torch.tanh(
                    self.lrp_transforms[i](discrete_lrp_support)
                )

                state_input = torch.cat([memory_state, y_hat_discrete], dim=1)
                memory_state = memory_state + self.memory_updaters[i](state_input)

        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": torch.cat(y_likelihood, dim=1), "z": z_likelihoods},
            "aux_loss": self.aux_loss(),
            "dispersion_loss": total_dispersion,
        }

    def compress(self, x):
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(
                f"Model expects input dimensions divisible by 64. Got {x.shape}"
            )

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        dt = self.hyper_to_dict(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices, y_strings = [], []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        spatial_epsilon = self._compute_spatial_epsilon(hyper_prior)

        memory_state = self.init_memory(hyper_prior)
        k_dict = self.k_proj(dt)
        v_dict = self.v_proj(dt)

        for i, y_slice in enumerate(y_slices):
            query = torch.cat([hyper_prior, memory_state], dim=1)

            # Explicitly disable dispersion loss computation during compression
            dict_info, _ = self.eot_attention(
                query, k_dict, v_dict, spatial_epsilon, calc_disp=False
            )
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = torch.clamp(self.cc_scale_transforms[i](support), min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, index, means=mu)
            y_strings.append(y_string)

            y_hat_slice = self.gaussian_conditional.decompress(
                y_string, index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms[i](lrp_support)
            )
            y_hat_slices.append(y_hat_slice)

            if i < self.num_slices - 1:
                state_input = torch.cat([memory_state, y_hat_slice], dim=1)
                memory_state = memory_state + self.memory_updaters[i](state_input)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
        }

    def decompress(self, strings, shape):
        y_strings, z_strings = strings[0], strings[1]

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        dt = self.hyper_to_dict(z_hat)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_hat_slices = []
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        spatial_epsilon = self._compute_spatial_epsilon(hyper_prior)

        memory_state = self.init_memory(hyper_prior)
        k_dict = self.k_proj(dt)
        v_dict = self.v_proj(dt)

        for i in range(self.num_slices):
            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _ = self.eot_attention(
                query, k_dict, v_dict, spatial_epsilon, calc_disp=False
            )
            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = torch.clamp(self.cc_scale_transforms[i](support), min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_hat_slice = self.gaussian_conditional.decompress(
                y_strings[i], index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms[i](lrp_support)
            )
            y_hat_slices.append(y_hat_slice)

            if i < self.num_slices - 1:
                state_input = torch.cat([memory_state, y_hat_slice], dim=1)
                memory_state = memory_state + self.memory_updaters[i](state_input)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}
