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
from modules.wavelet_blocks import FrequencyDisentangledMamba, GatedMemoryUpdater


class WMDC(CompressionModel):
    """
    Wavelet-Mamba Dictionary Compression model.

    Architecture overview
    ----------------------
    Encoder  g_a : image → latent y  (4× stride via FrequencyDisentangledMamba)
    Hyper    h_a : y → z             (2× stride via HyperResBlock)
    Hyper-dec:    z_hat → scales, means, dictionary dt
    Slice loop:   5 autoregressive slices with per-slice EOT dictionary attention
    Decoder  g_s : y_hat → image reconstruction

    Args:
        N              : channels in hyper-encoder/decoder
        M              : latent channels (split into num_slices slices)
        num_slices     : number of autoregressive channel slices
        dict_head_num  : dict_dim = 32 × dict_head_num
        dict_num       : number of dictionary tokens
        routing_mode   : {'softmax', 'balanced_eot', 'unbalanced_eot'}
        ot_eps         : Sinkhorn entropic regularisation ε (match train value)
        sinkhorn_iters : Sinkhorn iterations (≥ 20 recommended for ε = 0.1)
        tv_weight      : spatial TV weight on transport plan P (0 = disabled)
    """

    def __init__(
        self,
        N: int = 192,
        M: int = 320,
        num_slices: int = 5,
        dict_head_num: int = 20,
        dict_num: int = 128,
        routing_mode: str = "unbalanced_eot",
        ot_eps: float = 0.1,
        sinkhorn_iters: int = 20,
        tv_weight: float = 0.0,
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.slice_ch = M // num_slices
        self.dict_num = dict_num
        self.dict_dim = 32 * dict_head_num
        self.routing_mode = routing_mode

        # Populated during eval forward/compress/decompress for analysis.
        # Each entry is (B, HW, N) for the corresponding slice.
        self.slice_attn_probs: list[torch.Tensor] = []

        # ── Encoder / Decoder ──────────────────────────────────────────────
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

        # ── Hyper-encoder / decoder ────────────────────────────────────────
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

        # ── Entropy models ─────────────────────────────────────────────────
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

        # ── Dictionary generator ───────────────────────────────────────────
        self.hyper_to_dict = QueryDictionaryGenerator(
            in_dim=192,
            dict_num=self.dict_num,
            dict_dim=self.dict_dim,
            num_heads=4,
        )

        # ── Spatially-Adaptive KL Mass Penalty Predictor (per-slice) ──────
        # Slice gets its own rho predictor conditioned on the growing
        # context (hyper_prior + accumulated reconstruction so far).
        # Slice i sees hyper_prior (2*M) + i decoded slices (i * slice_ch).
        self.rho_predictors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(2 * M + i * self.slice_ch, 32, 1),
                    nn.GELU(),
                    nn.Conv2d(32, 1, 1),
                )
                for i in range(num_slices)
            ]
        )
        # Bias init: softplus(2.0) ≈ 2.13 → rho starts moderately large,
        # meaning the OT starts close to the balanced (hard-constraint) regime
        # and can relax as training progresses.
        for predictor in self.rho_predictors:
            nn.init.zeros_(predictor[-1].weight)
            nn.init.constant_(predictor[-1].bias, 2.0)

        # ── Bootstrap memory state ─────────────────────────────────────────
        self.init_memory = nn.Conv2d(2 * M, self.slice_ch, kernel_size=3, padding=1)

        # ── Per-slice K/V projections ──────────────────────────────────────
        self.k_projs = nn.ModuleList(
            [nn.Linear(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )

        # Per-slice independent EOT attention to prevent query projection bottlenecks
        self.eot_attentions = nn.ModuleList(
            [
                UnifiedDictionaryAttention(
                    input_dim=2 * M + self.slice_ch,
                    output_dim=M,
                    dict_num=self.dict_num,
                    dict_dim=self.dict_dim,
                    tau=0.5,
                    ot_eps=ot_eps,
                    iters=sinkhorn_iters,
                    routing_mode=routing_mode,
                    tv_weight=tv_weight,
                )
                for _ in range(num_slices)
            ]
        )

        # ── GatedMemoryUpdater ─────────────────────────────────────────────
        self.memory_updaters = nn.ModuleList(
            [GatedMemoryUpdater(self.slice_ch) for _ in range(num_slices - 1)]
        )

        # ── Slice-specific transforms ──────────────────────────────────────
        shared_input_dim = 3 * M + self.slice_ch

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
        # Zero-init LRP final layer: identity at start of training.
        for transform in self.lrp_transforms:
            nn.init.zeros_(transform[-1].weight)
            nn.init.zeros_(transform[-1].bias)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            # Expand scale table upper limit to 1024 to support low-bitrate compression
            scale_table = torch.exp(
                torch.linspace(math.log(0.11), math.log(1024), 64, dtype=torch.float32)
            )
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def aux_loss(self) -> torch.Tensor:
        """Auxiliary loss for entropy bottleneck CDF approximation."""
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _compute_rho_spatial(
        self, slice_idx: int, hyper_prior: torch.Tensor, decoded_slices: list
    ) -> torch.Tensor:
        """
        Compute spatially-varying KL mass-conservation strength ρ(x) for a
        given slice.  The predictor for slice i is conditioned on the global
        hyper-prior (2*M channels) plus all previously decoded slices
        (i * slice_ch channels), so the routing budget adapts to how much of
        the image has already been described.

        Args:
            slice_idx     : index of the current slice (0 … num_slices-1)
            hyper_prior   : (B, 2*M, H, W)
            decoded_slices: list of previously decoded y_hat_slices,
                            each (B, slice_ch, H, W).  Length == slice_idx.

        Returns:
            rho : (B, H, W)  strictly positive
        """
        if slice_idx == 0:
            context = hyper_prior
        else:
            context = torch.cat([hyper_prior] + decoded_slices, dim=1)

        rho = F.softplus(self.rho_predictors[slice_idx](context))  # (B, 1, H, W)
        rho = rho.clamp(min=0.05) + 1e-4  # strictly positive
        return rho.squeeze(1)  # (B, H, W)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict:
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        # Clear per-slice attention storage (populated in eval mode below).
        self.slice_attn_probs.clear()

        y = self.g_a(x)
        z = self.h_a(y)

        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        dt = self.hyper_to_dict(z_hat)

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices: list = []
        y_likelihood: list = []

        memory_state = self.init_memory(hyper_prior)

        total_dispersion = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i, y_slice in enumerate(y_slices):
            # rho always flows gradient to rho_predictors regardless of mode.
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            # For non-unbalanced modes the attention ignores rho internally,
            # but computing it keeps rho_predictors in the gradient graph so
            # DDP never encounters truly unused parameters.
            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)

            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, disp_loss = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=True
            )
            if self.routing_mode != "unbalanced_eot" and self.training:
                disp_loss = disp_loss + 0.0 * rho_spatial.sum()
            total_dispersion = total_dispersion + disp_loss / self.num_slices

            if not self.training and self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[i](support)
            scale = torch.clamp(self.cc_scale_transforms[i](support), min=0.11)

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, means=mu
            )

            # Straight-Through Estimator (STE) for LRP module quantization gap
            if self.training:
                y_hat_hard = torch.round(y_slice - mu) + mu
                y_hat_for_lrp = y_hat_hard.detach() - y_hat_slice.detach() + y_hat_slice
            else:
                y_hat_for_lrp = y_hat_slice

            lrp_support = torch.cat([support, y_hat_for_lrp], dim=1)
            y_hat_slice_lrp = y_hat_for_lrp + 0.5 * torch.tanh(
                self.lrp_transforms[i](lrp_support)
            )

            # Append the LRP-corrected slice for downstream context and output.
            y_hat_slices.append(y_hat_slice_lrp)
            y_likelihood.append(y_slice_likelihood)

            # Memory update uses the LRP-corrected slice so that subsequent
            # slices benefit from the best available reconstruction signal.
            if i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1))

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": torch.cat(y_likelihood, dim=1), "z": z_likelihoods},
            "aux_loss": self.aux_loss(),
            "dispersion_loss": total_dispersion,
        }

    # -----------------------------------------------------------------------
    # Compress
    # -----------------------------------------------------------------------

    def compress(self, x: torch.Tensor) -> dict:
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        self.slice_attn_probs.clear()

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        dt = self.hyper_to_dict(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices: list = []
        y_strings: list = []

        memory_state = self.init_memory(hyper_prior)

        for i, y_slice in enumerate(y_slices):
            # Per-slice rho using context decoded so far (matches decompress)
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)

            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _ = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=False
            )

            if self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = torch.clamp(self.cc_scale_transforms[i](support), min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, index, means=mu)
            y_strings.append(y_string)

            # Decompress to get the quantised slice (identical to decompress())
            y_hat_slice = self.gaussian_conditional.decompress(
                y_string, index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice_lrp = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms[i](lrp_support)
            )
            y_hat_slices.append(y_hat_slice_lrp)

            if i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
        }

    # -----------------------------------------------------------------------
    # Decompress
    # -----------------------------------------------------------------------

    def decompress(self, strings: list, shape) -> dict:
        y_strings, z_strings = strings[0], strings[1]

        self.slice_attn_probs.clear()

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        dt = self.hyper_to_dict(z_hat)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_hat_slices: list = []
        memory_state = self.init_memory(hyper_prior)

        for i in range(self.num_slices):
            # Per-slice rho using context decoded so far (matches compress)
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)

            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _ = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=False
            )

            if self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            support = torch.cat([query, dict_info], dim=1)

            mu = self.cc_mean_transforms[i](support)
            scale = torch.clamp(self.cc_scale_transforms[i](support), min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_hat_slice = self.gaussian_conditional.decompress(
                y_strings[i], index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            y_hat_slice_lrp = y_hat_slice + 0.5 * torch.tanh(
                self.lrp_transforms[i](lrp_support)
            )
            y_hat_slices.append(y_hat_slice_lrp)

            if i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
