import contextlib
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

# ---------------------------------------------------------------------------
# Context manager for safe STE toggling
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def ste_mode(model: "WMDC"):
    """
    Temporarily enable straight-through estimation, restoring the previous
    state on exit — even if an exception is raised.

    Usage::

        with ste_mode(accelerator.unwrap_model(model)) as m:
            out = m(batch)
    """
    prev = model.use_ste
    model.use_ste = True
    try:
        yield model
    finally:
        model.use_ste = prev


# ---------------------------------------------------------------------------
# WMDC
# ---------------------------------------------------------------------------


class WMDC(CompressionModel):
    """
    Wavelet-Mamba Dictionary Compression model.

    Architecture overview
    ----------------------
    Encoder  g_a : image → latent y        (4× stride, 4 FreqDisentMamba blocks)
    Hyper    h_a : y → z                   (2× stride, 1 VSSBlock)
    Hyper-dec:    z_hat → scales, means    (shared trunk, split heads)
                  z_hat → dictionary dt    (QueryDictionaryGenerator)
    Slice loop:   num_slices autoregressive slices
                  - per-slice GatedMemoryUpdater
                  - per-slice UnifiedDictionaryAttention (EOT / softmax)
                  - cc_mean / cc_scale transforms
                  - Latent Residual Prediction (LRP) with softplus gate
    Decoder  g_s : y_hat → image           (4× stride, 4 FreqDisentMamba blocks)

    Parameters
    ----------
    N              : channels in hyper-encoder/decoder
    M              : total latent channels  (split into num_slices equal slices)
    num_slices     : number of autoregressive channel slices
    dict_head_num  : dict_dim = 32 × dict_head_num
    dict_num       : number of dictionary tokens
    routing_mode   : {'softmax', 'balanced_eot', 'unbalanced_eot'}
    ot_eps         : Sinkhorn entropic regularisation ε  (match train value at eval)
    sinkhorn_iters : total Sinkhorn iterations  (≥ 20 recommended for ε = 0.1)
    tv_weight      : spatial TV weight on transport plan P  (0 = disabled)
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

        # use_ste as a bool buffer so it is saved in state_dict and
        # automatically synchronised across DDP ranks via checkpoint loading.
        self.register_buffer("_use_ste_flag", torch.zeros(1, dtype=torch.bool))

        # Populated during eval forward/compress/decompress for analysis.
        # Each entry is a (B, HW, N) tensor for the corresponding slice.
        self.slice_attn_probs: list[torch.Tensor] = []

        # ── Encoder ───────────────────────────────────────────────────────────
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            conv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            conv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            conv(N, M, kernel_size=5, stride=2),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            deconv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            deconv(N, N, kernel_size=5, stride=2),
            FrequencyDisentangledMamba(N, drop_path=0.1),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        # ── Hyper-encoder ─────────────────────────────────────────────────────
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
            conv(N, 192, kernel_size=5, stride=2),
        )

        # ── Hyper-decoder (scale and mean share a trunk) ───────────────────────
        # Shared trunk processes z_hat once; two independent heads predict
        # latent_scales and latent_means respectively.  This halves the number
        # of VSSBlock parameters in the hyper-decoder and forces both predictors
        # to operate from the same representation, improving consistency.
        self.h_trunk = nn.Sequential(
            deconv(192, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
        )
        self.h_scale_head = deconv(N, M, kernel_size=5, stride=2)
        self.h_mean_head = deconv(N, M, kernel_size=5, stride=2)

        # ── Entropy models ────────────────────────────────────────────────────
        self.entropy_bottleneck = EntropyBottleneck(192)
        self.gaussian_conditional = GaussianConditional(None)

        # ── Dictionary generator ──────────────────────────────────────────────
        self.hyper_to_dict = QueryDictionaryGenerator(
            in_dim=192,
            dict_num=self.dict_num,
            dict_dim=self.dict_dim,
            num_heads=4,
        )

        # ── Spatially-adaptive KL mass strength predictors (per-slice) ────────
        #
        # ALWAYS register rho_predictors as an nn.ModuleList so that
        # state_dict keys are identical regardless of routing_mode.  This allows
        # any checkpoint to be loaded with strict=True under any routing mode.
        #
        # Slice i is conditioned on:
        #   hyper_prior  (2*M channels)
        #   + i decoded slices  (i * slice_ch channels)
        #
        # _compute_rho_spatial() returns None at runtime when
        # routing_mode != 'unbalanced_eot', so the eot_attention modules
        # simply ignore rho in those modes.
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
        # Zero-init output layer: rho starts at softplus(0) + bias = constant.
        # The 0.5 bias gives a reasonable initial ρ at the start of training.
        for predictor in self.rho_predictors:
            nn.init.zeros_(predictor[-1].weight)
            nn.init.constant_(predictor[-1].bias, 0.5)

        # ── Bootstrap memory state ─────────────────────────────────────────────
        # Projects the full hyper_prior (2*M channels) down to slice_ch channels
        # to initialise the memory state for slice 0.
        self.init_memory = nn.Conv2d(2 * M, self.slice_ch, kernel_size=3, padding=1)

        # ── Per-slice K/V projections ──────────────────────────────────────────
        # Independent linear projections keep each slice's dictionary view
        # disentangled, preventing a shared bottleneck across slices.
        self.k_projs = nn.ModuleList(
            [nn.Linear(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )

        # ── Per-slice EOT dictionary attention ────────────────────────────────
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

        # ── Gated memory updaters (num_slices − 1, last slice has no successor) ─
        self.memory_updaters = nn.ModuleList(
            [GatedMemoryUpdater(self.slice_ch) for _ in range(num_slices - 1)]
        )

        # ── Slice-specific context transforms ─────────────────────────────────
        # Input to each transform:
        #   query       = hyper_prior (2*M)  + memory (slice_ch)  = 2M + S
        #   dict_info   = M  (output of eot_attention)
        #   total       = 3M + S   =  shared_input_dim
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

        # ── Latent Residual Prediction (LRP) transforms ───────────────────────
        # Input: support (3M+S) + quantised y_hat_slice (S)  = 3M + 2S
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
        # Zero-init LRP output: identity mapping at start of training so
        # the model first learns the main RD objective before refining LRP.
        for transform in self.lrp_transforms:
            nn.init.zeros_(transform[-1].weight)
            nn.init.zeros_(transform[-1].bias)

        # Learnable softplus gate per slice: controls how much the LRP
        # residual is blended in.  Initialised at -2.25 → softplus ≈ 0.10,
        # so LRP starts with a small but non-zero contribution.
        self.lrp_scales = nn.ParameterList(
            [
                nn.Parameter(torch.full((1, self.slice_ch, 1, 1), -2.25))
                for _ in range(num_slices)
            ]
        )

    # =========================================================================
    # use_ste property
    # =========================================================================

    @property
    def use_ste(self) -> bool:
        """
        Whether to use straight-through estimator for quantisation.

        Backed by a registered bool buffer so the value is:
          - Saved and restored by torch.save / load_state_dict.
          - Automatically broadcast to all DDP ranks via checkpoint loading.
          - Never silently reset to False on checkpoint resume.
        """
        return bool(self._use_ste_flag.item())

    @use_ste.setter
    def use_ste(self, val: bool) -> None:
        self._use_ste_flag.fill_(int(val))

    # =========================================================================
    # Helpers
    # =========================================================================

    def update(self, scale_table=None, force: bool = False) -> bool:
        """
        Rebuild entropy coder CDFs.

        Scale table upper limit is set to 1024 (instead of the compressai
        default of 256) to support very low-bitrate compression where large
        Gaussian scales can appear.
        """
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

    def _hyper_decode(self, z_hat: torch.Tensor):
        """
        Shared hyper-decoder trunk → latent scale and mean maps.

        Returns
        -------
        latent_scales : (B, M, H_y, W_y)
        latent_means  : (B, M, H_y, W_y)
        """
        trunk = self.h_trunk(z_hat)
        return self.h_scale_head(trunk), self.h_mean_head(trunk)

    def _compute_rho_spatial(
        self,
        slice_idx: int,
        hyper_prior: torch.Tensor,
        decoded_slices: list,
    ) -> torch.Tensor | None:
        """
        Per-slice spatially-varying KL mass strength ρ(x).

        Returns None when routing_mode != 'unbalanced_eot' so that
        UnifiedDictionaryAttention simply ignores the rho argument.

        The predictor for slice i is conditioned on:
          - hyper_prior  : (B, 2*M, H, W)  — global image summary
          - decoded_slices[0:i] — growing context from previous slices

        This allows the routing budget to adapt as the image is progressively
        described by the autoregressive slice loop.

        Returns
        -------
        rho : (B, H, W)  strictly positive, or None
        """
        if self.routing_mode != "unbalanced_eot":
            return None

        if slice_idx == 0:
            context = hyper_prior
        else:
            context = torch.cat([hyper_prior] + decoded_slices, dim=1)

        # (B, 1, H, W) → softplus keeps rho positive, clamp enforces lower bound
        rho = F.softplus(self.rho_predictors[slice_idx](context))
        rho = rho.clamp(min=0.05) + 1e-4  # strictly positive
        return rho.squeeze(1)  # (B, H, W)

    # =========================================================================
    # Forward pass (training)
    # =========================================================================

    def forward(self, x: torch.Tensor) -> dict:
        """
        Training forward pass.

        Quantisation is approximated by additive uniform noise (standard
        compressai convention) unless use_ste=True, in which case
        straight-through rounding is used for both z and y_slice.

        Returns dict with keys:
            x_hat          : (B, 3, H, W)  reconstructed image
            likelihoods    : {'y': ..., 'z': ...}
            aux_loss       : scalar — entropy bottleneck CDF loss
            dispersion_loss: scalar — negative dictionary entropy (–H, bits)
        """
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        # Clear per-slice attention cache (populated in eval mode only)
        self.slice_attn_probs.clear()

        # ── Encode ────────────────────────────────────────────────────────────
        y = self.g_a(x)
        z = self.h_a(y)

        # Entropy bottleneck: additive noise relaxation or STE
        z_hat_soft, z_likelihoods = self.entropy_bottleneck(z)
        if self.training and self.use_ste:
            # STE: forward is hard rounding, backward is identity
            z_hat = torch.round(z) - z.detach() + z
        else:
            z_hat = z_hat_soft

        # ── Hyper-prior ───────────────────────────────────────────────────────
        dt = self.hyper_to_dict(z_hat)  # (B, N, dict_dim)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)  # (B, 2M, H, W)

        # ── Autoregressive slice loop ──────────────────────────────────────────
        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_likelihood: list[torch.Tensor] = []

        memory_state = self.init_memory(hyper_prior)  # (B, slice_ch, H, W)
        total_dispersion = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for i, y_slice in enumerate(y_slices):
            # ρ always computed; returns None for non-unbalanced modes.
            # This keeps rho_predictors in the gradient graph on all ranks
            # so DDP never encounters unused parameters (find_unused=False safe).
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            k_dict = self.k_projs[i](dt)  # (B, N, dict_dim)
            v_dict = self.v_projs[i](dt)  # (B, N, dict_dim)

            # Query = hyper_prior ⊕ memory  → (B, 2M+slice_ch, H, W)
            query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, disp_loss = self.eot_attentions[i](
                query,
                k_dict,
                v_dict,
                rho_spatial,
                calc_disp=self.training,
            )
            total_dispersion = total_dispersion + disp_loss / self.num_slices

            # Collect eval-mode attention maps
            if not self.training and self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            # ── Gaussian conditional ─────────────────────────────────────────
            support = torch.cat([query, dict_info], dim=1)  # (B, 3M+S, H, W)
            mu = self.cc_mean_transforms[i](support)  # (B, S, H, W)
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11)

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, means=mu
            )

            # ── LRP with STE proxy ───────────────────────────────────────────
            # During training we feed a hard-rounded proxy to the LRP network
            # so LRP learns to correct quantisation error rather than noise.
            # The STE trick (detach + add) keeps gradients flowing through the
            # soft y_hat_slice path for the entropy model.
            if self.training:
                y_hat_hard = torch.round(y_slice - mu) + mu
                y_hat_for_lrp = y_hat_hard.detach() - y_hat_slice.detach() + y_hat_slice
            else:
                y_hat_for_lrp = y_hat_slice

            lrp_support = torch.cat([support, y_hat_for_lrp], dim=1)
            residual = self.lrp_transforms[i](lrp_support)
            lrp_gate = F.softplus(self.lrp_scales[i])
            y_hat_slice_lrp = y_hat_for_lrp + lrp_gate * residual

            y_hat_slices.append(y_hat_slice_lrp)
            y_likelihood.append(y_slice_likelihood)

            # Update memory with the best available (LRP-corrected) signal.
            # Last slice has no successor so no update needed.
            if i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        # ── Decode ────────────────────────────────────────────────────────────
        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1))

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": torch.cat(y_likelihood, dim=1),
                "z": z_likelihoods,
            },
            "aux_loss": self.aux_loss(),
            "dispersion_loss": total_dispersion,
        }

    # =========================================================================
    # Compress
    # =========================================================================

    def compress(self, x: torch.Tensor) -> dict:
        """
        Lossless encode x to byte strings.

        The slice loop mirrors decompress() exactly:
          1. Decode z_strings → z_hat (same as decompress)
          2. For each slice: compute identical rho / query / dict_info / mu / scale
          3. Arithmetic encode y_slice
          4. Arithmetic decode y_hat_slice (so LRP and memory match decompress)

        Returns
        -------
        dict with:
            strings : [y_strings (list of lists), z_strings (list)]
            shape   : (H_z, W_z) — needed by decompress to reconstruct z_hat
        """
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        self.slice_attn_probs.clear()

        y = self.g_a(x)
        z = self.h_a(y)

        # Encode z with entropy bottleneck
        z_strings = self.entropy_bottleneck.compress(z)
        # Immediately decode to get the same z_hat as decompress() will use
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        dt = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_strings: list = []

        memory_state = self.init_memory(hyper_prior)

        for i, y_slice in enumerate(y_slices):
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
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11)

            # Arithmetic encode
            index = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, index, means=mu)
            y_strings.append(y_string)

            # Arithmetic decode to recover the exact quantised slice that
            # decompress() will reconstruct — required for LRP and memory
            # to be byte-identical between compress and decompress.
            y_hat_slice = self.gaussian_conditional.decompress(
                y_string, index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            residual = self.lrp_transforms[i](lrp_support)
            lrp_gate = F.softplus(self.lrp_scales[i])
            y_hat_slice_lrp = y_hat_slice + lrp_gate * residual

            y_hat_slices.append(y_hat_slice_lrp)

            if i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
        }

    # =========================================================================
    # Decompress
    # =========================================================================

    def decompress(self, strings: list, shape) -> dict:
        """
        Lossless decode byte strings back to a reconstructed image.

        Parameters
        ----------
        strings : [y_strings, z_strings]  — output of compress()
        shape   : (H_z, W_z)              — output of compress()

        Returns
        -------
        dict with key 'x_hat' : (B, 3, H, W)  in [0, 1]
        """
        y_strings, z_strings = strings[0], strings[1]

        self.slice_attn_probs.clear()

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        dt = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_hat_slices: list[torch.Tensor] = []
        memory_state = self.init_memory(hyper_prior)

        for i in range(self.num_slices):
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
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11)

            index = self.gaussian_conditional.build_indexes(scale)
            y_hat_slice = self.gaussian_conditional.decompress(
                y_strings[i], index, means=mu
            )

            lrp_support = torch.cat([support, y_hat_slice], dim=1)
            residual = self.lrp_transforms[i](lrp_support)
            lrp_gate = F.softplus(self.lrp_scales[i])
            y_hat_slice_lrp = y_hat_slice + lrp_gate * residual

            y_hat_slices.append(y_hat_slice_lrp)

            if i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
