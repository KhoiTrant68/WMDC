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


def _make_backbone(name: str, dim: int, drop_path: float = 0.1) -> nn.Module:
    """
    Construct an FDM-replacement block by name. Used by the Table 2
    backbone ablation. Falls back to the production FDM when name=='fdm'.
    """
    if name == "fdm":
        return FrequencyDisentangledMamba(dim, drop_path=drop_path)
    # Lazy import — the ablation_models package only needs to exist when
    # a non-FDM backbone is requested.
    from ablation_models.backbone_variants import build_backbone

    return build_backbone(name, dim=dim, drop_path=drop_path)


# ---------------------------------------------------------------------------
# Context manager for safe STE toggling
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def ste_mode(model: "WMDC"):
    """
    Temporarily enable straight-through estimation, restoring the previous
    state on exit — even if an exception is raised inside the block.

    Usage:

        with ste_mode(accelerator.unwrap_model(model)) as m:
            out_fwd  = m(batch)
            out_enc  = m.compress(batch)      # safe even if this raises
    """
    prev = model.use_ste
    model.use_ste = True
    try:
        yield model
    finally:
        model.use_ste = prev


# ---------------------------------------------------------------------------
# Legacy checkpoint migration helper  (FIX-N1)
# ---------------------------------------------------------------------------


def load_legacy_checkpoint(
    model: "WMDC", ckpt_path: str, device: str = "cpu"
) -> "WMDC":
    """
    Load a checkpoint from any of three historical state_dict layouts into
    the current model layout.

    Three layouts handled
    ---------------------
    1) ORIGINAL (pre-trunk-share):
       h_scale_s.{0,1,2}.*  +  h_mean_s.{0,1,2}.*

    2) TRANSITIONAL (shared trunk, symmetric heads):
       h_trunk.{0,1}.*  +  h_scale_head.{weight,bias}  +  h_mean_head.{weight,bias}
       (both heads were a single deconv)

    3) CURRENT (shared trunk, asymmetric scale head):
       h_trunk.{0,1}.*
       h_scale_head.0.*  (extra Conv 3x3, gives scale a larger receptive field)
       h_scale_head.2.*  (final deconv)
       h_mean_head.{weight,bias}  (single deconv, unchanged)

    Mapping rules (layout 1 → layout 3):
        h_trunk.0.*    ←  h_scale_s.0.*
        h_trunk.1.*    ←  h_scale_s.1.*
        h_scale_head.2.*  ←  h_scale_s.2.*       (final deconv slot)
        h_scale_head.0.*  ←  newly initialised   (no source in old ckpt)
        h_mean_head.*  ←  h_mean_s.2.*
        h_mean_s.{0,1}.*  → discarded
                          (trunk already populated from h_scale_s)

    Mapping rules (layout 2 → layout 3):
        h_scale_head.weight  →  h_scale_head.2.weight
        h_scale_head.bias    →  h_scale_head.2.bias
        (h_scale_head.0.* — the new conv — is left at random init)
        all other keys pass through unchanged
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    remap: dict = {}
    for k, v in sd.items():
        # ── Layout 1: original h_scale_s / h_mean_s ──────────────────────
        if k.startswith("h_scale_s."):
            suffix = k[len("h_scale_s.") :]  # e.g. "0.weight" or "2.bias"
            idx_str, _, rest = suffix.partition(".")
            idx = int(idx_str)
            if idx < 2:
                remap[f"h_trunk.{idx}.{rest}"] = v
            else:  # idx == 2 (final deconv)
                # In the new layout, scale's final deconv is at index 2 of
                # the Sequential (Conv → GELU → Deconv).
                remap[f"h_scale_head.2.{rest}"] = v
        elif k.startswith("h_mean_s."):
            suffix = k[len("h_mean_s.") :]
            idx_str, _, rest = suffix.partition(".")
            idx = int(idx_str)
            if idx == 2:
                # h_mean_head is a single Deconv (no Sequential wrap).
                remap[f"h_mean_head.{rest}"] = v
            # idx 0,1 discarded (trunk initialised from h_scale_s above)
        # ── Layout 2: transitional (h_scale_head as single Deconv) ───────
        elif k.startswith("h_scale_head."):
            tail = k[len("h_scale_head.") :]
            head = tail.split(".", 1)[0]
            if head in ("weight", "bias"):
                # Layout 2: the whole key is "h_scale_head.weight" /
                # "h_scale_head.bias" — no nested index. Migrate to slot 2.
                remap[f"h_scale_head.2.{tail}"] = v
            else:
                # Layout 3 (already current) — pass through untouched.
                remap[k] = v
        else:
            remap[k] = v

    missing, unexpected = model.load_state_dict(remap, strict=False)
    if missing:
        print(f"[load_legacy_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_legacy_checkpoint] Unexpected keys: {unexpected}")
    return model


# ---------------------------------------------------------------------------
# WMDC
# ---------------------------------------------------------------------------


class WMDC(CompressionModel):
    """
    Wavelet-Mamba Dictionary Compression model.

    Architecture overview
    ----------------------
    Encoder  g_a : image → latent y          (4× stride, 4 FreqDisentMamba blocks)
    Hyper    h_a : y → z                     (2× stride, 1 VSSBlock)
    Hyper-dec:    z_hat → scales, means      (shared trunk h_trunk, split heads)
                  z_hat → dictionary dt      (QueryDictionaryGenerator)
    Slice loop:   num_slices autoregressive slices
                  - per-slice GatedMemoryUpdater
                  - per-slice UnifiedDictionaryAttention (EOT / softmax)
                  - cc_mean / cc_scale transforms
                  - Latent Residual Prediction (LRP) with softplus gate
    Decoder  g_s : y_hat → image             (4× stride, 4 FreqDisentMamba blocks)

    Parameters
    ----------
    N              : channels in hyper-encoder/decoder
    M              : total latent channels (split into num_slices equal slices)
    num_slices     : number of autoregressive channel slices
    dict_head_num  : dict_dim = 32 × dict_head_num
    dict_num       : number of dictionary tokens
    routing_mode   : {'softmax', 'balanced_eot', 'unbalanced_eot'}
    marginal_div   : divergence for unbalanced_eot row/col marginals — 'kl' or 'tv'
    ot_eps         : Sinkhorn entropic regularisation ε
    sinkhorn_iters : total Sinkhorn iterations (≥ 20 recommended for ε = 0.1)
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
        marginal_div: str = "kl",
        ot_eps: float = 0.1,
        sinkhorn_iters: int = 20,
        tv_weight: float = 0.0,
        backbone: str = "fdm",
        use_dense_concat: bool = False,
        memory_init: str = "bootstrap",
    ):
        super().__init__()
        self.N = N
        self.M = M
        self.num_slices = num_slices
        self.slice_ch = M // num_slices
        self.dict_num = dict_num
        self.dict_dim = 32 * dict_head_num
        self.routing_mode = routing_mode
        self.marginal_div = marginal_div
        self.use_dense_concat = use_dense_concat
        self.memory_init = memory_init

        # DDP-safe, never silently reset on checkpoint resume.
        self.register_buffer("_use_ste_flag", torch.zeros(1, dtype=torch.bool))

        # Per-slice attention maps: populated in eval/compress/decompress.
        self.slice_attn_probs: list[torch.Tensor] = []

        # ── Encoder ───────────────────────────────────────────────────────────
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            _make_backbone(backbone, N, drop_path=0.1),
            conv(N, N, kernel_size=5, stride=2),
            _make_backbone(backbone, N, drop_path=0.1),
            conv(N, N, kernel_size=5, stride=2),
            _make_backbone(backbone, N, drop_path=0.1),
            conv(N, M, kernel_size=5, stride=2),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            _make_backbone(backbone, N, drop_path=0.1),
            deconv(N, N, kernel_size=5, stride=2),
            _make_backbone(backbone, N, drop_path=0.1),
            deconv(N, N, kernel_size=5, stride=2),
            _make_backbone(backbone, N, drop_path=0.1),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        # ── Hyper-encoder ─────────────────────────────────────────────────────
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
            conv(N, 192, kernel_size=5, stride=2),
        )

        # ── Hyper-decoder: shared trunk + two independent prediction heads ────
        # Sharing the trunk halves VSSBlock parameters and forces both the
        # scale and mean predictors to operate from the same representation.
        self.h_trunk = nn.Sequential(
            deconv(192, N, kernel_size=5, stride=2),
            VSSBlock(hidden_dim=N, drop_path=0.0),
        )
        self.h_scale_head = nn.Sequential(
            conv(N, N, kernel_size=3, stride=1),
            nn.GELU(),
            deconv(N, M, kernel_size=5, stride=2),
        )
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

        # ── Spatially-adaptive KL mass predictors ────────

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
        # Only created when stateful mode is used and bootstrap init is selected.
        # Dense-concat has no Markov memory; zero-init skips the learned projection.
        if not use_dense_concat and memory_init == "bootstrap":
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
        # Query input_dim varies by context variant:
        #   stateful:     2M + slice_ch  (hyper_prior + Markov memory)
        #   dense-concat: 2M + i*slice_ch (hyper_prior + i prev slices; 2M for i=0)
        _eot_in = [
            2 * M + (i * self.slice_ch if use_dense_concat else self.slice_ch)
            for i in range(num_slices)
        ]
        self.eot_attentions = nn.ModuleList(
            [
                UnifiedDictionaryAttention(
                    input_dim=_eot_in[i],
                    output_dim=M,
                    dict_num=self.dict_num,
                    dict_dim=self.dict_dim,
                    tau=0.5,
                    ot_eps=ot_eps,
                    iters=sinkhorn_iters,
                    routing_mode=routing_mode,
                    marginal_div=marginal_div,
                    tv_weight=tv_weight,
                )
                for i in range(num_slices)
            ]
        )

        # ── Gated memory updaters (num_slices − 1, last slice has no successor) ─
        # Only created in stateful mode; dense-concat has no memory to update.
        if not use_dense_concat:
            self.memory_updaters = nn.ModuleList(
                [GatedMemoryUpdater(self.slice_ch) for _ in range(num_slices - 1)]
            )

        # ── Slice-specific context transforms ─────────────────────────────────
        # cc input_dim[i]  = eot_input_dim[i] + M  (query + dict_info)
        # lrp input_dim[i] = cc_input_dim[i]  + slice_ch  (support + y_hat_slice)
        _cc_in = [_eot_in[i] + M for i in range(num_slices)]

        self.cc_mean_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(_cc_in[i], 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, self.slice_ch, 3, 1, 1),
                )
                for i in range(num_slices)
            ]
        )

        self.cc_scale_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(_cc_in[i], 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, self.slice_ch, 3, 1, 1),
                )
                for i in range(num_slices)
            ]
        )

        # ── Latent Residual Prediction (LRP) transforms ───────────────────────
        self.lrp_transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(_cc_in[i] + self.slice_ch, 128, 1),
                    nn.GELU(),
                    nn.Conv2d(128, 224, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(224, 128, 3, 1, 1),
                    nn.GELU(),
                    nn.Conv2d(128, self.slice_ch, 3, 1, 1),
                )
                for i in range(num_slices)
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
    ) -> torch.Tensor:
        """
        Per-slice spatially-varying KL mass strength ρ(x).

        The predictor for slice i is conditioned on:
          - hyper_prior  : (B, 2*M, H, W)  — global image summary
          - decoded_slices[0:i] — growing context from previous slices

        This allows the routing budget to adapt as the image is progressively
        described by the autoregressive slice loop.

        Returns
        -------
        rho_raw * 0  : in non-unbalanced modes — zero-valued tensor with live
                       grad_fn, so rho_predictors stay in the DDP backward graph.
                       Callers pass None to UnifiedDictionaryAttention instead.
        rho          : (B, H, W) strictly positive — in unbalanced_eot mode
        """
        context = (
            hyper_prior
            if slice_idx == 0
            else torch.cat([hyper_prior] + decoded_slices, dim=1)
        )

        # Always execute rho_predictors
        rho_raw = self.rho_predictors[slice_idx](context)  # (B, 1, H, W)

        if self.routing_mode != "unbalanced_eot":
            # Return a zero-valued tensor in the backward graph.
            # The caller multiplies this into total_dispersion to satisfy DDP.
            return rho_raw * 0.0

        rho = F.softplus(rho_raw).clamp(min=0.05) + 1e-4
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
            dispersion_loss: scalar — negative dictionary-routing entropy (−H, bits),
                             summed over slices and divided by num_slices.
                             Sign convention: minimising  dispersion_loss
                             ⟺ maximising H ⟺ uniform dictionary use.
            dict_penalty   : scalar ≥ 0 — pairwise off-diagonal cosine
                             similarity penalty on the dictionary tokens
                             (encourages diverse/orthogonal tokens).
                             Returned SEPARATELY from dispersion_loss because
                             the two have different sign conventions and very
                             different magnitudes; conflating them breaks the
                             EMA-based adaptive scaling in RateDistortionLoss.
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
        dt, dict_penalty = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)  # (B, 2M, H, W)

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_likelihood: list[torch.Tensor] = []

        if self.use_dense_concat:
            pass  # context is built from y_hat_slices directly in the loop
        elif self.memory_init == "bootstrap":
            memory_state = self.init_memory(hyper_prior)  # (B, slice_ch, H, W)
        else:
            memory_state = torch.zeros(
                hyper_prior.size(0), self.slice_ch,
                hyper_prior.size(2), hyper_prior.size(3),
                device=hyper_prior.device, dtype=hyper_prior.dtype,
            )

        # Routing entropy accumulator: ONLY −H from the per-slice attention.
        # dict_penalty is tracked separately and returned as its own key,
        # so RateDistortionLoss can apply an independent (non-EMA) weight to it.
        total_dispersion = torch.zeros((), device=x.device, dtype=x.dtype)

        for i, y_slice in enumerate(y_slices):
            rho_out = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            if self.routing_mode != "unbalanced_eot":
                # Add zero contribution to keep rho_predictors in the backward
                # graph for DDP correctness with find_unused_parameters=False.
                # rho_out is rho_raw * 0.0 in this branch (see _compute_rho_spatial),
                # so the numerical contribution is exactly zero.
                if self.training:
                    total_dispersion = total_dispersion + rho_out.sum()
                rho_spatial = None  # attention routing unchanged
            else:
                rho_spatial = rho_out

            k_dict = self.k_projs[i](dt)  # (B, N, dict_dim)
            v_dict = self.v_projs[i](dt)  # (B, N, dict_dim)

            # Query = hyper_prior ⊕ context
            if self.use_dense_concat:
                query = hyper_prior if i == 0 else torch.cat([hyper_prior] + y_hat_slices, dim=1)
            else:
                query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, disp_loss = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=self.training
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

            # Update memory for stateful mode; dense mode relies on y_hat_slices.
            if not self.use_dense_concat and i < self.num_slices - 1:
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
            "dispersion_loss": total_dispersion,  # only −H from routing
            "dict_penalty": dict_penalty,  # ≥ 0, separate signal
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

        dt, _ = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_strings: list = []

        if self.use_dense_concat:
            pass
        elif self.memory_init == "bootstrap":
            memory_state = self.init_memory(hyper_prior)
        else:
            memory_state = torch.zeros(
                hyper_prior.size(0), self.slice_ch,
                hyper_prior.size(2), hyper_prior.size(3),
                device=hyper_prior.device, dtype=hyper_prior.dtype,
            )

        for i, y_slice in enumerate(y_slices):
            rho_out = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)
            rho_spatial = rho_out if self.routing_mode == "unbalanced_eot" else None

            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)
            if self.use_dense_concat:
                query = hyper_prior if i == 0 else torch.cat([hyper_prior] + y_hat_slices, dim=1)
            else:
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

            if not self.use_dense_concat and i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

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

        dt, _ = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_hat_slices: list[torch.Tensor] = []

        if self.use_dense_concat:
            pass
        elif self.memory_init == "bootstrap":
            memory_state = self.init_memory(hyper_prior)
        else:
            memory_state = torch.zeros(
                hyper_prior.size(0), self.slice_ch,
                hyper_prior.size(2), hyper_prior.size(3),
                device=hyper_prior.device, dtype=hyper_prior.dtype,
            )

        for i in range(self.num_slices):
            rho_out = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)
            rho_spatial = rho_out if self.routing_mode == "unbalanced_eot" else None

            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)
            if self.use_dense_concat:
                query = hyper_prior if i == 0 else torch.cat([hyper_prior] + y_hat_slices, dim=1)
            else:
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

            if not self.use_dense_concat and i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
