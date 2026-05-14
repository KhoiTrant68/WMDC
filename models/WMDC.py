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


def _make_backbone(
    name: str,
    dim: int,
    drop_path: float = 0.1,
    use_content_adaptive: bool = False,
    cluster_num: int = 8,
) -> nn.Module:
    """
    Construct an FDM-replacement block by name. Used by the Table 2
    backbone ablation. Falls back to the production FDM when name=='fdm'.
    """
    if name == "fdm":
        return FrequencyDisentangledMamba(
            dim,
            drop_path=drop_path,
            use_content_adaptive=use_content_adaptive,
            cluster_num=cluster_num,
        )
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
# Legacy checkpoint migration helper
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
        use_content_adaptive: bool = False,
        cluster_num: int = 8,
    ):
        super().__init__()
        if M % num_slices != 0:
            raise ValueError(f"M ({M}) must be divisible by num_slices ({num_slices}).")
        if routing_mode not in ("softmax", "balanced_eot", "unbalanced_eot"):
            raise ValueError(f"Unknown routing_mode: {routing_mode!r}")
        if memory_init not in ("bootstrap", "zero"):
            raise ValueError(f"Unknown memory_init: {memory_init!r}")
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
        self.use_content_adaptive = use_content_adaptive
        self.cluster_num = cluster_num

        # DDP-safe, never silently reset on checkpoint resume.
        self.register_buffer("_use_ste_flag", torch.zeros(1, dtype=torch.bool))

        # Per-slice attention maps: populated in eval/compress/decompress.
        self.slice_attn_probs: list[torch.Tensor] = []

        # ── Encoder ───────────────────────────────────────────────────────────
        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
            ),
            conv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
            ),
            conv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
            ),
            conv(N, M, kernel_size=5, stride=2),
        )

        # ── Decoder ───────────────────────────────────────────────────────────
        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
            ),
            deconv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
            ),
            deconv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
            ),
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
        # Only created when actually used (unbalanced_eot).  Other routing
        # modes leave `rho_predictors` as None so DDP can run with
        # find_unused_parameters=False.
        if routing_mode == "unbalanced_eot":
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
        else:
            self.rho_predictors = None

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

        Scale table covers [0.11, 256] in 64 log-spaced bins.  The forward,
        compress, and decompress paths all clamp scale to the same range, so
        every emitted Gaussian falls inside a table bin — preventing index
        saturation that previously inflated bpp at low λ.
        """
        if scale_table is None:
            scale_table = torch.exp(
                torch.linspace(math.log(0.11), math.log(256.0), 64, dtype=torch.float32)
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

    @staticmethod
    @torch.no_grad()
    def _compute_complexity(
        x: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        """
        Per-image content-complexity map at z-grid resolution, normalised
        to [0, 1].  Used as the *target* of the anti-leakage alignment
        regulariser — detached, no learnable parameters.

        Definition
        ----------
            edge(x)  =  ‖∇x‖₂  via 3×3 Sobel on the grey channel
            c       =  AvgPool_{H/Hz × W/Wz}( edge(x) )
            c       =  (c − min(c)) / (max(c) − min(c) + ε)

        Why edge-magnitude?  In a learned codec the rate is dominated by
        the high-frequency / textured regions; these are exactly the
        positions where the dictionary side-information should help most.
        The dictionary's per-pixel mass (row_mass) therefore must be
        positively correlated with this complexity proxy.

        Parameters
        ----------
        x           : (B, 3, H, W)
        target_size : (Hz, Wz)

        Returns
        -------
        (B, Hz * Wz) complexity map normalised per-image to [0, 1].
        """
        x_gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        gx = F.conv2d(x_gray, kx, padding=1)
        gy = F.conv2d(x_gray, ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)  # (B, 1, H, W)

        edge_z = F.adaptive_avg_pool2d(edge, target_size).flatten(1)  # (B, Hz*Wz)
        mn = edge_z.amin(dim=1, keepdim=True)
        mx = edge_z.amax(dim=1, keepdim=True)
        return (edge_z - mn) / (mx - mn + 1e-8)

    def _compute_rho_spatial(
        self,
        slice_idx: int,
        hyper_prior: torch.Tensor,
        decoded_slices: list,
    ) -> torch.Tensor | None:
        """
        Per-slice spatially-varying KL mass strength ρ(x).

        In non-unbalanced routing modes the predictor module is not created at
        all (`self.rho_predictors is None`) and this function returns None.
        Callers pass None to UnifiedDictionaryAttention which then skips the
        unbalanced branch entirely.

        For unbalanced_eot, the predictor for slice i is conditioned on:
          - hyper_prior  : (B, 2*M, H, W)  — global image summary
          - decoded_slices[0:i] — growing context from previous slices

        Returns (B, H, W) strictly positive ρ.
        """
        if self.rho_predictors is None:
            return None

        context = (
            hyper_prior
            if slice_idx == 0
            else torch.cat([hyper_prior] + decoded_slices, dim=1)
        )
        rho_raw = self.rho_predictors[slice_idx](context)  # (B, 1, H, W)
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
            x_hat              : (B, 3, H, W)  reconstructed image
            likelihoods        : {'y': ..., 'z': ...}
            aux_loss           : scalar — entropy bottleneck CDF loss
            column_neg_entropy : scalar −H_col (bits), averaged over slices.
                                 Minimise → maximise H_col (anti-dead-code).
            row_entropy        : scalar H_row (bits, ≥ 0), averaged over slices.
                                 Minimise → sparse per-pixel selection.
            row_mass           : (B, num_slices, Hz·Wz) per-pixel row marginal,
                                 grad-bearing.  Identically 1 for non-unbalanced
                                 modes (alignment loss is then a no-op).
            complexity         : (B, Hz·Wz) detached content-complexity target
                                 for the anti-leakage alignment regulariser.
            dict_penalty       : scalar ≥ 0 — off-diagonal cosine similarity on
                                 the dictionary tokens (token diversity).
            tv_loss            : scalar ≥ 0 — spatial-TV regulariser on P,
                                 averaged over slices (already weight-scaled).

        Loss-side recipe (in RateDistortionLoss)
        ----------------------------------------
            L = λ·D + R
              + β_col · column_neg_entropy        (= −β_col · H_col)
              + β_row · row_entropy               (= +β_row · H_row)
              + γ · ReLU( −Pearson(row_mass, complexity) )
              + δ · dict_penalty
              + tv_loss
        """
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        # Clear per-slice attention cache (populated in eval mode only)
        self.slice_attn_probs.clear()

        # ── Encode ────────────────────────────────────────────────────────────
        y = self.g_a(x)
        z = self.h_a(y)

        # Entropy bottleneck: additive noise relaxation or STE.
        # In STE mode the forward value is rounded AROUND THE MEDIANS so that
        # train forward and compress/decompress see the same quantisation grid.
        # (The previous version rounded around zero — a 1.5 dB train/infer
        # gap source on its own when medians drifted away from 0.)
        z_hat_soft, z_likelihoods = self.entropy_bottleneck(z)
        if self.training and self.use_ste:
            medians = self.entropy_bottleneck._get_medians()  # (C, 1, 1)
            medians = medians.reshape(1, -1, 1, 1)  # broadcast over (B,H,W)
            z_round = torch.round(z - medians) + medians
            z_hat = z_round.detach() - z.detach() + z  # STE
        else:
            z_hat = z_hat_soft

        # ── Hyper-prior ───────────────────────────────────────────────────────
        dt, dict_penalty = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)  # (B, 2M, H, W)
        Hz, Wz = hyper_prior.shape[-2:]

        # ── Content complexity proxy (for anti-leakage alignment) ─────────────
        # Detached, no params; computed once at z-grid resolution.
        complexity = self._compute_complexity(x, (Hz, Wz))  # (B, Hz*Wz)

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_likelihood: list[torch.Tensor] = []

        if self.use_dense_concat:
            pass  # context is built from y_hat_slices directly in the loop
        elif self.memory_init == "bootstrap":
            memory_state = self.init_memory(hyper_prior)  # (B, slice_ch, H, W)
        else:
            memory_state = torch.zeros(
                hyper_prior.size(0),
                self.slice_ch,
                hyper_prior.size(2),
                hyper_prior.size(3),
                device=hyper_prior.device,
                dtype=hyper_prior.dtype,
            )

        # Per-slice accumulators for the routing-entropy signals.
        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        total_col_neg_H = zero
        total_row_H = zero
        total_tv = zero
        row_mass_list: list[torch.Tensor] = []

        for i, y_slice in enumerate(y_slices):
            # rho_spatial is None for softmax/balanced modes (rho_predictors
            # does not exist); UnifiedDictionaryAttention handles None.
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            k_dict = self.k_projs[i](dt)  # (B, N, dict_dim)
            v_dict = self.v_projs[i](dt)  # (B, N, dict_dim)

            # Query = hyper_prior ⊕ context
            if self.use_dense_concat:
                query = (
                    hyper_prior
                    if i == 0
                    else torch.cat([hyper_prior] + y_hat_slices, dim=1)
                )
            else:
                query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, attn_aux = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=self.training
            )

            # Accumulate routing-entropy signals (per-slice averages).
            total_col_neg_H = total_col_neg_H + attn_aux["column_neg_entropy"]
            total_row_H = total_row_H + attn_aux["row_entropy"]
            total_tv = total_tv + attn_aux["tv_loss"]
            row_mass_list.append(attn_aux["row_mass"])  # (B, Hz*Wz)

            # Collect eval-mode attention maps
            if not self.training and self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            # ── Gaussian conditional ─────────────────────────────────────────
            support = torch.cat([query, dict_info], dim=1)  # (B, 3M+S, H, W)
            mu = self.cc_mean_transforms[i](support)  # (B, S, H, W)
            # Clamp scale to BOTH ends of the entropy-coder scale_table so the
            # estimated rate matches the bytes the coder will actually emit
            # (saturating to the largest table index used to silently inflate
            # bpp at low-rate λ).
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11, max=256.0)

            if self.training and self.use_ste:
                # STE-y: forward is hard-rounded, gradient flows through y_slice.
                # Likelihood is computed on the hard-rounded value (= what the
                # arithmetic coder will see), closing the rate side of the
                # train/infer gap.
                y_hat_hard = torch.round(y_slice - mu) + mu
                y_slice_likelihood = self.gaussian_conditional._likelihood(
                    y_hat_hard, scale, means=mu
                )
                if self.gaussian_conditional.use_likelihood_bound:
                    y_slice_likelihood = (
                        self.gaussian_conditional.likelihood_lower_bound(
                            y_slice_likelihood
                        )
                    )
                y_hat_slice = y_hat_hard.detach() - y_slice.detach() + y_slice
            else:
                y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                    y_slice, scale, means=mu
                )

            # ── LRP with STE proxy ───────────────────────────────────────────
            # The LRP network expects the hard-rounded value (what the codec
            # produces) — keep this regardless of self.use_ste so the LRP
            # supervision target is consistent across training regimes.
            if self.training:
                y_hat_hard = torch.round(y_slice - mu) + mu
                y_hat_for_lrp = y_hat_hard.detach() - y_hat_slice.detach() + y_hat_slice
            else:
                # Match compress/decompress: feed hard-rounded value to LRP so
                # that forward(eval) metrics align with the actual codec path.
                y_hat_for_lrp = torch.round(y_slice - mu) + mu

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
        # Clamp to [0, 1] in BOTH training and eval so distortion is measured
        # consistently with eval.py / compress→decompress.  Gradient flows
        # through clamp on the non-saturated interior (vanishes outside [0,1],
        # which is the intended behaviour for an RGB codec).
        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1)).clamp(0.0, 1.0)

        # ── Slice-averaged entropy signals + stacked row_mass ────────────────
        S = float(self.num_slices)
        column_neg_entropy = total_col_neg_H / S
        row_entropy = total_row_H / S
        tv_loss = total_tv / S
        row_mass = torch.stack(row_mass_list, dim=1)  # (B, S, Hz*Wz)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": torch.cat(y_likelihood, dim=1),
                "z": z_likelihoods,
            },
            "aux_loss": self.aux_loss(),
            "column_neg_entropy": column_neg_entropy,
            "row_entropy": row_entropy,
            "row_mass": row_mass,
            "complexity": complexity,
            "dict_penalty": dict_penalty,
            "tv_loss": tv_loss,
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
                hyper_prior.size(0),
                self.slice_ch,
                hyper_prior.size(2),
                hyper_prior.size(3),
                device=hyper_prior.device,
                dtype=hyper_prior.dtype,
            )

        for i, y_slice in enumerate(y_slices):
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)
            if self.use_dense_concat:
                query = (
                    hyper_prior
                    if i == 0
                    else torch.cat([hyper_prior] + y_hat_slices, dim=1)
                )
            else:
                query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _aux = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=False
            )

            if self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11, max=256.0)

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
                hyper_prior.size(0),
                self.slice_ch,
                hyper_prior.size(2),
                hyper_prior.size(3),
                device=hyper_prior.device,
                dtype=hyper_prior.dtype,
            )

        for i in range(self.num_slices):
            rho_spatial = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            k_dict = self.k_projs[i](dt)
            v_dict = self.v_projs[i](dt)
            if self.use_dense_concat:
                query = (
                    hyper_prior
                    if i == 0
                    else torch.cat([hyper_prior] + y_hat_slices, dim=1)
                )
            else:
                query = torch.cat([hyper_prior, memory_state], dim=1)

            dict_info, _aux = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=False
            )

            if self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11, max=256.0)

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
