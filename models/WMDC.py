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
from modules.utils import OLP, conv, deconv
from modules.VSS_module import VSSBlock
from modules.wavelet_blocks import (
    WLS,
    FrequencyDisentangledMamba,
    GatedMemoryUpdater,
    iWLS,
)


def _make_backbone(
    name: str,
    dim: int,
    drop_path: float = 0.1,
    use_content_adaptive: bool = False,
    cluster_num: int = 8,
    use_layer_scale: bool = True,
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
            use_layer_scale=use_layer_scale,
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

    # Module paths whose nn.Linear became OLP (which wraps a .linear).
    # If the old key ends at the Linear itself (no .linear), prepend .linear.
    olp_prefixes = (
        "k_projs.",
        "v_projs.",
        "hyper_to_dict.proj.0.",
        "hyper_to_dict.proj.2.",
    )

    def _migrate_to_olp(key: str) -> str:
        """For old keys like 'k_projs.3.weight' → 'k_projs.3.linear.weight'."""
        for pref in olp_prefixes:
            if not key.startswith(pref):
                continue
            tail = key[len(pref) :]
            # k_projs.X.{weight,bias}: insert 'linear.' after the numeric index.
            if pref in ("k_projs.", "v_projs."):
                idx, _, rest = tail.partition(".")
                if rest in ("weight", "bias"):
                    return f"{pref}{idx}.linear.{rest}"
            # hyper_to_dict.proj.X.{weight,bias}: insert 'linear.' before suffix.
            elif tail in ("weight", "bias"):
                return f"{pref}linear.{tail}"
        return key

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

    # ── OLP migration: nn.Linear → OLP(.linear) for dict projections ─────
    # Apply AFTER the layout-1/2 mapping so it catches both legacy and current
    # keys that point at projections we've since wrapped in OLP.
    remap = {_migrate_to_olp(k): v for k, v in remap.items()}

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
        use_wls_shortcut: bool = False,
        use_conditional_marginals: bool = False,
        cond_alpha: float = 0.5,

        use_adaptive_eps: bool = True,
        image_conditional_range: bool = True,
        # ── ε-scaled OT router + weighted dict + bounded dict_info ─────
        use_eps_scaling: bool = True,
        eps_scaling_levels: int = 5,
        use_weighted_dict: bool = True,
        # ── LayerScale residual on FrequencyDisentangledMamba ──────────
        use_layer_scale: bool = True,
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
        self.use_content_adaptive = use_content_adaptive
        self.cluster_num = cluster_num
        self.use_wls_shortcut = use_wls_shortcut
        # ── Conditional multi-marginal OT across slices ──────────────────
        # When enabled, the column target marginal b for slice i is shifted
        # by the cumulative column usage of slices 0..i-1, so later slices
        # avoid dictionary atoms heavily used by earlier ones.  Theoretical
        # rationale: the dictionary is a shared budget across the
        # autoregressive slice loop; cross-slice specialisation should be
        # enforced by the routing layer itself, not just by independent
        # per-slice K/V projections.  See THEORY.md §2 for the derivation.
        self.use_conditional_marginals = use_conditional_marginals
        self.cond_alpha = float(cond_alpha)
        # Routing modes that have a column-marginal concept.  Softmax does
        # not, so the conditional override is a no-op there.
        self._cond_active = use_conditional_marginals and routing_mode in {
            "balanced_eot",
            "unbalanced_eot",
        }

        # ── C2: adaptive-eps master switch ───────────────────────────────
        self.use_adaptive_eps = bool(use_adaptive_eps)
        # ── ε-scaling / weighted-dict / LayerScale flags ─────────────────
        self.use_eps_scaling = bool(use_eps_scaling)
        self.eps_scaling_levels = int(eps_scaling_levels)
        self.use_weighted_dict = bool(use_weighted_dict)
        self.use_layer_scale = bool(use_layer_scale)
        # ── C1: image-conditional adaptive_range bias ───────────────────
        # When enabled, a per-slice tiny MLP predicts a scalar bias added
        # to that slice's `log_adaptive_range` from a global summary of
        # the hyperprior.  Lets simple images use a tighter spread (closer
        # to scalar eps) and complex images a wider spread.  Wired only
        # if adaptive eps is on — pointless otherwise.
        self.image_conditional_range = (
            bool(image_conditional_range) and self.use_adaptive_eps
        )

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
                use_layer_scale=self.use_layer_scale,
            ),
            conv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
                use_layer_scale=self.use_layer_scale,
            ),
            conv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
                use_layer_scale=self.use_layer_scale,
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
                use_layer_scale=self.use_layer_scale,
            ),
            deconv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
                use_layer_scale=self.use_layer_scale,
            ),
            deconv(N, N, kernel_size=5, stride=2),
            _make_backbone(
                backbone,
                N,
                drop_path=0.1,
                use_content_adaptive=use_content_adaptive,
                cluster_num=cluster_num,
                use_layer_scale=self.use_layer_scale,
            ),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        # ── WLS / iWLS multi-scale shortcuts (CMIC-style, opt-in) ─────────────
        # Adds a wavelet-domain auxiliary path that injects detail at every
        # encoder/decoder scale.  Channel-matched to g_a/g_s downsample stages.
        if use_wls_shortcut:
            self.aux_enc = nn.ModuleList([WLS(3, N), WLS(N, N), WLS(N, N), WLS(N, M)])
            self.aux_dec = nn.ModuleList(
                [iWLS(M, N), iWLS(N, N), iWLS(N, N), iWLS(N, 3)]
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
            [OLP(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )
        self.v_projs = nn.ModuleList(
            [OLP(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
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
                    use_adaptive_eps=self.use_adaptive_eps,
                    use_eps_scaling=self.use_eps_scaling,
                    eps_scaling_levels=self.eps_scaling_levels,
                    use_weighted_dict=self.use_weighted_dict,
                )
                for i in range(num_slices)
            ]
        )

        # ── C1: per-slice range-bias predictors ─────────────────────────
        # Tiny MLP from pooled hyperprior summary → scalar log-range bias.
        # hyper_prior is cat([latent_scales, latent_means]) with shape
        # (B, 2*M, Hz, Wz) — so the global-avg-pool summary is (B, 2*M).
        # Output clamped via tanh(±1.0) so the per-image bias never
        # overrides the global log_adaptive_range parameter.
        if self.image_conditional_range:
            self.range_bias_predictors = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(2 * M, 32),
                        nn.GELU(),
                        nn.Linear(32, 1),
                    )
                    for _ in range(num_slices)
                ]
            )
        else:
            self.range_bias_predictors = None

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

    def ortho_loss(self) -> torch.Tensor:
        """Aggregate ‖W Wᵀ − I‖² over every OLP module (CMIC-style)."""
        terms = [m.loss() for m in self.modules() if isinstance(m, OLP)]
        if not terms:
            return torch.zeros((), device=next(self.parameters()).device)
        return torch.stack(terms).mean()

    # =========================================================================
    # B2 — Dead-atom revival (called periodically from the training loop)
    # =========================================================================

    @torch.no_grad()
    def revive_dead_atoms(self, agreement: int | None = None) -> int:
        """
        Aggregate per-slice dead-atom masks; atoms flagged dead by at
        least `agreement` slices are revived in QueryDictionaryGenerator.

        Parameters
        ----------
        agreement : int | None
            How many slices must agree an atom is dead before revival.
            Default = num_slices (atom must be dead EVERYWHERE — most
            conservative).  Set to e.g. num_slices // 2 + 1 for majority
            voting (more aggressive revival).

        Returns the number of atoms revived.

        After revival, every slice's column-usage EMA is reset to give
        the new atoms a fresh start in the dead-detector accounting.
        """
        if agreement is None:
            agreement = self.num_slices
        masks = []
        for attn in self.eot_attentions:
            masks.append(attn.dead_atom_mask())
        if not masks:
            return 0
        stacked = torch.stack(masks, dim=0).int().sum(dim=0)  # (N,)
        global_dead = stacked >= int(agreement)
        n_revived = self.hyper_to_dict.revive_queries(global_dead)
        if n_revived > 0:
            for attn in self.eot_attentions:
                attn.reset_col_usage_ema()
        return int(n_revived)

    # =========================================================================
    # B3 — Per-slice k_dict coherence loss
    # =========================================================================

    def slice_coherence_loss(self) -> torch.Tensor:
        """Sum of k_dict_i coherence losses from the LAST forward pass.

        Populated by `forward()` via `self._last_slice_coherence`.  Returns
        zero before any forward call (e.g., during model construction tests).
        """
        v = getattr(self, "_last_slice_coherence", None)
        if v is None:
            return torch.zeros((), device=next(self.parameters()).device)
        return v

    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        """Run g_a, optionally with WLS multi-scale shortcuts.

        g_a layout: [down0, bb0, down1, bb1, down2, bb2, down3].  Aux WLS
        path is added AFTER each downsample (so spatial size matches).
        """
        if not self.use_wls_shortcut:
            return self.g_a(x)
        aux = x
        h = self.g_a[0](x)
        aux = self.aux_enc[0](aux)
        h = h + aux
        h = self.g_a[1](h)
        h = self.g_a[2](h)
        aux = self.aux_enc[1](aux)
        h = h + aux
        h = self.g_a[3](h)
        h = self.g_a[4](h)
        aux = self.aux_enc[2](aux)
        h = h + aux
        h = self.g_a[5](h)
        h = self.g_a[6](h)
        aux = self.aux_enc[3](aux)
        h = h + aux
        return h

    def _decode_latent(self, y_hat: torch.Tensor) -> torch.Tensor:
        """Run g_s, optionally with iWLS multi-scale shortcuts.

        g_s layout: [up0, bb0, up1, bb1, up2, bb2, up3].  Aux iWLS path
        starts from y_hat and is added AFTER each up-sample stage.
        """
        if not self.use_wls_shortcut:
            return self.g_s(y_hat)
        aux = y_hat
        h = self.g_s[0](y_hat)
        aux = self.aux_dec[0](aux)
        h = h + aux
        h = self.g_s[1](h)
        h = self.g_s[2](h)
        aux = self.aux_dec[1](aux)
        h = h + aux
        h = self.g_s[3](h)
        h = self.g_s[4](h)
        aux = self.aux_dec[2](aux)
        h = h + aux
        h = self.g_s[5](h)
        h = self.g_s[6](h)
        aux = self.aux_dec[3](aux)
        h = h + aux
        return h

    def set_dict_eps_anneal_bias(self, value: float) -> None:
        """
        Broadcast a warm-up bias on log_eps to every slice's dictionary
        attention.  See `UnifiedDictionaryAttention._log_eps_bias` for the
        rationale; called by the training loop once per epoch.
        """
        for attn in self.eot_attentions:
            attn.set_log_eps_bias(value)

    def last_dt(self) -> torch.Tensor | None:
        """Return the most recent dictionary frame `dt` (B, N, D) computed by
        `hyper_to_dict` during forward/compress/decompress.  None until the
        first forward.  Used by Prop. B coherence telemetry."""
        return getattr(self, "_last_dt", None)

    def coherence_stats(self) -> dict | None:
        """Convenience wrapper: returns Welch-bound / tight-frame stats of
        the last `dt`.  None if forward has not been called.  Forwards to
        `QueryDictionaryGenerator.coherence_stats`."""
        dt = self.last_dt()
        if dt is None:
            return None
        return QueryDictionaryGenerator.coherence_stats(dt)

    def sinkhorn_telemetry(self) -> dict:
        """
        Aggregate Sinkhorn-stability telemetry across all slice attentions.

        Returns a dict suitable for both logging and inclusion in the paper's
        stability section.  `fallback_rate` is the fraction of forward calls
        across all slices that triggered the softmax fallback — reviewers ask
        for this number directly, so it lives next to the model rather than
        scattered across attention modules.
        """
        total_calls = 0
        total_fb = 0
        per_slice: list[dict] = []
        for i, attn in enumerate(self.eot_attentions):
            t = attn.sinkhorn_telemetry()
            per_slice.append({"slice": i, **t})
            total_calls += t["calls"]
            total_fb += t["fallbacks"]
        return {
            "total_calls": total_calls,
            "total_fallbacks": total_fb,
            "fallback_rate": total_fb / max(total_calls, 1),
            "per_slice": per_slice,
        }

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
        # Robust percentile normalisation (5 % – 95 %).  A single bright
        # pixel can dominate amin/amax and flatten the rest of the map; the
        # quantile-based version is invariant to <5 % outliers and keeps the
        # bulk of the histogram in [0, 1].
        lo = torch.quantile(edge_z, 0.05, dim=1, keepdim=True)
        hi = torch.quantile(edge_z, 0.95, dim=1, keepdim=True)
        return ((edge_z - lo) / (hi - lo + 1e-8)).clamp_(0.0, 1.0)

    def _conditional_log_b(
        self,
        cum_col_usage: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """
        Build the per-slice column-target marginal `log_b_override` from the
        cumulative column usage of previously-decoded slices.

        Math
        ----
        For uniform target `b_unif = 1/N` and cumulative usage `u` (already
        a probability vector over atoms) we set

            b_i  ∝  max(b_unif − α · u, b_floor)            (un-normalised)
            b_i  ← b_i / Σ_j b_i_j                          (re-normalised)
            log_b_override = log b_i                        (shape (B, 1, N))

        Edge cases
        ----------
        - First slice (no prior usage)   → return None (uniform b is used).
        - Conditional mode disabled       → return None.
        - α = 0                           → b_i = b_unif exactly; we still
                                            return None to skip the override
                                            (numerically identical, cheaper).
        """
        if not self._cond_active:
            return None
        if cum_col_usage is None or self.cond_alpha == 0.0:
            return None

        N = self.dict_num
        b_unif = 1.0 / N
        b_floor = b_unif * 1e-2  # never let an atom be fully blocked
        b = (b_unif - self.cond_alpha * cum_col_usage).clamp(min=b_floor)  # (B, N)
        b = b / b.sum(dim=-1, keepdim=True)
        return b.to(device=device, dtype=dtype).log().unsqueeze(1)  # (B, 1, N)

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
        y = self._encode_image(x)
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
        # Stash for Prop. B coherence telemetry — detached, no grad surface.
        # Accessor: `model.last_dt()` (used by analyze/plot_prop_b_coherence.py
        # and the val-epoch coherence logging in train.py).
        self._last_dt = dt.detach()
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
        total_slice_coh = zero  # B3: per-slice k_dict coherence accumulator
        row_mass_list: list[torch.Tensor] = []
        # Running mean of column marginals over previously-decoded slices
        # (no_grad — only used to build log_b for the next slice).
        cum_col_usage: torch.Tensor | None = None

        # C1: pooled hyperprior summary for image-conditional range bias.
        # AdaptiveAvgPool2d(1) → (B, 2N) — one summary per image.
        if self.image_conditional_range and self.range_bias_predictors is not None:
            pooled_hp = hyper_prior.mean(dim=(-2, -1))  # (B, 2N)
        else:
            pooled_hp = None

        for i, y_slice in enumerate(y_slices):
            rho_out = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)

            if self.routing_mode != "unbalanced_eot":
                # Keep rho_predictors in the backward graph for DDP correctness
                # with find_unused_parameters=False.  rho_out is rho_raw * 0.0 in
                # this branch (see _compute_rho_spatial), so contribution = 0.
                if self.training:
                    total_col_neg_H = total_col_neg_H + rho_out.sum()
                rho_spatial = None
            else:
                rho_spatial = rho_out

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

            log_b_override = self._conditional_log_b(
                cum_col_usage, query.size(0), query.device, query.dtype
            )
            # C1: image-conditional log_adaptive_range bias from pooled hp.
            # tanh-bounded to ±1.0 (sub-octave bias on top of the global
            # learnable scalar).  None when image-conditional is disabled.
            if pooled_hp is not None:
                range_bias_i = torch.tanh(
                    self.range_bias_predictors[i](pooled_hp).squeeze(-1)
                )
            else:
                range_bias_i = None
            dict_info, attn_aux = self.eot_attentions[i](
                query,
                k_dict,
                v_dict,
                rho_spatial,
                calc_disp=self.training,
                log_b_override=log_b_override,
                range_bias=range_bias_i,
            )

            # B3: accumulate per-slice k_dict coherence (cheap, ~N² flops).
            if self.training:
                total_slice_coh = total_slice_coh + (
                    UnifiedDictionaryAttention.k_coherence_loss(k_dict)
                )

            # Safety guard: if the EOT pipeline produced NaN/Inf (e.g. routing
            # fully collapsed to a single token and the bmm with v_norm hit a
            # degenerate scale), zero the dict_info and let the rest of the
            # slice run from the hyper-prior alone.  Without this the NaN
            # propagates through cc_mean / cc_scale into the Gaussian
            # conditional and every downstream pixel of x_hat becomes NaN —
            # which the eval table then mis-reports as 100 dB.
            if not torch.isfinite(dict_info).all():
                dict_info = torch.zeros_like(dict_info)

            # Update cumulative column usage (no-grad).  Running average so it
            # stays a probability vector regardless of slice count.
            if self._cond_active and "col_mass" in attn_aux:
                col_mass_i = attn_aux["col_mass"].detach()  # (B, N)
                if cum_col_usage is None:
                    cum_col_usage = col_mass_i
                else:
                    cum_col_usage = (cum_col_usage * i + col_mass_i) / (i + 1)

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
            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11)

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, means=mu
            )

            # ── STE for y (CMIC-style): when use_ste=True, replace the noisy
            # y_hat_slice with a hard-rounded identity-backward proxy. The
            # likelihood above still uses noise-relaxed quantisation, which
            # is required for the rate term to remain differentiable.
            if self.training and self.use_ste:
                y_hat_slice = (
                    (torch.round(y_slice - mu) + mu)
                    - y_hat_slice.detach()
                    + y_hat_slice
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
        x_hat = self._decode_latent(torch.cat(y_hat_slices, dim=1))

        # ── Slice-averaged entropy signals + stacked row_mass ────────────────
        S = float(self.num_slices)
        column_neg_entropy = total_col_neg_H / S
        row_entropy = total_row_H / S
        tv_loss = total_tv / S
        slice_coherence = total_slice_coh / S  # B3: averaged across slices
        # Stash for slice_coherence_loss() accessor (used by ablation tools).
        self._last_slice_coherence = slice_coherence.detach() if not self.training else slice_coherence
        row_mass = torch.stack(row_mass_list, dim=1)  # (B, S, Hz*Wz)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": torch.cat(y_likelihood, dim=1),
                "z": z_likelihoods,
            },
            "aux_loss": self.aux_loss(),
            "ortho_loss": self.ortho_loss(),
            "column_neg_entropy": column_neg_entropy,
            "row_entropy": row_entropy,
            "row_mass": row_mass,
            "complexity": complexity,
            "dict_penalty": dict_penalty,
            "slice_coherence": slice_coherence,
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

        y = self._encode_image(x)
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

        cum_col_usage: torch.Tensor | None = None

        # C1: pooled hyperprior summary for image-conditional range bias.
        if self.image_conditional_range and self.range_bias_predictors is not None:
            pooled_hp = hyper_prior.mean(dim=(-2, -1))
        else:
            pooled_hp = None

        for i, y_slice in enumerate(y_slices):
            rho_out = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)
            rho_spatial = rho_out if self.routing_mode == "unbalanced_eot" else None

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

            log_b_override = self._conditional_log_b(
                cum_col_usage, query.size(0), query.device, query.dtype
            )
            range_bias_i = (
                torch.tanh(self.range_bias_predictors[i](pooled_hp).squeeze(-1))
                if pooled_hp is not None
                else None
            )
            dict_info, _aux = self.eot_attentions[i](
                query,
                k_dict,
                v_dict,
                rho_spatial,
                calc_disp=False,
                log_b_override=log_b_override,
                range_bias=range_bias_i,
            )

            # NaN guard — parity with forward().  Without this, a single bad
            # slice nukes the whole image; codec strings still get written
            # because compress() never checks, but decompress() reads back
            # a finite quantised value through extreme μ → overflow at g_s.
            if not torch.isfinite(dict_info).all():
                dict_info = torch.zeros_like(dict_info)

            if self._cond_active and "col_mass" in _aux:
                col_mass_i = _aux["col_mass"].detach()
                cum_col_usage = (
                    col_mass_i
                    if cum_col_usage is None
                    else (cum_col_usage * i + col_mass_i) / (i + 1)
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
                hyper_prior.size(0),
                self.slice_ch,
                hyper_prior.size(2),
                hyper_prior.size(3),
                device=hyper_prior.device,
                dtype=hyper_prior.dtype,
            )

        cum_col_usage: torch.Tensor | None = None

        # C1: pooled hyperprior summary for image-conditional range bias.
        if self.image_conditional_range and self.range_bias_predictors is not None:
            pooled_hp = hyper_prior.mean(dim=(-2, -1))
        else:
            pooled_hp = None

        for i in range(self.num_slices):
            rho_out = self._compute_rho_spatial(i, hyper_prior, y_hat_slices)
            rho_spatial = rho_out if self.routing_mode == "unbalanced_eot" else None

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

            log_b_override = self._conditional_log_b(
                cum_col_usage, query.size(0), query.device, query.dtype
            )
            range_bias_i = (
                torch.tanh(self.range_bias_predictors[i](pooled_hp).squeeze(-1))
                if pooled_hp is not None
                else None
            )
            dict_info, _aux = self.eot_attentions[i](
                query,
                k_dict,
                v_dict,
                rho_spatial,
                calc_disp=False,
                log_b_override=log_b_override,
                range_bias=range_bias_i,
            )

            # NaN guard — parity with forward() and compress().  Must match
            # compress() byte-for-byte: if compress zeroed dict_info because
            # of NaN, decompress must do the same to recover the same μ/σ.
            if not torch.isfinite(dict_info).all():
                dict_info = torch.zeros_like(dict_info)

            if self._cond_active and "col_mass" in _aux:
                col_mass_i = _aux["col_mass"].detach()
                cum_col_usage = (
                    col_mass_i
                    if cum_col_usage is None
                    else (cum_col_usage * i + col_mass_i) / (i + 1)
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
        x_hat = self._decode_latent(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}
