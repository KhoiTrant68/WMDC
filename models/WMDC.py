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
    if name == "fdm":
        return FrequencyDisentangledMamba(
            dim,
            drop_path=drop_path,
            use_content_adaptive=use_content_adaptive,
            cluster_num=cluster_num,
        )
    from ablation_models.backbone_variants import build_backbone

    return build_backbone(name, dim=dim, drop_path=drop_path)


@contextlib.contextmanager
def ste_mode(model: "WMDC"):
    prev = model.use_ste
    model.use_ste = True
    try:
        yield model
    finally:
        model.use_ste = prev


def load_legacy_checkpoint(
    model: "WMDC", ckpt_path: str, device: str = "cpu"
) -> "WMDC":
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    remap: dict = {}
    for k, v in sd.items():
        if k.startswith("h_scale_s."):
            suffix = k[len("h_scale_s.") :]
            idx_str, _, rest = suffix.partition(".")
            idx = int(idx_str)
            if idx < 2:
                remap[f"h_trunk.{idx}.{rest}"] = v
            else:
                remap[f"h_scale_head.2.{rest}"] = v
        elif k.startswith("h_mean_s."):
            suffix = k[len("h_mean_s.") :]
            idx_str, _, rest = suffix.partition(".")
            idx = int(idx_str)
            if idx == 2:
                remap[f"h_mean_head.{rest}"] = v
        elif k.startswith("h_scale_head."):
            tail = k[len("h_scale_head.") :]
            head = tail.split(".", 1)[0]
            if head in ("weight", "bias"):
                remap[f"h_scale_head.2.{tail}"] = v
            else:
                remap[k] = v
        else:
            remap[k] = v
    missing, unexpected = model.load_state_dict(remap, strict=False)
    if missing:
        print(f"[load_legacy_checkpoint] Missing keys: {missing}")
    if unexpected:
        print(f"[load_legacy_checkpoint] Unexpected keys: {unexpected}")
    return model


class WMDC(CompressionModel):
    """
    Wavelet-Mamba Dictionary Compression model.
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

        self.register_buffer("_use_ste_flag", torch.zeros(1, dtype=torch.bool))
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

        # ── Hyper-decoder ─────────────────────────────────────────────────────
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

        # ── Spatially-adaptive ρ predictors ────────────────────────────────────
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
            for predictor in self.rho_predictors:
                nn.init.zeros_(predictor[-1].weight)
                nn.init.constant_(predictor[-1].bias, 0.5)
        else:
            self.rho_predictors = None

        # ── Bootstrap memory — ALWAYS registered as nn.Module ────────────────
        if not use_dense_concat and memory_init == "bootstrap":
            self.init_memory = nn.Conv2d(2 * M, self.slice_ch, kernel_size=3, padding=1)
        else:
            # nn.Identity as a no-op placeholder so state_dict is consistent
            self.init_memory = nn.Identity()

        # ── Per-slice K/V projections ──────────────────────────────────────────
        self.k_projs = nn.ModuleList(
            [nn.Linear(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )
        self.v_projs = nn.ModuleList(
            [nn.Linear(self.dict_dim, self.dict_dim) for _ in range(num_slices)]
        )

        # ── Per-slice EOT dictionary attention ────────────────────────────────
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

        # ── Gated memory updaters ─────────────────────────────────────────────
        if not use_dense_concat:
            self.memory_updaters = nn.ModuleList(
                [GatedMemoryUpdater(self.slice_ch) for _ in range(num_slices - 1)]
            )

        # ── Slice-specific context transforms ─────────────────────────────────
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

        # ── LRP transforms ────────────────────────────────────────────────────
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
        for transform in self.lrp_transforms:
            nn.init.zeros_(transform[-1].weight)
            nn.init.zeros_(transform[-1].bias)

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
        """
        if scale_table is None:
            scale_table = torch.exp(
                torch.linspace(
                    math.log(0.11), math.log(1024.0), 64, dtype=torch.float32
                )
            )
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def aux_loss(self) -> torch.Tensor:
        return sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))

    def _hyper_decode(self, z_hat: torch.Tensor):
        trunk = self.h_trunk(z_hat)
        return self.h_scale_head(trunk), self.h_mean_head(trunk)

    def _get_medians_safe(self) -> torch.Tensor:
        """
        Robust access to entropy bottleneck medians.
        """
        if hasattr(self.entropy_bottleneck, "medians"):
            m = self.entropy_bottleneck.medians
            if m is not None:
                return m.detach()
        if hasattr(self.entropy_bottleneck, "_get_medians"):
            try:
                return self.entropy_bottleneck._get_medians().detach()
            except Exception:
                pass
        C = (
            self.entropy_bottleneck._quantized_cdf.shape[0]
            if hasattr(self.entropy_bottleneck, "_quantized_cdf")
            else self.N
        )
        return torch.zeros(C, device=next(self.parameters()).device)

    @staticmethod
    @torch.no_grad()
    def _compute_complexity(
        x: torch.Tensor, target_size: tuple[int, int]
    ) -> torch.Tensor:
        x_gray = x.mean(dim=1, keepdim=True)
        kx = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
            dtype=x.dtype,
            device=x.device,
        ).view(1, 1, 3, 3)
        ky = kx.transpose(2, 3)
        gx = F.conv2d(x_gray, kx, padding=1)
        gy = F.conv2d(x_gray, ky, padding=1)
        edge = torch.sqrt(gx * gx + gy * gy + 1e-12)
        edge_z = F.adaptive_avg_pool2d(edge, target_size).flatten(1)
        mn = edge_z.amin(dim=1, keepdim=True)
        mx = edge_z.amax(dim=1, keepdim=True)
        return (edge_z - mn) / (mx - mn + 1e-8)

    def _compute_rho_spatial(
        self,
        slice_idx: int,
        hyper_prior: torch.Tensor,
        decoded_slices: list,
    ) -> torch.Tensor | None:
        if self.rho_predictors is None:
            return None
        context = (
            hyper_prior
            if slice_idx == 0
            else torch.cat([hyper_prior] + decoded_slices, dim=1)
        )
        rho_raw = self.rho_predictors[slice_idx](context)
        rho = F.softplus(rho_raw).clamp(min=0.05) + 1e-4
        return rho.squeeze(1)

    def _init_memory_state(self, hyper_prior: torch.Tensor) -> torch.Tensor | None:
        """
        Initialise the slice-loop memory state.
        """
        if self.use_dense_concat:
            return None
        if self.memory_init == "bootstrap":
            return self.init_memory(hyper_prior)
        # zero init
        return torch.zeros(
            hyper_prior.size(0),
            self.slice_ch,
            hyper_prior.size(2),
            hyper_prior.size(3),
            device=hyper_prior.device,
            dtype=hyper_prior.dtype,
        )

    # =========================================================================
    # Forward pass (training)
    # =========================================================================

    def forward(self, x: torch.Tensor) -> dict:
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        self.slice_attn_probs.clear()

        # ── Encode ────────────────────────────────────────────────────────────
        y = self.g_a(x)
        z = self.h_a(y)

        # Entropy bottleneck: noise relaxation (train) or STE (STE epochs)
        z_hat_soft, z_likelihoods = self.entropy_bottleneck(z)
        if self.training and self.use_ste:
            medians = self._get_medians_safe().reshape(1, -1, 1, 1)
            z_round = torch.round(z - medians) + medians
            z_hat = z_round.detach() - z.detach() + z  # STE
        else:
            z_hat = z_hat_soft

        # ── Hyper-prior ───────────────────────────────────────────────────────
        dt, dict_penalty = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)
        Hz, Wz = hyper_prior.shape[-2:]

        complexity = self._compute_complexity(x, (Hz, Wz))

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_likelihood: list[torch.Tensor] = []

        memory_state = self._init_memory_state(hyper_prior)

        zero = torch.zeros((), device=x.device, dtype=x.dtype)
        total_col_neg_H = zero
        total_row_H = zero
        total_tv = zero
        row_mass_list: list[torch.Tensor] = []

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

            dict_info, attn_aux = self.eot_attentions[i](
                query, k_dict, v_dict, rho_spatial, calc_disp=self.training
            )

            total_col_neg_H = total_col_neg_H + attn_aux["column_neg_entropy"]
            total_row_H = total_row_H + attn_aux["row_entropy"]
            total_tv = total_tv + attn_aux["tv_loss"]
            row_mass_list.append(attn_aux["row_mass"])

            if not self.training and self.eot_attentions[i].attn_probs is not None:
                self.slice_attn_probs.append(self.eot_attentions[i].attn_probs)
                self.eot_attentions[i].attn_probs = None

            support = torch.cat([query, dict_info], dim=1)
            mu = self.cc_mean_transforms[i](support)
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11, max=1024.0)

            y_hat_slice, y_slice_likelihood = self.gaussian_conditional(
                y_slice, scale, means=mu
            )

            if self.training:
                y_hat_hard = torch.round(y_slice - mu) + mu
                y_hat_for_lrp = y_hat_hard.detach() - y_hat_slice.detach() + y_hat_slice
            else:
                y_hat_for_lrp = torch.round(y_slice - mu) + mu

            lrp_support = torch.cat([support, y_hat_for_lrp], dim=1)
            residual = self.lrp_transforms[i](lrp_support)
            lrp_gate = F.softplus(self.lrp_scales[i])
            y_hat_slice_lrp = y_hat_for_lrp + lrp_gate * residual

            y_hat_slices.append(y_hat_slice_lrp)
            y_likelihood.append(y_slice_likelihood)

            if not self.use_dense_concat and i < self.num_slices - 1:
                memory_state = self.memory_updaters[i](memory_state, y_hat_slice_lrp)

        # ── Decode ────────────────────────────────────────────────────────────
        x_hat = self.g_s(torch.cat(y_hat_slices, dim=1))
        # Only clamp in eval — during training, raw x_hat keeps MSE gradient
        # flowing for out-of-range pixels.
        if not self.training:
            x_hat = x_hat.clamp(0.0, 1.0)

        S = float(self.num_slices)
        column_neg_entropy = total_col_neg_H / S
        row_entropy = total_row_H / S
        tv_loss = total_tv / S
        row_mass = torch.stack(row_mass_list, dim=1)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": torch.cat(y_likelihood, dim=1), "z": z_likelihoods},
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
        x = x.float()
        if x.size(2) % 64 != 0 or x.size(3) % 64 != 0:
            raise ValueError(f"Input must be divisible by 64. Got {x.shape}")

        self.slice_attn_probs.clear()

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        dt, _ = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_slices = y.chunk(self.num_slices, dim=1)
        y_hat_slices: list[torch.Tensor] = []
        y_strings: list = []

        memory_state = self._init_memory_state(hyper_prior)

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
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11, max=1024.0)

            index = self.gaussian_conditional.build_indexes(scale)
            y_string = self.gaussian_conditional.compress(y_slice, index, means=mu)
            y_strings.append(y_string)

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
        y_strings, z_strings = strings[0], strings[1]

        self.slice_attn_probs.clear()

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)

        dt, _ = self.hyper_to_dict(z_hat)
        latent_scales, latent_means = self._hyper_decode(z_hat)
        hyper_prior = torch.cat([latent_scales, latent_means], dim=1)

        y_hat_slices: list[torch.Tensor] = []

        memory_state = self._init_memory_state(hyper_prior)

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
            scale = self.cc_scale_transforms[i](support).clamp(min=0.11, max=1024.0)

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
