import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# QueryDictionaryGenerator
# ---------------------------------------------------------------------------


class QueryDictionaryGenerator(nn.Module):
    """
    Generates a content-adaptive dictionary from the quantised hyperprior z_hat.

    Cross-attention between learnable dictionary prototype queries and the
    spatial hyperprior context produces per-image, per-token embeddings.

    Args:
        in_dim   : number of channels in z_hat
        dict_num : number of dictionary tokens N
        dict_dim : embedding dimension D per token
        num_heads: number of attention heads
    """

    def __init__(
        self,
        in_dim: int = 192,
        dict_num: int = 128,
        dict_dim: int = 640,
        num_heads: int = 4,
    ):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim

        # Learnable dictionary prototype queries  (1, N, in_dim)
        self.dict_queries = nn.Parameter(torch.randn(1, dict_num, in_dim))

        # Depthwise positional encoding injected into the hyperprior context
        self.pos_enc = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)

        # Project from in_dim to dict_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, dict_dim),
            nn.GELU(),
            nn.Linear(dict_dim, dict_dim),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_hat : (B, C, H, W) quantised hyperprior

        Returns:
            dt    : (B, N, dict_dim) content-adaptive dictionary
        """
        B, C, H, W = z_hat.shape
        context = z_hat + self.pos_enc(z_hat)  # (B, C, H, W)
        context = context.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)

        queries = self.dict_queries.expand(B, -1, -1)  # (B, N, C)
        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)  # (B, N, C)
        return self.proj(out)  # (B, N, dict_dim)


# ---------------------------------------------------------------------------
# UnifiedDictionaryAttention
# ---------------------------------------------------------------------------


class UnifiedDictionaryAttention(nn.Module):
    """
    Unified Dictionary Attention with three routing modes.

    Routing modes
    -------------
    'softmax'
        Standard softmax attention with temperature tau.

    'balanced_eot'
        Balanced Sinkhorn OT. Sinkhorn updates:
            g_j = softmin_x(f_i)
            f_i = softmin_y(g_j)

    'unbalanced_eot'
        KL-unbalanced Sinkhorn OT. Sinkhorn updates:
            g_j = shrink_col * softmin_x(f_i)
            f_i = shrink_row * softmin_y(g_j)
        where shrink = rho / (rho + eps),
        and rho(x) is spatially-varying (Séjourné §4.7.2).

    Softmin definitions (directly from authors' utils.py)
    ------------------------------------------------------
    For source measure alpha (uniform, mass 1/HW) and
    target measure beta  (uniform, mass 1/N):

        softmin_x(f)_j = -eps * LSE_i[ (f_i + eps*log(1/HW) - C_{ij}) / eps ]
        softmin_y(g)_i = -eps * LSE_j[ (g_j + eps*log(1/N)  - C_{ij}) / eps ]

    Transport plan (Eq. 8 of paper)
    --------------------------------
        pi_{ij} = exp( (f_i + g_j - C_{ij}) / eps ) * (1/HW) * (1/N)
        log pi_{ij} = (f_i + g_j - C_{ij}) / eps + log_alpha + log_beta

    Args:
        input_dim   : input channel count for query projection
        output_dim  : output channels after out_proj
        dict_num    : N (number of dictionary tokens)
        dict_dim    : D (token embedding dimension)
        tau         : softmax temperature (softmax mode only)
        ot_eps      : Sinkhorn entropic regularisation ε (scalar).
        iters       : Sinkhorn iterations (≥20 recommended for eps=0.1)
        routing_mode: {'softmax', 'balanced_eot', 'unbalanced_eot'}
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        dict_num: int = 128,
        dict_dim: int = 640,
        tau: float = 0.5,
        ot_eps: float = 0.1,
        iters: int = 20,
        routing_mode: str = "unbalanced_eot",
    ):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim
        self.tau = tau  # softmax temperature only
        self.ot_eps = ot_eps  # Sinkhorn blur (eps / blur in authors' code)
        self.iters = iters

        valid = {"softmax", "balanced_eot", "unbalanced_eot"}
        if routing_mode not in valid:
            raise ValueError(f"routing_mode must be one of {valid}")
        self.routing_mode = routing_mode

        # Query projection
        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)

        # Learnable depthwise spatial smoothing, identity-initialised.
        # Frozen uniform average has no compression-theoretic justification.
        self.spatial_smooth = nn.Conv2d(
            dict_num,
            dict_num,
            kernel_size=3,
            padding=1,
            groups=dict_num,
            bias=False,
            padding_mode="reflect",
        )
        # Initialise to identity: centre pixel only, zero bias.
        nn.init.zeros_(self.spatial_smooth.weight)
        self.spatial_smooth.weight.data[:, 0, 1, 1] = 1.0  # centre pixel
        nn.init.zeros_(self.spatial_smooth.bias)
        # Now learnable — will deviate from identity as training proceeds.

        # Output projection with GroupNorm to stabilise unnormalised OT plan.
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GroupNorm(1, dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        # Set externally by forward hook for visualisation.
        self.attn_probs: torch.Tensor = None

    # -----------------------------------------------------------------------
    # Cost matrix
    # -----------------------------------------------------------------------

    def _cost_matrix(
        self, x: torch.Tensor, k: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Smoothed cosine-distance cost matrix C ∈ [0, 2].
            C_{ij} = 1 − cos(q_i, k_j)

        Args:
            x : (B, input_dim, H, W)
            k : (B, N, D) dictionary keys (pre-projected, per-slice)
            H, W : spatial dimensions

        Returns:
            C_mat : (B, HW, N)
        """
        B = x.shape[0]
        HW = H * W

        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2)  # (B, HW, D)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, HW, N)

        # Learnable spatial smoothing in (B, N, H, W) space.
        C_sp = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_sp = self.spatial_smooth(C_sp)
        C_mat = C_sp.view(B, self.dict_num, HW).transpose(1, 2).contiguous()
        return C_mat

    # -----------------------------------------------------------------------
    # Log-domain Sinkhorn: balanced
    # -----------------------------------------------------------------------

    def _route_balanced_eot(self, C_mat: torch.Tensor) -> torch.Tensor:
        """
        Balanced Sinkhorn OT in log-domain.

        Log-domain updates (Peyré & Cuturi 2019, Alg. 2):
            log g_j = log β_j − LSE_i[ log α_i + log f_i − C_{ij}/ε ]
            log f_i = log α_i − LSE_j[ log β_j + log g_j − C_{ij}/ε ]

        Uniform marginals: α_i = 1/HW, β_j = 1/N.
        Transport plan:
            log π_{ij} = (log f_i + log g_j − C_{ij}/ε) + log α_i + log β_j

        Returns:
            P : (B, HW, N) transport plan (not row-stochastic; mass ~1)
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)  # scalar log(1/HW)
        log_beta = -math.log(N)  # scalar log(1/N)

        # Scaled cost for numerical stability.
        M = C_mat / eps  # (B, HW, N)

        log_f = torch.zeros(B, HW, device=C_mat.device, dtype=C_mat.dtype)
        log_g = torch.zeros(B, N, device=C_mat.device, dtype=C_mat.dtype)

        for _ in range(self.iters):
            # log g_j = log_beta − LSE_i[ log_alpha + log_f_i − M_{ij} ]
            #         = log_beta − LSE_i[ (log_f_i + log_alpha) − M_{ij} ]
            inner_g = (log_f + log_alpha).unsqueeze(2) - M  # (B, HW, N)
            log_g = log_beta - torch.logsumexp(inner_g, dim=1)  # (B, N)

            # log f_i = log_alpha − LSE_j[ log_beta + log_g_j − M_{ij} ]
            inner_f = (log_g + log_beta).unsqueeze(1) - M  # (B, HW, N)
            log_f = log_alpha - torch.logsumexp(inner_f, dim=2)  # (B, HW)

        # log π_{ij} = log_f_i + log_g_j − M_{ij} + log_alpha + log_beta
        log_P = (
            log_f.unsqueeze(2) + log_g.unsqueeze(1) - M + log_alpha + log_beta
        )  # (B, HW, N)
        return torch.exp(log_P)  # safe: log_P is properly bounded by log-domain ops

    # -----------------------------------------------------------------------
    # Log-domain Sinkhorn: KL-unbalanced (spatially-varying rho)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        KL-unbalanced Sinkhorn OT in log-domain with spatially-varying ρ(x).

        KL aprox (Séjourné et al. 2022, KullbackLeibler.aprox):
            aprox_ρ(p) = ρ/(ρ+ε) · p          (scalar p)

        Log-domain unbalanced updates:
            log g_j = shrink_col · (log β_j − LSE_i[ log α_i + log f_i − M_{ij} ])
            log f_i = shrink_row_i · (log α_i − LSE_j[ log β_j + log g_j − M_{ij} ])

        where:
            shrink_row_i = ρ(x_i) / (ρ(x_i) + ε)    per source position (B, HW)
            shrink_col   = ρ_mean / (ρ_mean + ε)      global mean for target (B, 1)

        rho_flat is fully differentiable — gradients flow to rho_predictor
        through every Sinkhorn iteration.

        Args:
            C_mat    : (B, HW, N)
            rho_flat : (B, HW) spatially-varying KL reach, strictly > 0

        Returns:
            P : (B, HW, N) transport plan
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)
        log_beta = -math.log(N)

        M = C_mat / eps  # (B, HW, N)

        # Per-source shrinkage: (B, HW) — differentiable w.r.t. rho_flat.
        shrink_row = rho_flat / (rho_flat + eps)  # (B, HW)

        # Global shrinkage for target (dictionary) side.
        rho_mean = rho_flat.mean(dim=1, keepdim=True)  # (B, 1)
        shrink_col = rho_mean / (rho_mean + eps)  # (B, 1)

        log_f = torch.zeros(B, HW, device=C_mat.device, dtype=C_mat.dtype)
        log_g = torch.zeros(B, N, device=C_mat.device, dtype=C_mat.dtype)

        for _ in range(self.iters):
            # Unbalanced target update
            inner_g = (log_f + log_alpha).unsqueeze(2) - M  # (B, HW, N)
            raw_g = log_beta - torch.logsumexp(inner_g, dim=1)  # (B, N)
            log_g = shrink_col * raw_g  # (B, N); shrink_col is (B, 1) — broadcasts

            # Unbalanced source update (spatially-varying shrinkage)
            inner_f = (log_g + log_beta).unsqueeze(1) - M  # (B, HW, N)
            raw_f = log_alpha - torch.logsumexp(inner_f, dim=2)  # (B, HW)
            log_f = shrink_row * raw_f  # (B, HW); element-wise

        log_P = (
            log_f.unsqueeze(2) + log_g.unsqueeze(1) - M + log_alpha + log_beta
        )  # (B, HW, N)
        return torch.exp(log_P)

    # -----------------------------------------------------------------------
    # Routing: softmax
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        return F.softmax(-C_mat / self.tau, dim=-1)  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Dispersion loss
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """
        Maximise Shannon entropy of the *true* OT target marginal.

        Original bug: the code row-normalised P (making it row-stochastic),
        then averaged rows to get a "marginal".  This is NOT the OT marginal
        and discards the spatial mass variation produced by rho_spatial.

        Correct target marginal:
            marginal_j = Σ_i P_{ij}    (total mass routed to token j)

        This is the quantity that should be uniform for a well-used dictionary.

        Args:
            P : (B, HW, N) routing plan (NOT necessarily row-stochastic)

        Returns:
            scalar ≥ 0  (0 when all tokens receive equal total mass)
        """
        # True target marginal: sum over source positions.
        target_marginal = P.sum(dim=1)  # (B, N) — mass at each dictionary token
        # Normalise to a probability vector for entropy computation.
        target_marginal = target_marginal / (
            target_marginal.sum(dim=1, keepdim=True) + 1e-8
        )  # (B, N) in simplex
        # Entropy H = -Σ p log p; we return -H so minimising → maximising diversity.
        H = -(target_marginal * (target_marginal + 1e-8).log()).sum(dim=1).mean()
        return -H  # scalar ≥ 0

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rho_spatial: torch.Tensor,
        calc_disp: bool = False,
    ) -> tuple:
        """
        Args:
            x           : (B, input_dim, H, W)  spatial query features
            k           : (B, N, D)  pre-projected dict keys  (per-slice from WMDC)
            v           : (B, N, D)  pre-projected dict values (per-slice from WMDC)
            rho_spatial : (B, H, W)  spatially-varying KL mass penalty ρ(x)
                          Ignored for 'softmax' and 'balanced_eot' modes.
            calc_disp   : if True, compute and return dispersion loss

        Returns:
            out       : (B, output_dim, H, W)
            disp_loss : scalar tensor (0.0 if calc_disp=False)
        """
        B, _, H, W = x.shape
        HW = H * W

        C_mat = self._cost_matrix(x, k, H, W)  # (B, HW, N)

        # ── Routing ──────────────────────────────────────────────────────────
        if self.routing_mode == "softmax":
            P = self._route_softmax(C_mat)
            if self.training:
                # Keep rho in the compute graph to avoid DDP unused-param error.
                P = P + 0.0 * rho_spatial.sum()

        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)
            if self.training:
                P = P + 0.0 * rho_spatial.sum()

        elif self.routing_mode == "unbalanced_eot":
            rho_flat = rho_spatial.view(B, HW).clamp(min=0.01)
            P = self._route_unbalanced_eot(C_mat, rho_flat)

        else:
            raise RuntimeError(f"Unknown routing_mode: {self.routing_mode}")

        # Store for external analysis / visualisation hooks.
        self.attn_probs = P.detach()

        # ── Value aggregation ─────────────────────────────────────────────────
        # (B, HW, N) × (B, N, D) → (B, HW, D) → (B, D, H, W)
        out = torch.bmm(P, v).transpose(1, 2).contiguous().view(B, -1, H, W)

        # ── Dispersion loss ───────────────────────────────────────────────────
        if self.training and calc_disp:
            disp_loss = self._dispersion_loss(P)
        else:
            disp_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        return self.out_proj(out), disp_loss
