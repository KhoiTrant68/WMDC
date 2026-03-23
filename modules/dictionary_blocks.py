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
    Unified Dictionary Attention with three routing modes, derived from
    the authors' reference implementation.

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
        input_dim   : input channel count
        output_dim  : output channels after out_proj
        dict_num    : N (number of dictionary tokens)
        dict_dim    : D (token embedding dimension)
        tau         : softmax temperature (only for 'softmax' mode)
        ot_eps      : fixed Sinkhorn blur eps (scalar). Fixed for convergence.
        iters       : Sinkhorn iterations
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
        iters: int = 3,
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

        # Frozen depthwise spatial smoothing of cost matrix.
        # Kept frozen: learned smoothing can degenerate to identity or sharpen
        # excessively, both of which hurt routing stability.
        self.spatial_smooth = nn.Conv2d(
            dict_num,
            dict_num,
            kernel_size=3,
            padding=1,
            groups=dict_num,
            bias=False,
            padding_mode="reflect",
        )
        nn.init.constant_(self.spatial_smooth.weight, 1.0 / 9.0)
        self.spatial_smooth.weight.requires_grad_(False)

        # Output projection.  GroupNorm(1, dict_dim) stabilises magnitudes from
        # the unnormalised unbalanced OT plan (entries need not sum to 1).
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GroupNorm(1, dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        # Set externally by forward hook for visualisation
        self.attn_probs: torch.Tensor = None

    # -----------------------------------------------------------------------
    # Cost matrix
    # -----------------------------------------------------------------------

    def _cost_matrix(
        self, x: torch.Tensor, k: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Compute the smoothed cosine-distance cost matrix C ∈ [0, 2].

        C_{ij} = 1 − cos(q_i, k_j)

        Args:
            x : (B, input_dim, H, W)
            k : (B, N, D) dictionary keys
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

        # Spatial smoothing in (B, N, H, W) space
        C_sp = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_sp = self.spatial_smooth(C_sp)
        C_mat = C_sp.view(B, self.dict_num, HW).transpose(1, 2).contiguous()
        return C_mat

    # -----------------------------------------------------------------------
    # Softmin operators
    # -----------------------------------------------------------------------

    @staticmethod
    def _softmin_x(
        f: torch.Tensor, log_alpha: float, C_mat: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """
        Softmin over source positions i (HW dim → becomes col output N).

        Matches authors' softmin_x from utils.py:
            softmin_x(f_i, ep)_j = -ep * LSE_i[ (f_i/ep + log(a_i)) - C_{ij}/ep ]
                                  = -ep * LSE_i[ (f_i + ep*log(a_i) - C_{ij}) / ep ]

        Args:
            f        : (B, HW)  source dual potential
            log_alpha: scalar = log(1/HW)
            C_mat    : (B, HW, N)
            eps      : scalar

        Returns:
            (B, N)  target dual potential update (positive for typical C > 0)
        """
        # (f/eps + log_alpha): (B, HW)
        # Expand to (B, 1, HW) then broadcast with (B, N, HW) from C.T / eps
        # logsumexp over the HW dimension (dim=2 in B×N×HW)
        inner = (f / eps + log_alpha).unsqueeze(1) - C_mat.transpose(
            1, 2
        ) / eps  # (B, N, HW)
        return -eps * torch.logsumexp(inner, dim=2)  # (B, N)

    @staticmethod
    def _softmin_y(
        g: torch.Tensor, log_beta: float, C_mat: torch.Tensor, eps: float
    ) -> torch.Tensor:
        """
        Softmin over target positions j (N dim → becomes row output HW).

        Matches authors' softmin_y from utils.py:
            softmin_y(g_j, ep)_i = -ep * LSE_j[ (g_j/ep + log(b_j)) - C_{ij}/ep ]
                                  = -ep * LSE_j[ (g_j + ep*log(b_j) - C_{ij}) / ep ]

        Args:
            g        : (B, N)   target dual potential
            log_beta : scalar = log(1/N)
            C_mat    : (B, HW, N)
            eps      : scalar

        Returns:
            (B, HW)  source dual potential update
        """
        # (g/eps + log_beta): (B, N)
        # Expand to (B, 1, N) then broadcast with (B, HW, N) / eps
        # logsumexp over the N dimension (dim=2 in B×HW×N)
        inner = (g / eps + log_beta).unsqueeze(1) - C_mat / eps  # (B, HW, N)
        return -eps * torch.logsumexp(inner, dim=2)  # (B, HW)

    # -----------------------------------------------------------------------
    # Routing: softmax
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        return F.softmax(-C_mat / self.tau, dim=-1)  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Routing: balanced EOT
    # -----------------------------------------------------------------------

    def _route_balanced_eot(self, C_mat: torch.Tensor) -> torch.Tensor:
        """
        Balanced Sinkhorn OT using authors' update rule:
            g_j = -aprox(-softmin_x(f_i))  with aprox = identity
                = softmin_x(f_i)
            f_i = softmin_y(g_j)

        Plan: pi_{ij} = exp((f_i+g_j-C_{ij})/eps) * alpha_i * beta_j
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = math.log(1.0 / HW)  # log(alpha_i), scalar
        log_beta = math.log(1.0 / N)  # log(beta_j),  scalar

        f = C_mat.new_zeros(B, HW)
        g = C_mat.new_zeros(B, N)

        for _ in range(self.iters):
            # g_j = softmin_x(f_i)  [authors: g = -aprox(-softmin_x(f)) = softmin_x(f)]
            g = self._softmin_x(f, log_alpha, C_mat, eps)  # (B, N)
            # f_i = softmin_y(g_j)
            f = self._softmin_y(g, log_beta, C_mat, eps)  # (B, HW)

        # pi_{ij} = exp((f_i+g_j-C_{ij})/eps + log_alpha + log_beta)
        logits = (f.unsqueeze(2) + g.unsqueeze(1) - C_mat) / eps + log_alpha + log_beta
        return torch.exp(torch.clamp(logits, max=20.0))  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Routing: KL-unbalanced EOT (KL penalty, spatially-varying rho)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        KL-unbalanced Sinkhorn OT using authors' update rule:
            g_j = -aprox(-softmin_x(f_i))
                = shrink_col * softmin_x(f_i)   [KL: -aprox(-p) = shrink*p]
            f_i = shrink_row * softmin_y(g_j)

        KL aprox from authors' entropy.py (KullbackLeibler.aprox):
            aprox(x) = (1/(1 + eps/rho)) * x = rho/(rho+eps) * x
            -aprox(-p) = rho/(rho+eps) * p = shrink * p

        Spatially-varying rho(x_i) per §4.7.2 of Séjourné et al.:
            shrink_row_i = rho(x_i) / (rho(x_i) + eps)   per source position
            shrink_col   = rho_mean / (rho_mean + eps)    global mean for target

        Args:
            C_mat    : (B, HW, N)
            rho_flat : (B, HW) spatially-varying KL reach, clamped > 0
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = math.log(1.0 / HW)
        log_beta = math.log(1.0 / N)

        # Per-source shrinkage: rho(x_i)/(rho(x_i)+eps), shape (B, HW)
        shrink_row = rho_flat / (rho_flat + eps)

        # Global shrinkage for target (dictionary) side
        rho_mean = rho_flat.mean(dim=1, keepdim=True)  # (B, 1)
        shrink_col = rho_mean / (rho_mean + eps)  # (B, 1)

        f = C_mat.new_zeros(B, HW)
        g = C_mat.new_zeros(B, N)

        for _ in range(self.iters):
            # g_j = shrink_col * softmin_x(f_i)
            g = shrink_col * self._softmin_x(f, log_alpha, C_mat, eps)  # (B, N)
            # f_i = shrink_row * softmin_y(g_j)   (spatially-varying shrinkage)
            f = shrink_row * self._softmin_y(g, log_beta, C_mat, eps)  # (B, HW)

        # pi_{ij} = exp((f_i+g_j-C_{ij})/eps + log_alpha + log_beta)
        logits = (f.unsqueeze(2) + g.unsqueeze(1) - C_mat) / eps + log_alpha + log_beta
        return torch.exp(torch.clamp(logits, max=20.0))  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Dispersion loss: token utilisation entropy maximisation
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """
        Maximise Shannon entropy of the token utilisation marginal.

        Prevents dictionary collapse by encouraging all N tokens to be used
        roughly equally across the image.

        Steps:
          1. Row-normalise P → proper conditional routing over tokens.
          2. Average over HW positions → token utilisation marginal ∈ Delta^{N-1}.
          3. Return -H(marginal) as loss (minimising → maximising diversity).

        Args:
            P : (B, HW, N) routing plan

        Returns:
            scalar ≥ 0  (0 when all tokens used uniformly)
        """
        # Row-normalise: conditional distribution over tokens for each spatial pos
        P_row = P / (P.sum(dim=2, keepdim=True) + 1e-8)  # (B, HW, N)
        # Marginal: average over spatial positions
        marg = P_row.mean(dim=1)  # (B, N)
        # Entropy
        H = -(marg * (marg + 1e-8).log()).sum(dim=1).mean()  # scalar
        return -H  # minimise → maximise diversity

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
            k           : (B, N, D)              pre-projected dict keys  (from WMDC)
            v           : (B, N, D)              pre-projected dict values (from WMDC)
            rho_spatial : (B, H, W)              spatially-varying KL mass penalty ρ(x)
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
                P = P + 0.0 * rho_spatial.sum()

        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)
            if self.training:
                P = P + 0.0 * rho_spatial.sum()

        elif self.routing_mode == "unbalanced_eot":
            # Flatten and clamp rho: (B, H, W) → (B, HW), strictly positive
            rho_flat = rho_spatial.view(B, HW).clamp(min=0.01)
            P = self._route_unbalanced_eot(C_mat, rho_flat)

        else:
            raise RuntimeError(f"Unknown routing_mode: {self.routing_mode}")

        # Store for external analysis / visualisation hooks
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
