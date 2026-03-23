"""
modules/dictionary_blocks.py

Mathematically corrected implementation of:
  - QueryDictionaryGenerator
  - UnifiedDictionaryAttention

References
----------
Séjourné et al. (2023) "Sinkhorn Divergences for Unbalanced Optimal Transport"
  Algorithm 1 : Sinkhorn updates (log-domain, unbalanced)
  Table 1     : Aprox^eps_{phi*} for KL = rho/(rho+eps) * p
  Eq. 8       : plan pi_{ij} = exp((f_i+g_j-C_{ij})/eps) * alpha_i * beta_j
  §4.7.2      : spatially-varying phi-divergence => spatially-varying rho(x), fixed eps
"""

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
    Unified Dictionary Attention supporting three routing modes:

      'softmax'        — standard softmax routing (temperature tau)
      'balanced_eot'   — balanced entropic OT (Algorithm 1 of Séjourné et al.)
      'unbalanced_eot' — KL-unbalanced entropic OT (Table 1 + §4.7.2)
                         with spatially-adaptive rho(x) and fixed scalar eps

    Mathematical setup
    ------------------
    Source measure alpha : HW spatial query positions, uniform mass 1/HW
    Target measure beta  : N  dictionary tokens,       uniform mass 1/N

    Sinkhorn dual potentials (f, g) satisfy (Algorithm 1):
        f_i <- -Aprox( -ε·LSE_j[ log β_j + (g_j - C_{ij})/ε ] )
        g_j <- -Aprox( -ε·LSE_i[ log α_i + (f_i - C_{ij})/ε ] )

    For balanced OT:   Aprox(p) = p  (identity)
    For KL-unbalanced: Aprox^ε_{φ*}(p) = ρ/(ρ+ε) · p       [Table 1]
    For spatially-varying ρ(x) [§4.7.2]:
                       Aprox^ε_{φ*(·,x_i)}(p) = ρ(x_i)/(ρ(x_i)+ε) · p

    Transport plan (Eq. 8):
        π_{ij} = exp( (f_i + g_j - C_{ij})/ε ) · α_i · β_j
               = exp( (f_i + g_j - C_{ij})/ε - log(HW) - log(N) )

    Args:
        input_dim   : input channel count (2M + slice_ch in WMDC)
        output_dim  : output channels after projection (M in WMDC)
        dict_num    : N — number of dictionary tokens
        dict_dim    : D — token embedding dimension
        tau         : softmax temperature (only for 'softmax' mode)
        ot_eps      : fixed scalar ε for Sinkhorn regularisation.
                      MUST NOT be made spatially varying — that role belongs to rho.
                      Typical value: 0.05–0.2
        iters       : number of Sinkhorn iterations (3 is minimum; ablate 1,3,5,10,20)
        routing_mode: one of {'softmax', 'balanced_eot', 'unbalanced_eot'}
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
        self.tau = tau
        self.ot_eps = ot_eps  # fixed epsilon — NOT to be confused with rho
        self.iters = iters

        valid_modes = {"softmax", "balanced_eot", "unbalanced_eot"}
        if routing_mode not in valid_modes:
            raise ValueError(
                f"routing_mode must be one of {valid_modes}, got '{routing_mode}'"
            )
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
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        H: int,
        W: int,
    ) -> torch.Tensor:
        """
        Compute the smoothed cosine-distance cost matrix C ∈ [0, 2].

        C_{ij} = 1 − cos(q_i, k_j)

        Args:
            x : (B, input_dim, H, W)
            k : (B, N, D) dictionary keys (already projected by WMDC)
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

        # Spatial smoothing in (B, N, H, W) then flatten
        C_sp = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_sp = self.spatial_smooth(C_sp)
        C_mat = C_sp.view(B, self.dict_num, HW).transpose(1, 2).contiguous()

        return C_mat

    # -----------------------------------------------------------------------
    # Routing: softmax
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        """Standard softmax routing with temperature tau."""
        return F.softmax(-C_mat / self.tau, dim=-1)  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Routing: balanced EOT
    # -----------------------------------------------------------------------

    def _route_balanced_eot(
        self,
        C_mat: torch.Tensor,
    ) -> torch.Tensor:
        """
        Balanced Sinkhorn OT — Algorithm 1 of Séjourné et al. with Aprox = identity.

        Both row and col updates use the SAME scalar eps (no spatial asymmetry).

        Log-domain updates:
            f_i <- ε·log(α_i) − ε·LSE_j[(g_j − C_{ij})/ε]
            g_j <- ε·log(β_j) − ε·LSE_i[(f_i − C_{ij})/ε]

        Plan (Eq. 8):
            log π_{ij} = (f_i + g_j − C_{ij})/ε + log(α_i) + log(β_j)
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)  # log(1/HW)
        log_beta = -math.log(N)  # log(1/N)

        f = C_mat.new_zeros(B, HW)
        g = C_mat.new_zeros(B, N)

        for _ in range(self.iters):
            # Row update (source positions)
            f = log_alpha * eps - eps * torch.logsumexp(
                (g.unsqueeze(1) - C_mat) / eps, dim=2
            )  # (B, HW)
            # Col update (dictionary tokens)
            g = log_beta * eps - eps * torch.logsumexp(
                (f.unsqueeze(2) - C_mat) / eps, dim=1
            )  # (B, N)

        # Transport plan — INCLUDE log(alpha_i * beta_j) from Eq. 8
        logits = (f.unsqueeze(2) + g.unsqueeze(1) - C_mat) / eps + log_alpha + log_beta
        return torch.exp(torch.clamp(logits, max=20.0))  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Routing: unbalanced EOT (KL penalty, spatially-varying rho)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self,
        C_mat: torch.Tensor,
        rho_flat: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL-unbalanced Sinkhorn OT — Table 1 + §4.7.2 of Séjourné et al.

        Spatially-varying ρ(x_i) controls per-source-position mass conservation.
        A single fixed scalar ε provides entropic regularisation.

        KL Aprox (Table 1):
            Aprox^ε_{φ*(·,x_i)}(p) = ρ(x_i)/(ρ(x_i)+ε) · p

        Log-domain updates (Algorithm 1 + spatially-varying Aprox):
            raw_g_j = ε·log(α_i) − ε·LSE_i[(f_i − C_{ij})/ε]
            g_j     = ρ̄/(ρ̄+ε) · raw_g_j          (global mean rho for target side)

            raw_f_i = ε·log(β_j) − ε·LSE_j[(g_j − C_{ij})/ε]
            f_i     = ρ(x_i)/(ρ(x_i)+ε) · raw_f_i  (spatially-varying Aprox)

        Plan (Eq. 8):
            log π_{ij} = (f_i + g_j − C_{ij})/ε + log(α_i) + log(β_j)

        Note: the plan π need not be row-stochastic (unbalanced OT).
        GroupNorm in out_proj handles magnitude variation.

        Args:
            C_mat    : (B, HW, N)
            rho_flat : (B, HW) — spatially varying ρ, clamped ≥ 0.01
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)
        log_beta = -math.log(N)

        # Shrinkage for source (rows): ρ(x_i)/(ρ(x_i)+ε), shape (B, HW)
        shrink_row = rho_flat / (rho_flat + eps)

        # Shrinkage for target (cols): mean ρ, shape (B, 1)
        rho_mean = rho_flat.mean(dim=1, keepdim=True)
        shrink_col = rho_mean / (rho_mean + eps)

        f = C_mat.new_zeros(B, HW)
        g = C_mat.new_zeros(B, N)

        for _ in range(self.iters):
            # Target-side (col) update with global shrinkage
            raw_g = log_alpha * eps - eps * torch.logsumexp(
                (f.unsqueeze(2) - C_mat) / eps, dim=1
            )  # (B, N)
            g = shrink_col * raw_g  # (B, N)

            # Source-side (row) update with spatially-varying shrinkage
            raw_f = log_beta * eps - eps * torch.logsumexp(
                (g.unsqueeze(1) - C_mat) / eps, dim=2
            )  # (B, HW)
            f = shrink_row * raw_f  # (B, HW)

        # Transport plan — include log(alpha * beta) per Eq. 8
        logits = (f.unsqueeze(2) + g.unsqueeze(1) - C_mat) / eps + log_alpha + log_beta
        return torch.exp(torch.clamp(logits, max=20.0))  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Dispersion loss
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """
        Token-utilisation entropy maximisation.

        Prevents dictionary collapse by encouraging all N tokens to be
        used roughly equally across the image.

        Steps:
          1. Row-normalise P → proper conditional routing distribution.
          2. Average over HW spatial positions → token utilisation marginal p ∈ Δ^{N-1}.
          3. Compute Shannon entropy H(p) = −Σ_j p_j log p_j.
          4. Return −H(p) as the loss (minimising it maximises diversity).

        The loss is 0 when all tokens are used uniformly and increases toward
        log(N) ≈ 4.85 nats when a single token captures everything.

        Args:
            P : (B, HW, N) routing plan (need not be row-stochastic)

        Returns:
            scalar loss ≥ 0
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

        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)

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
