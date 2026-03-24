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

        self.dict_queries = nn.Parameter(torch.randn(1, dict_num, in_dim))

        self.pos_enc = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)

        self.proj = nn.Sequential(
            nn.Linear(in_dim, dict_dim),
            nn.GELU(),
            nn.Linear(dict_dim, dict_dim),
        )

    def forward(self, z_hat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = z_hat.shape
        context = z_hat + self.pos_enc(z_hat)
        context = context.view(B, C, H * W).transpose(1, 2)

        queries = self.dict_queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)
        return self.proj(out)


# ---------------------------------------------------------------------------
# UnifiedDictionaryAttention
# ---------------------------------------------------------------------------


class UnifiedDictionaryAttention(nn.Module):
    """
    Unified Dictionary Attention with three routing modes.

    routing_mode options:
      'softmax'        — standard temperature-scaled softmax over cost matrix
      'balanced_eot'   — log-domain Sinkhorn with uniform marginals
      'unbalanced_eot' — KL-unbalanced Sinkhorn (Séjourné et al. 2019) with
                         spatially-varying row marginal strength ρ(x)
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
        self.tau = tau
        self.ot_eps = ot_eps
        self.iters = iters

        valid = {"softmax", "balanced_eot", "unbalanced_eot"}
        if routing_mode not in valid:
            raise ValueError(f"routing_mode must be one of {valid}")
        self.routing_mode = routing_mode

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)

        self.spatial_smooth = nn.Conv2d(
            dict_num,
            dict_num,
            kernel_size=3,
            padding=1,
            groups=dict_num,
            bias=False,
            padding_mode="zeros",
        )
        # Initialise as near-identity so smoothing starts conservatively
        self.spatial_smooth.weight.data[:, 0, 1, 1] = 1.0  # centre pixel

        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GroupNorm(1, dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        self.attn_probs: torch.Tensor = None

    # -----------------------------------------------------------------------
    # Cost matrix
    # -----------------------------------------------------------------------

    def _cost_matrix(
        self, x: torch.Tensor, k: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Smoothed cosine-distance cost matrix C ∈ [0, 2].
        """
        B = x.shape[0]
        HW = H * W

        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2)  # (B, HW, dict_dim)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)  # (B, dict_num, dict_dim)

        # C_mat[b, hw, n] = 1 - cos(q_hw, k_n)  ∈ [0, 2]
        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, HW, N)

        # Spatial smoothing on cost: encourages neighbouring pixels to use
        # similar dictionary tokens (piecewise-smooth prior).
        C_sp = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_sp = self.spatial_smooth(C_sp)
        C_mat = C_sp.view(B, self.dict_num, HW).transpose(1, 2).contiguous()
        return C_mat  # (B, HW, N)

    # -----------------------------------------------------------------------
    # Log-domain Sinkhorn: balanced
    # -----------------------------------------------------------------------

    def _route_balanced_eot(self, C_mat: torch.Tensor) -> torch.Tensor:
        """
        Balanced Sinkhorn OT in log-domain.

        Marginals: uniform row (each pixel gets equal total mass = 1/HW)
                   uniform col (each dictionary token gets equal mass = 1/N)

        Potentials (f, g) satisfy:
            f_i = log(1/HW) − log∑_j exp((g_j − C_ij)/ε)
            g_j = log(1/N)  − log∑_i exp((f_i − C_ij)/ε)

        Initialising both at zero (log 1 = 0) is a valid warm start.
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)  # log of uniform row marginal
        log_beta = -math.log(N)  # log of uniform col marginal

        M = C_mat / eps  # (B, HW, N)

        log_f = C_mat.new_zeros(B, HW)
        log_g = C_mat.new_zeros(B, N)

        for _ in range(self.iters):
            # Update column potentials from current row potentials
            # log_g_j = log_beta − logsumexp_i( log_f_i − M_ij )
            inner_f = log_f.unsqueeze(2) - M  # (B, HW, N)
            log_g = log_beta - torch.logsumexp(inner_f, dim=1)  # (B, N)

            # Update row potentials from updated column potentials
            # log_f_i = log_alpha − logsumexp_j( log_g_j − M_ij )
            inner_g = log_g.unsqueeze(1) - M  # (B, HW, N)
            log_f = log_alpha - torch.logsumexp(inner_g, dim=2)  # (B, HW)

        # Transport plan: P_ij = exp(f_i + g_j − M_ij)
        log_P = log_f.unsqueeze(2) + log_g.unsqueeze(1) - M  # (B, HW, N)
        return torch.exp(log_P)

    # -----------------------------------------------------------------------
    # Log-domain Sinkhorn: KL-unbalanced (spatially-varying rho)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        KL-unbalanced Sinkhorn OT (Séjourné et al., 2019).

        For row i: KL-proximal relaxation with strength ρ_i.
            f_i ← [ρ_i/(ρ_i+ε)] · [log_alpha − logsumexp_j(g_j − M_ij)]

        For columns: shared ρ̄ = mean(ρ) for a symmetric treatment.
            g_j ← [ρ̄/(ρ̄+ε)] · [log_beta − logsumexp_i(f_i − M_ij)]

        The shrinkage factor s = ρ/(ρ+ε) ∈ (0,1) softens the marginal
        constraints: s→0 means no constraint, s→1 recovers balanced OT.

        Crucially, the shrinkage is applied to each *update step*, not
        accumulated across iterations, so potentials converge rather than
        collapsing to zero.
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)
        log_beta = -math.log(N)

        M = C_mat / eps  # (B, HW, N)

        # Per-pixel row shrinkage: (B, HW)
        shrink_row = rho_flat / (rho_flat + eps)  # ∈ (0, 1)

        # Shared column shrinkage derived from mean row strength: (B, 1)
        rho_mean = rho_flat.mean(dim=1, keepdim=True)  # (B, 1)
        shrink_col = rho_mean / (rho_mean + eps)  # (B, 1)

        log_f = C_mat.new_zeros(B, HW)
        log_g = C_mat.new_zeros(B, N)

        for _ in range(self.iters):
            # Column update: apply shrinkage to the raw dual update
            inner_f = log_f.unsqueeze(2) - M  # (B, HW, N)
            raw_log_g = log_beta - torch.logsumexp(inner_f, dim=1)  # (B, N)
            log_g = shrink_col * raw_log_g  # (B, N)

            # Row update: per-pixel shrinkage
            inner_g = log_g.unsqueeze(1) - M  # (B, HW, N)
            raw_log_f = log_alpha - torch.logsumexp(inner_g, dim=2)  # (B, HW)
            log_f = shrink_row * raw_log_f  # (B, HW)

        log_P = log_f.unsqueeze(2) + log_g.unsqueeze(1) - M
        return torch.exp(log_P)

    # -----------------------------------------------------------------------
    # Routing: softmax
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        return F.softmax(-C_mat / self.tau, dim=-1)

    # -----------------------------------------------------------------------
    # Dispersion loss
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """
        Encourages diverse dictionary usage by maximising the entropy of the
        column-marginal distribution of P.

        The column marginal m_j = Σ_i P_ij measures how much each dictionary
        token j is used across all spatial locations.  High entropy → all
        tokens used uniformly.  Low entropy → tokens collapse.

        We return the *negative* entropy so that the caller can treat this as
        a standard minimisation loss.  The training loop should *subtract*
        this term (or equivalently pass a negative weight), which maximises H.

        Returns:
            neg_H : scalar, −H(m)  (negative entropy, to be subtracted from
                    the total loss or added with a negative weight coefficient)
        """
        # Column marginal: sum over spatial dim → (B, N)
        marginal = P.sum(dim=1)
        # Normalise to a probability simplex
        marginal = marginal / (marginal.sum(dim=1, keepdim=True) + 1e-8)  # (B, N)
        # Shannon entropy H = −Σ_j m_j log(m_j), averaged over batch
        H = -(marginal * marginal.clamp(min=1e-8).log()).sum(dim=1).mean()
        return -H  # negative entropy; caller subtracts this to maximise H

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
        B, _, H, W = x.shape
        HW = H * W

        C_mat = self._cost_matrix(x, k, H, W)

        # ── Routing ──────────────────────────────────────────────────────────
        if self.routing_mode == "softmax":
            P = self._route_softmax(C_mat)

        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)

        elif self.routing_mode == "unbalanced_eot":
            rho_flat = rho_spatial.view(B, HW).clamp(min=0.01)
            P = self._route_unbalanced_eot(C_mat, rho_flat)

        else:
            raise RuntimeError(f"Unknown routing_mode: {self.routing_mode}")

        self.attn_probs = P.detach()

        # ── Value aggregation ─────────────────────────────────────────────────
        # P: (B, HW, N),  v: (B, N, dict_dim)
        out = torch.bmm(P, v).transpose(1, 2).contiguous().view(B, -1, H, W)

        # ── Dispersion loss ───────────────────────────────────────────────────
        # Only compute during training; returns negative entropy (to subtract).
        if self.training and calc_disp:
            disp_loss = self._dispersion_loss(P)
        else:
            disp_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return self.out_proj(out), disp_loss
