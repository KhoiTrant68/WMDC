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

        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)

        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))

        C_sp = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_sp = self.spatial_smooth(C_sp)
        C_mat = C_sp.view(B, self.dict_num, HW).transpose(1, 2).contiguous()
        return C_mat

    # -----------------------------------------------------------------------
    # Log-domain Sinkhorn: balanced
    # -----------------------------------------------------------------------

    def _route_balanced_eot(self, C_mat: torch.Tensor) -> torch.Tensor:
        """
        Balanced Sinkhorn OT in log-domain mathematically corrected.
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)
        log_beta = -math.log(N)

        M = C_mat / eps

        log_f = torch.zeros(B, HW, device=C_mat.device, dtype=C_mat.dtype)
        log_g = torch.zeros(B, N, device=C_mat.device, dtype=C_mat.dtype)

        for _ in range(self.iters):
            inner_g = log_f.unsqueeze(2) - M
            log_g = log_beta - torch.logsumexp(inner_g, dim=1)

            inner_f = log_g.unsqueeze(1) - M
            log_f = log_alpha - torch.logsumexp(inner_f, dim=2)

        log_P = log_f.unsqueeze(2) + log_g.unsqueeze(1) - M
        return torch.exp(log_P)

    # -----------------------------------------------------------------------
    # Log-domain Sinkhorn: KL-unbalanced (spatially-varying rho)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        KL-unbalanced Sinkhorn OT in log-domain mathematically corrected.
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_alpha = -math.log(HW)
        log_beta = -math.log(N)

        M = C_mat / eps

        shrink_row = rho_flat / (rho_flat + eps)

        rho_mean = rho_flat.mean(dim=1, keepdim=True)
        shrink_col = rho_mean / (rho_mean + eps)

        log_f = torch.zeros(B, HW, device=C_mat.device, dtype=C_mat.dtype)
        log_g = torch.zeros(B, N, device=C_mat.device, dtype=C_mat.dtype)

        for _ in range(self.iters):
            inner_g = log_f.unsqueeze(2) - M
            raw_g = log_beta - torch.logsumexp(inner_g, dim=1)
            log_g = shrink_col * raw_g

            inner_f = log_g.unsqueeze(1) - M
            raw_f = log_alpha - torch.logsumexp(inner_f, dim=2)
            log_f = shrink_row * raw_f

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
        target_marginal = P.sum(dim=1)
        target_marginal = target_marginal / (
            target_marginal.sum(dim=1, keepdim=True) + 1e-8
        )
        H = -(target_marginal * (target_marginal + 1e-8).log()).sum(dim=1).mean()
        return -H

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
            if self.training:
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

        self.attn_probs = P.detach()

        # ── Value aggregation ─────────────────────────────────────────────────
        out = torch.bmm(P, v).transpose(1, 2).contiguous().view(B, -1, H, W)

        # ── Dispersion loss ───────────────────────────────────────────────────
        if self.training and calc_disp:
            disp_loss = self._dispersion_loss(P)
        else:
            disp_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        return self.out_proj(out), disp_loss
