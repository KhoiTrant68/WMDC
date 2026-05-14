from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# QueryDictionaryGenerator
# ---------------------------------------------------------------------------


class QueryDictionaryGenerator(nn.Module):
    """
    Generates a content-adaptive dictionary from the quantised hyperprior z_hat.
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
        self._sqrt_dict_dim = math.sqrt(dict_dim)

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

    def forward(self, z_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = z_hat.shape
        context = z_hat + self.pos_enc(z_hat)
        context = context.view(B, C, H * W).transpose(1, 2)

        queries = self.dict_queries.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)

        projected = self.proj(out)
        dt = F.normalize(projected, p=2, dim=-1)

        penalty = torch.tensor(0.0, device=z_hat.device, dtype=z_hat.dtype)
        if self.training:
            sim_matrix = torch.bmm(dt, dt.transpose(1, 2))
            I = torch.eye(self.dict_num, device=dt.device).unsqueeze(0)
            penalty = F.relu(sim_matrix - I).pow(2).mean()

        return dt * self._sqrt_dict_dim, penalty


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
        marginal_div: str = "kl",
        tv_weight: float = 0.0,
        store_attn_probs: bool = False,
        chunk_threshold: int = 2048,
    ):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim
        self._sqrt_dict_dim = math.sqrt(dict_dim)
        self.iters = iters
        self.tv_weight = tv_weight
        self.chunk_threshold = chunk_threshold

        self.n_grad_iters = max(5, iters // 3)
        self.n_nograd_iters = max(0, iters - self.n_grad_iters)

        valid = {"softmax", "balanced_eot", "unbalanced_eot"}
        if routing_mode not in valid:
            raise ValueError(f"routing_mode must be one of {valid}")
        self.routing_mode = routing_mode

        valid_div = {"kl", "tv"}
        if marginal_div not in valid_div:
            raise ValueError(f"marginal_div must be one of {valid_div}")
        self.marginal_div = marginal_div

        # log_tau (softmax temperature) — always present; serves as a fallback
        # in NaN cases for the OT branches.
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))

        # OT-specific parameters: only register when actually used so that
        # DDP can run with find_unused_parameters=False.
        if routing_mode in ("balanced_eot", "unbalanced_eot"):
            self.log_eps = nn.Parameter(torch.tensor(math.log(ot_eps)))
        else:
            self.register_parameter("log_eps", None)

        if routing_mode == "unbalanced_eot":
            self.log_rho_col = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_parameter("log_rho_col", None)

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        self.store_attn_probs: bool = store_attn_probs
        self.attn_probs: torch.Tensor | None = None
        self.last_row_mass: torch.Tensor | None = None
        self.sinkhorn_divergence_count: int = 0

    # -----------------------------------------------------------------------
    # Chunked logsumexp along the spatial dimension
    # -----------------------------------------------------------------------

    def _chunked_logsumexp_spatial(
        self, x: torch.Tensor, chunk: int = 256
    ) -> torch.Tensor:
        """
        Memory-efficient logsumexp(x, dim=1) for x: (B, HW, N).

        Standard logsumexp materialises the full (B, HW, N) shifted tensor.
        At 768×512 with N=128 and B=8 this is ~6 MB per slice.  Chunking
        along HW processes `chunk` spatial positions at a time, keeping peak
        memory proportional to `chunk` rather than HW.

        Returns: (B, 1, N)
        """
        B, HW, N = x.shape
        result = x.new_full((B, 1, N), float("-inf"))
        for start in range(0, HW, chunk):
            end = min(start + chunk, HW)
            chunk_lse = x[:, start:end, :].logsumexp(dim=1, keepdim=True)  # (B, 1, N)
            result = torch.logaddexp(result, chunk_lse)
        return result

    def _logsumexp_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Wrapper: use chunked version when HW exceeds chunk_threshold.
        x: (B, HW, N) -> returns (B, 1, N)
        """
        HW = x.shape[1]
        if HW > self.chunk_threshold:
            return self._chunked_logsumexp_spatial(x, chunk=256)
        return x.logsumexp(dim=1, keepdim=True)

    # -----------------------------------------------------------------------
    # Cost matrix
    # -----------------------------------------------------------------------

    def _cost_matrix(
        self, x: torch.Tensor, k: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        B = x.shape[0]
        HW = H * W
        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        return 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))

    # -----------------------------------------------------------------------
    # Spatial TV regularisation
    # -----------------------------------------------------------------------

    def _spatial_tv(self, P: torch.Tensor, H: int, W: int) -> torch.Tensor:
        row_sum = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        P_norm = P / row_sum
        B, HW, N = P_norm.shape
        P_sp = P_norm.transpose(1, 2).view(B, N, H, W)
        tv_h = (P_sp[:, :, 1:, :] - P_sp[:, :, :-1, :]).abs().mean()
        tv_w = (P_sp[:, :, :, 1:] - P_sp[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    # -----------------------------------------------------------------------
    # Balanced Sinkhorn (log-domain)
    # -----------------------------------------------------------------------

    def _route_balanced_eot(self, C_mat: torch.Tensor) -> torch.Tensor:
        B, HW, N = C_mat.shape
        eps = F.softplus(self.log_eps) + 0.01
        log_a = -math.log(HW)
        log_b = -math.log(N)
        M = C_mat / eps
        log_f = C_mat.new_zeros(B, HW, 1)
        log_g = C_mat.new_zeros(B, 1, N)

        if self.training and self.n_nograd_iters > 0:
            with torch.no_grad():
                for _ in range(self.n_nograd_iters):
                    log_g = log_b - self._logsumexp_spatial(log_f - M)
                    log_f = log_a - (log_g - M).logsumexp(dim=2, keepdim=True)

        n_iters = self.n_grad_iters if self.training else self.iters
        for _ in range(n_iters):
            log_g = log_b - self._logsumexp_spatial(log_f - M)
            log_f = log_a - (log_g - M).logsumexp(dim=2, keepdim=True)

        log_P = log_f + log_g - M

        if torch.isnan(log_P).any():
            self.sinkhorn_divergence_count += 1
            warnings.warn(
                f"[BalancedEOT] NaN detected (eps={eps.item():.4f}, "
                f"count={self.sinkhorn_divergence_count}). Falling back to softmax."
            )
            # Fallback keeps log_eps in graph so DDP find_unused_parameters=False
            # remains happy when OT branch silently degrades.
            fallback_P = self._route_softmax(C_mat) + self.log_eps * 0.0
            return fallback_P / HW

        return torch.exp(log_P.clamp(min=-60.0))

    # -----------------------------------------------------------------------
    # Shared Sinkhorn loop runner
    # -----------------------------------------------------------------------

    def _sinkhorn_loop(self, log_f, log_g, M, log_a, log_b, col_fn, row_fn):
        """
        Run warm-up (no-grad) + grad Sinkhorn iterations.
        Uses chunked logsumexp for the spatial (dim=1) direction.
        """
        if self.training and self.n_nograd_iters > 0:
            with torch.no_grad():
                for _ in range(self.n_nograd_iters):
                    lse_g = self._logsumexp_spatial(log_f - M)
                    log_g = log_b - col_fn(lse_g)
                    lse_f = (log_g - M).logsumexp(dim=2, keepdim=True)
                    log_f = log_a - row_fn(lse_f)

        n_iters = self.n_grad_iters if self.training else self.iters
        for _ in range(n_iters):
            lse_g = self._logsumexp_spatial(log_f - M)
            log_g = log_b - col_fn(lse_g)
            lse_f = (log_g - M).logsumexp(dim=2, keepdim=True)
            log_f = log_a - row_fn(lse_f)

        return log_f, log_g

    # -----------------------------------------------------------------------
    # Unbalanced Sinkhorn
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        Unbalanced entropic OT in log-domain.
        """
        B, HW, N = C_mat.shape
        eps = F.softplus(self.log_eps) + 0.01

        rho_col = F.softplus(self.log_rho_col) + 0.01  # strictly positive scalar

        log_a = -math.log(HW)
        log_b = -math.log(N)
        M = C_mat / eps

        log_f = C_mat.new_zeros(B, HW, 1)
        log_g = C_mat.new_zeros(B, 1, N)

        if self.marginal_div == "kl":
            shrink_row = (rho_flat / (rho_flat + eps)).unsqueeze(2)  # (B, HW, 1)
            shrink_col = rho_col / (rho_col + eps)
            log_f, log_g = self._sinkhorn_loop(
                log_f,
                log_g,
                M,
                log_a,
                log_b,
                col_fn=lambda lse: shrink_col * lse,
                row_fn=lambda lse: shrink_row * lse,
            )
        else:  # tv
            rpe_row = (rho_flat / eps).unsqueeze(2)  # (B, HW, 1)
            rpe_col = rho_col / eps
            log_f, log_g = self._sinkhorn_loop(
                log_f,
                log_g,
                M,
                log_a,
                log_b,
                col_fn=lambda lse: torch.clamp(lse, -rpe_col, rpe_col),
                row_fn=lambda lse: torch.clamp(lse, -rpe_row, rpe_row),
            )

        log_P = log_f + log_g - M

        if torch.isnan(log_P).any():
            self.sinkhorn_divergence_count += 1
            warnings.warn(
                f"[UnbalancedEOT/{self.marginal_div.upper()}] NaN detected "
                f"(eps={eps.item():.4f}, rho_col={rho_col.item():.4f}, "
                f"count={self.sinkhorn_divergence_count}). Falling back to softmax."
            )
            # Fallback keeps OT-specific params in the autograd graph so DDP
            # with find_unused_parameters=False does not error.
            fallback_P = (
                self._route_softmax(C_mat)
                + (rho_flat * 0.0).sum()
                + self.log_eps * 0.0
                + self.log_rho_col * 0.0
            )
            return fallback_P / HW

        return torch.exp(log_P.clamp(min=-60.0))

    # -----------------------------------------------------------------------
    # Softmax routing
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        tau = F.softplus(self.log_tau) + 0.01
        return F.softmax(-C_mat / tau, dim=-1)

    # -----------------------------------------------------------------------
    # Entropy signals
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """Negative column entropy -H_col (bits). Minimise to maximise H_col."""
        marginal = P.sum(dim=1)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp(min=1e-8)
        H = -(marginal * torch.log2(marginal.clamp(min=1e-8))).sum(dim=1).mean()
        return -H

    @staticmethod
    def _row_entropy(P: torch.Tensor) -> torch.Tensor:
        """
        Mass-weighted mean per-pixel row entropy H_row (bits, >= 0).
        """
        row_sums = P.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # (B, HW, 1)
        P_norm = P / row_sums
        H_per_pixel = -(P_norm * torch.log2(P_norm.clamp(min=1e-8))).sum(
            dim=-1
        )  # (B, HW)

        weights = row_sums.squeeze(-1).detach().clamp(0.0, 1.0)  # (B, HW)
        denom = weights.sum().clamp(min=1e-8)
        return (H_per_pixel * weights).sum() / denom

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        rho_spatial: torch.Tensor | None,
        calc_disp: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        B, _, H, W = x.shape
        HW = H * W

        C_mat = self._cost_matrix(x, k, H, W).float()

        if self.routing_mode == "softmax":
            P = self._route_softmax(C_mat)
        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)
            P = P * HW
        elif self.routing_mode == "unbalanced_eot":
            assert rho_spatial is not None, "rho_spatial required for unbalanced_eot"
            rho_flat = rho_spatial.view(B, HW).float().clamp(min=0.01)
            P = self._route_unbalanced_eot(C_mat, rho_flat)
            P = P * HW
        else:
            raise RuntimeError(f"Unknown routing_mode: {self.routing_mode!r}")

        if not self.training and self.store_attn_probs:
            self.attn_probs = P.detach()
        else:
            self.attn_probs = None

        if self.routing_mode == "unbalanced_eot":
            row_mass = P.sum(dim=-1)
            if not self.training and self.store_attn_probs:
                self.last_row_mass = row_mass.detach()
            else:
                self.last_row_mass = None
        else:
            row_mass = P.new_ones(B, HW)
            if not self.training:
                self.last_row_mass = None

        tv_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.tv_weight > 0.0:
            tv_loss = self._spatial_tv(P, H, W) * self.tv_weight

        v_norm = F.normalize(v, p=2, dim=-1) * self._sqrt_dict_dim
        out = torch.bmm(P, v_norm).transpose(1, 2).contiguous().view(B, -1, H, W)

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if calc_disp:
            column_neg_entropy = self._dispersion_loss(P)
            row_entropy = self._row_entropy(P)
            # In the OT branches, log_tau is unused unless we hit a NaN fallback;
            # keep it in the graph for DDP find_unused_parameters=False.
            if self.training and self.routing_mode in (
                "balanced_eot",
                "unbalanced_eot",
            ):
                column_neg_entropy = column_neg_entropy + self.log_tau * 0.0
        else:
            column_neg_entropy = zero
            row_entropy = zero

        aux = {
            "column_neg_entropy": column_neg_entropy,
            "row_entropy": row_entropy,
            "row_mass": row_mass,
            "tv_loss": tv_loss,
        }
        return self.out_proj(out), aux
