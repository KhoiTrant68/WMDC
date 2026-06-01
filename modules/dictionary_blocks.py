from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.utils import OLP

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
            OLP(in_dim, dict_dim),
            nn.GELU(),
            OLP(dict_dim, dict_dim),
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
            # Symmetric coherence penalty: off-diagonals should be near 0,
            # both positive (near-duplicate atoms) and negative (anti-aligned
            # near-duplicates) hurt incoherence equally.  ‖sim − I‖² minimises
            # |off-diag| without favouring sign — identical to a Welch-bound
            # / RIP-style frame-coherence loss.
            penalty = (sim_matrix - I).pow(2).mean()

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

        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_eps = nn.Parameter(torch.tensor(math.log(ot_eps)))

        self.log_rho_col = nn.Parameter(torch.tensor(0.0))

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        self.store_attn_probs: bool = store_attn_probs
        self.attn_probs: torch.Tensor | None = None
        self.last_row_mass: torch.Tensor | None = None

        # ── Sinkhorn telemetry (DDP-safe, persistent across checkpoints) ──
        # Reviewer-requested instrumentation: every forward increments the
        # call counter; every fallback path increments the divergence counter
        # and records the eps / rho_col / max|M| that triggered it.  These
        # buffers are registered (not Python ints) so they survive
        # state_dict() round-trips and aggregate correctly under DDP via
        # broadcast_buffers.
        self.register_buffer(
            "_sinkhorn_calls", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_sinkhorn_fallbacks", torch.zeros(1, dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "_last_eps_at_fallback",
            torch.zeros(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_rho_col_at_fallback",
            torch.zeros(1, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_last_max_M_at_fallback",
            torch.zeros(1, dtype=torch.float32),
            persistent=False,
        )
        # Public mirror kept for backward compatibility with older logging.
        self.sinkhorn_divergence_count: int = 0

        # ── Eps warm-up bias ────────────────────────────────────────────
        # Added to log_eps INSIDE `_eps()` to softly enlarge eps during the
        # first few epochs.  After fix #1 raised std(C) from ~0.04 to ~1,
        # initial routing with eps=0.1 is ~22000× peaked, which over-commits
        # to random init features and gets stuck.  The training loop sets
        # this bias > 0 at epoch 0 and anneals it to 0 over a few epochs so
        # learning starts with a smoother, more uniform routing.  Not
        # persistent — a resumed checkpoint always starts at bias = 0.
        self.register_buffer(
            "_log_eps_bias", torch.zeros(1, dtype=torch.float32), persistent=False
        )

    # -----------------------------------------------------------------------
    # Bounded eps — log-space exp with hard clamp
    # -----------------------------------------------------------------------
    #
    # Previous parameterisations and their failure modes:
    #   (a) eps = softplus(log_eps) + floor
    #         → softplus saturates at 0 for negative inputs, so once eps
    #           sits near the floor the gradient through log_eps vanishes.
    #   (b) eps = floor + range · sigmoid(log_eps)            (the pre-fix)
    #         → sigmoid saturates at BOTH ends.  Init log_eps = log(0.1)
    #           sits in the saturating tail (sigmoid' ≈ 0.08), and as the
    #           optimiser tries to drive eps toward the floor sigmoid'
    #           collapses to ~0.007.  eps got stuck at ≈ 0.137 throughout
    #           the 400-epoch run — too large to produce sharp routing on
    #           a cost matrix with std(C) ≈ 1, so column-marginal entropy
    #           pegged at ~93 % of max.
    #
    # Current parameterisation: eps = exp(log_eps), with log_eps clamped to
    # [log(EPS_FLOOR), log(EPS_CAP)] for numerical safety.
    #   • d eps / d log_eps = eps : RELATIVE gradient is identically 1, so
    #     the parameter remains responsive across the full operating range
    #     (no saturation tail).  The optimiser sees the same update size
    #     in log-space whether eps is 0.01 or 1.0.
    #   • Hard clamp only kicks in at the bounds — gradient is unaffected
    #     in the interior.  EPS_FLOOR = 0.005 (10× lower than the legacy
    #     floor) is needed to let Sinkhorn produce genuinely sharp routing.
    EPS_FLOOR: float = 0.005
    EPS_CAP: float = 5.0

    def _eps(self) -> torch.Tensor:
        log_eps_b = (self.log_eps + self._log_eps_bias).clamp(
            min=math.log(self.EPS_FLOOR), max=math.log(self.EPS_CAP)
        )
        return torch.exp(log_eps_b)

    def set_log_eps_bias(self, value: float) -> None:
        """Set the warm-up bias added to log_eps inside _eps()."""
        self._log_eps_bias.fill_(float(value))

    def sinkhorn_telemetry(self) -> dict:
        """Return current fallback statistics.  Safe to call any time."""
        calls = int(self._sinkhorn_calls.item())
        fb = int(self._sinkhorn_fallbacks.item())
        return {
            "calls": calls,
            "fallbacks": fb,
            "fallback_rate": fb / max(calls, 1),
            "last_eps_at_fallback": float(self._last_eps_at_fallback.item()),
            "last_rho_col_at_fallback": float(self._last_rho_col_at_fallback.item()),
            "last_max_M_at_fallback": float(self._last_max_M_at_fallback.item()),
        }

    def _record_fallback(
        self, eps: torch.Tensor, M: torch.Tensor, rho_col: torch.Tensor | None
    ) -> None:
        self._sinkhorn_fallbacks += 1
        self.sinkhorn_divergence_count += 1
        with torch.no_grad():
            self._last_eps_at_fallback.fill_(float(eps.detach()))
            self._last_max_M_at_fallback.fill_(float(M.detach().abs().max()))
            if rho_col is not None:
                self._last_rho_col_at_fallback.fill_(float(rho_col.detach()))

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
        # Cost = −similarity, computed against the UNNORMALISED dictionary
        # tokens.  QueryDictionaryGenerator returns L2-normalised atoms scaled
        # by sqrt(dict_dim) (line 71), so ‖k‖ = sqrt(d).  Re-normalising k —
        # as the pre-fix code did — undoes that scaling and caps std(C) at
        # ~1/sqrt(d) ≈ 0.04 (d=640).  Sinkhorn then needs eps ≪ 0.04 to
        # produce sharp routing, which the bounded `log_eps` parameterisation
        # could not reach, so column-marginal entropy stuck at ~93 % of max
        # and the dictionary collapsed to near-uniform routing.
        #
        # Keeping k at its native sqrt(d) scale and only normalising q gives
        # std(C) ≈ 1 — a ~25× contrast increase at d=640 — which lets the
        # Sinkhorn output respond to the row-entropy regulariser.
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

    def _route_balanced_eot(
        self,
        C_mat: torch.Tensor,
        log_b_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        log_b_override : optional (B, 1, N) tensor that *replaces* the uniform
                         column target marginal. Used by the multi-marginal /
                         conditional-OT slice loop to make later slices avoid
                         dictionary atoms already heavily used by earlier ones.
                         When None (default), behaves identically to balanced
                         Sinkhorn with uniform b = 1/N.
        """
        B, HW, N = C_mat.shape
        self._sinkhorn_calls += 1
        eps = self._eps()
        log_a = -math.log(HW)
        log_b = log_b_override if log_b_override is not None else -math.log(N)
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
            self._record_fallback(eps, M, rho_col=None)
            warnings.warn(
                f"[BalancedEOT] NaN detected (eps={eps.item():.4f}, "
                f"max|M|={float(M.abs().max()):.1f}, "
                f"count={self.sinkhorn_divergence_count}). Falling back to softmax."
            )
            fallback_P = self._route_softmax(C_mat)
            if self.training:
                fallback_P = fallback_P + self.log_eps * 0.0
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
        self,
        C_mat: torch.Tensor,
        rho_flat: torch.Tensor,
        log_b_override: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Unbalanced entropic OT in log-domain.

        log_b_override : optional (B, 1, N) tensor that *replaces* the uniform
                         column target marginal.  See `_route_balanced_eot`.
        """
        B, HW, N = C_mat.shape
        self._sinkhorn_calls += 1
        eps = self._eps()

        rho_col = F.softplus(self.log_rho_col) + 0.01  # strictly positive scalar

        log_a = -math.log(HW)
        log_b = log_b_override if log_b_override is not None else -math.log(N)
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
            self._record_fallback(eps, M, rho_col=rho_col)
            warnings.warn(
                f"[UnbalancedEOT/{self.marginal_div.upper()}] NaN detected "
                f"(eps={eps.item():.4f}, rho_col={rho_col.item():.4f}, "
                f"max|M|={float(M.abs().max()):.1f}, "
                f"count={self.sinkhorn_divergence_count}). Falling back to softmax."
            )
            fallback_P = self._route_softmax(C_mat)
            if self.training:
                fallback_P = fallback_P + (rho_flat * 0.0).sum() + self.log_eps * 0.0
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
        log_b_override: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        log_b_override : optional (B, 1, N) tensor.  When supplied, the column
                         target marginal `b` in the Sinkhorn iterations is set
                         to `exp(log_b_override)` instead of the uniform 1/N.
                         Used by the multi-marginal slice loop in WMDC: the
                         previous slices' column usage shifts later slices'
                         routing targets, enforcing cross-slice dictionary
                         specialisation.  Ignored by softmax routing (which
                         has no column-marginal concept).
        """
        B, _, H, W = x.shape
        HW = H * W

        C_mat = self._cost_matrix(x, k, H, W).float()

        if self.routing_mode == "softmax":
            P = self._route_softmax(C_mat)
        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat, log_b_override=log_b_override)
            P = P * HW
        elif self.routing_mode == "unbalanced_eot":
            assert rho_spatial is not None, "rho_spatial required for unbalanced_eot"
            rho_flat = rho_spatial.view(B, HW).float().clamp(min=0.01)
            P = self._route_unbalanced_eot(
                C_mat, rho_flat, log_b_override=log_b_override
            )
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
            if self.training:
                if self.routing_mode == "unbalanced_eot":
                    column_neg_entropy = column_neg_entropy + self.log_tau * 0.0
                elif self.routing_mode == "balanced_eot":
                    column_neg_entropy = column_neg_entropy + self.log_tau * 0.0
                    column_neg_entropy = column_neg_entropy + self.log_rho_col * 0.0
                else:  # softmax
                    column_neg_entropy = column_neg_entropy + self.log_eps * 0.0
                    column_neg_entropy = column_neg_entropy + self.log_rho_col * 0.0
        else:
            column_neg_entropy = zero
            row_entropy = zero

        # Detached column marginal — used by the multi-marginal slice loop
        # to build log_b_override for the next slice.  Cheap to compute and
        # adds no gradient surface (it is consumed only as a no-grad prior).
        with torch.no_grad():
            col_sum = P.sum(dim=1)  # (B, N)
            col_mass = col_sum / col_sum.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        aux = {
            "column_neg_entropy": column_neg_entropy,
            "row_entropy": row_entropy,
            "row_mass": row_mass,
            "col_mass": col_mass,
            "tv_loss": tv_loss,
        }
        return self.out_proj(out), aux
