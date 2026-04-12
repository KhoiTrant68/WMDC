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

        # Positional encoding: depthwise conv preserves spatial structure.
        self.pos_enc = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim
        )

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True,
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
        context = context.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)

        queries = self.dict_queries.expand(B, -1, -1)  # (B, N, C)
        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)
        return self.proj(out)  # (B, N, dict_dim)


# ---------------------------------------------------------------------------
# UnifiedDictionaryAttention
# ---------------------------------------------------------------------------


class UnifiedDictionaryAttention(nn.Module):
    """
    Unified Dictionary Attention with three routing modes.

    routing_mode options
    --------------------
    'softmax'        — standard temperature-scaled softmax over cost matrix.

    'balanced_eot'   — log-domain Sinkhorn with uniform marginals.
                       Iteration order follows Algorithm 1 of Séjourné et al.
                       2019 (column/g update first, row/f update second).
                       After Sinkhorn, P is scaled by HW so row sums ≈ 1,
                       matching the softmax normalisation convention.

    'unbalanced_eot' — KL-unbalanced Sinkhorn (Séjourné et al. 2019, §4.7.2)
                       with spatially-varying row marginal strength ρ(x).
                       P is scaled by HW after Sinkhorn.  Row sums are then
                       ρ_i/(ρ_i + ε) · HW/(1/HW) ≠ 1 in general: pixels
                       where no dictionary token matches well (high cost
                       everywhere) receive low row mass, attenuating the
                       aggregated feature.  This is the intended "spatial
                       gating" mechanism — normalising rows to 1 would
                       discard it (Séjourné et al. §4.7.2).  The raw row
                       mass per pixel is stored in `last_row_mass` (eval
                       mode only) for ablation studies.
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
        tv_weight: float = 0.0,
    ):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim
        self.tau = tau
        self.ot_eps = ot_eps
        self.iters = iters
        self.tv_weight = tv_weight

        valid = {"softmax", "balanced_eot", "unbalanced_eot"}
        if routing_mode not in valid:
            raise ValueError(f"routing_mode must be one of {valid}")
        self.routing_mode = routing_mode

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GroupNorm(1, dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        # Populated in eval mode only (cleared in training mode) to avoid
        # retaining (B, HW, N) tensors on the autograd graph each step.
        self.attn_probs: torch.Tensor | None = None

        # Row mass map for the unbalanced path — (B, HW) tensor, eval only.
        # Values > 1 indicate pixels strongly matched by a dictionary token;
        # values < 1 indicate low-confidence / outlier regions.
        self.last_row_mass: torch.Tensor | None = None

    # -----------------------------------------------------------------------
    # Cost matrix  C ∈ [0, 2]   (cosine distance, symmetric)
    # -----------------------------------------------------------------------

    def _cost_matrix(
        self, x: torch.Tensor, k: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Cosine-distance cost matrix C[b, hw, n] = 1 − cos(q_hw, k_n) ∈ [0, 2].

        No spatial smoothing is applied — smoothing C would make the distance
        non-metric and break the OT interpretation.
        """
        B = x.shape[0]
        HW = H * W

        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2)  # (B, HW, D)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)  # (B, N, D)

        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, HW, N)
        return C_mat

    # -----------------------------------------------------------------------
    # Spatial TV regularisation on P  (optional, training only)
    # -----------------------------------------------------------------------

    def _spatial_tv(self, P: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Total-variation regulariser on the transport plan, encouraging
        spatially smooth token assignments.

        P : (B, HW, N)
        Returns scalar TV loss (mean over batch and token dims).
        """
        B, HW, N = P.shape
        P_sp = P.transpose(1, 2).view(B, N, H, W)  # (B, N, H, W)
        tv_h = (P_sp[:, :, 1:, :] - P_sp[:, :, :-1, :]).abs().mean()
        tv_w = (P_sp[:, :, :, 1:] - P_sp[:, :, :, :-1]).abs().mean()
        return tv_h + tv_w

    # -----------------------------------------------------------------------
    # Balanced Sinkhorn (log-domain)
    # -----------------------------------------------------------------------

    def _route_balanced_eot(self, C_mat: torch.Tensor) -> torch.Tensor:
        """
        Balanced OT in log-domain.

        Marginals: uniform row (1/HW) and uniform col (1/N).

        Iteration order follows Algorithm 1 of Séjourné et al. 2019:
        column potential (g) is updated first, then row potential (f).
        This is correct — the paper initialises f=0 and runs g first.

        Truncated backprop: the first (iters - 5) iterations run under
        torch.no_grad() to reduce peak memory; gradients flow through
        the last 5 iterations only.
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_a = -math.log(HW)  # log of uniform row marginal
        log_b = -math.log(N)  # log of uniform col marginal

        M = C_mat.float() / eps  # (B, HW, N)

        log_f = C_mat.new_zeros(B, HW, 1)  # (B, HW, 1)
        log_g = C_mat.new_zeros(B, 1, N)  # (B,  1, N)

        # Truncated backpropagation to prevent OOM / unstable gradients
        n_grad_iters = 5
        n_nograd_iters = max(0, self.iters - n_grad_iters)

        if self.training and n_nograd_iters > 0:
            with torch.no_grad():
                for _ in range(n_nograd_iters):
                    # Column update first (Algorithm 1, Séjourné et al. 2019)
                    log_g = log_b - torch.logsumexp(log_f - M, dim=1, keepdim=True)
                    log_f = log_a - torch.logsumexp(log_g - M, dim=2, keepdim=True)

        for _ in range(n_grad_iters if self.training else self.iters):
            # Column update first (Algorithm 1, Séjourné et al. 2019)
            log_g = log_b - torch.logsumexp(log_f - M, dim=1, keepdim=True)
            # Row update:
            log_f = log_a - torch.logsumexp(log_g - M, dim=2, keepdim=True)

        log_P = log_f + log_g - M  # (B, HW, N)

        # exp(-60) ≈ 1e-26, contributing negligibly to row sums.
        P = torch.exp(log_P.clamp(min=-60.0))
        return P

    # -----------------------------------------------------------------------
    # KL-unbalanced Sinkhorn with spatially-varying ρ (log-domain)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        KL-unbalanced Sinkhorn OT (Séjourné et al. 2019, §3.2 and §4.7.2).

        Row marginal strength: per-pixel ρ_i  (spatially varying).
        Col marginal strength: mean(ρ) shared across all dictionary tokens.

        The Aprox operator for ρKL is Aprox(p) = (ρ/(ρ+ε)) · p (Table 1),
        which in log-domain becomes the direct scaling used below.

        Iteration order follows Algorithm 1: column (g) update first, then
        row (f) update.  Truncated backprop as in the balanced path.
        """
        B, HW, N = C_mat.shape
        eps = self.ot_eps
        log_a = -math.log(HW)
        log_b = -math.log(N)

        M = C_mat.float() / eps  # (B, HW, N)

        # Per-pixel row shrinkage: s_i = ρ_i / (ρ_i + ε) ∈ (0, 1)
        #   ρ → ∞ : s → 1 → hard marginal (balanced limit)
        #   ρ → 0 : s → 0 → no marginal constraint (free transport)
        shrink_row = (rho_flat / (rho_flat + eps)).unsqueeze(2)  # (B, HW, 1)

        # Shared column shrinkage from mean row strength.
        rho_mean = rho_flat.mean(dim=1, keepdim=True)  # (B, 1)
        shrink_col = (rho_mean / (rho_mean + eps)).unsqueeze(2)  # (B, 1, 1)

        log_f = C_mat.new_zeros(B, HW, 1)
        log_g = C_mat.new_zeros(B, 1, N)

        # Truncated backpropagation
        n_grad_iters = 5
        n_nograd_iters = max(0, self.iters - n_grad_iters)

        if self.training and n_nograd_iters > 0:
            with torch.no_grad():
                for _ in range(n_nograd_iters):
                    # Column update first (Algorithm 1, Séjourné et al. 2019)
                    raw_g = log_b - torch.logsumexp(log_f - M, dim=1, keepdim=True)
                    log_g = shrink_col * raw_g
                    raw_f = log_a - torch.logsumexp(log_g - M, dim=2, keepdim=True)
                    log_f = shrink_row * raw_f

        for _ in range(n_grad_iters if self.training else self.iters):
            # Column update first (Algorithm 1, Séjourné et al. 2019)
            raw_g = log_b - torch.logsumexp(log_f - M, dim=1, keepdim=True)
            log_g = shrink_col * raw_g
            raw_f = log_a - torch.logsumexp(log_g - M, dim=2, keepdim=True)
            log_f = shrink_row * raw_f

        log_P = log_f + log_g - M  # (B, HW, N)
        P = torch.exp(log_P.clamp(min=-60.0))
        return P

    # -----------------------------------------------------------------------
    # Softmax routing
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        return F.softmax(-C_mat / self.tau, dim=-1)

    # -----------------------------------------------------------------------
    # Dispersion loss  (returns −H in BITS)
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """
        Returns  −H(m)  where H(m) is the Shannon entropy of the column-
        marginal distribution m_j = Σ_i P_ij / Z.

        Entropy is in bits to be commensurate with bpp_loss units.
        """
        marginal = P.sum(dim=1)  # (B, N)
        marginal = marginal / (marginal.sum(dim=1, keepdim=True).clamp(min=1e-8))
        _log2e = 1.0 / math.log(2)
        H = -(marginal * marginal.clamp(min=1e-8).log()).sum(dim=1).mean() * _log2e
        return -H  # negative entropy in bits; caller subtracts to maximise H

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = x.shape
        HW = H * W

        # Cost matrix always computed in float32 regardless of input dtype.
        C_mat = self._cost_matrix(x, k, H, W).float()

        # ── Routing ──────────────────────────────────────────────────────────
        if self.routing_mode == "softmax":
            P = self._route_softmax(C_mat)

        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)
            # Scale so rows sum to ~1 (balanced marginals are uniform 1/HW
            # before scaling; this matches the softmax normalisation convention).
            P = P * HW

        elif self.routing_mode == "unbalanced_eot":
            rho_flat = rho_spatial.view(B, HW).float().clamp(min=0.01)
            P = self._route_unbalanced_eot(C_mat, rho_flat)
            # Scale by HW as in the balanced case.  Row sums are NOT forced to
            # 1 here — unbalanced OT intentionally relaxes the row marginal
            # constraint (Séjourné et al. §4.7.2).  Low-confidence pixels
            # (no good dictionary match) yield row sums < 1, attenuating the
            # aggregated feature (spatial gating).  High-affinity pixels may
            # exceed 1.  The GroupNorm in out_proj stabilises the resulting
            # magnitude variation before the final projection.
            P = P * HW

            # Store raw row mass in eval mode for ablation analysis.
            # Callers can inspect last_row_mass to verify the gating mechanism
            # correlates with image texture (high-frequency regions → low ρ
            # → low row mass → attenuated dict feature).
            if not self.training:
                self.last_row_mass = P.sum(dim=-1).detach()  # (B, HW)
            else:
                self.last_row_mass = None

        else:
            raise RuntimeError(f"Unknown routing_mode: {self.routing_mode}")

        # Store full attention plan in eval mode only.
        # During training this tensor is large ((B, HW, N) per slice × 5 slices)
        # and is never read, so we avoid retaining it on the graph.
        if not self.training:
            self.attn_probs = P.detach()
        else:
            self.attn_probs = None

        # ── Spatial TV regularisation (optional, training only) ───────────────
        tv_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.tv_weight > 0.0:
            tv_loss = self._spatial_tv(P, H, W) * self.tv_weight

        # ── Value aggregation ─────────────────────────────────────────────────
        # P: (B, HW, N)   v: (B, N, dict_dim)
        out = torch.bmm(P, v).transpose(1, 2).contiguous().view(B, -1, H, W)

        # ── Dispersion loss (bits, training only) ─────────────────────────────
        if calc_disp:
            disp_loss = self._dispersion_loss(P) + tv_loss
        else:
            disp_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return self.out_proj(out), disp_loss
