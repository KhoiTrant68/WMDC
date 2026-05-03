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
        self._sqrt_dict_dim = math.sqrt(dict_dim)

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
            sim_off_diag = sim_matrix - I
            penalty = F.relu(sim_off_diag).pow(2).mean()

        return dt * self._sqrt_dict_dim, penalty


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

    'unbalanced_eot' — Unbalanced Sinkhorn with spatially-varying row marginal
                       strength ρ(x).  The marginal divergence type is set by
                       `marginal_div`:

                       'kl'  (KL divergence, Séjourné et al. 2019):
                           Aprox_KL(x) = ρ/(ρ+ε) · x   (multiplicative shrinkage)
                           log_g ← log_b − shrink_col · lse(log_f − M, rows)
                           log_f ← log_a − shrink_row · lse(log_g − M, cols)
                           Smooth spatial gating: row sums taper continuously
                           from 0 (outlier) to 1 (perfect match).

                       'tv'  (Total Variation divergence, Séjourné et al. 2022):
                           Aprox_TV(x) = clamp(x, −ρ, +ρ)   (hard clip)
                           log_g ← log_b − clamp(lse(log_f − M, rows), −ρ/ε, ρ/ε)
                           log_f ← log_a − clamp(lse(log_g − M, cols), −ρ/ε, ρ/ε)
                           Sharp spatial gating: pixels either match the dict
                           (row_sum ≈ 1) or are cleanly suppressed (row_sum ≈ 0),
                           with no intermediate values.  Recommended for crisp
                           object / texture boundaries.

                       P is scaled by HW after Sinkhorn.  Row sums are NOT forced
                       to 1 — unbalanced OT intentionally relaxes the row marginal.
                       The raw row mass is stored in `last_row_mass` (eval only).

    Parameters
    ----------
    input_dim   : channels of the query feature map x
    output_dim  : output channels (= slice_ch in WMDC)
    dict_num    : number of dictionary tokens N
    dict_dim    : dimension of each token embedding D
    tau         : initial softmax temperature (only for 'softmax' mode)
    ot_eps      : initial Sinkhorn entropic regularisation ε
    iters       : total Sinkhorn iterations
    routing_mode: one of {'softmax', 'balanced_eot', 'unbalanced_eot'}
    marginal_div: divergence for row/col marginals in unbalanced_eot — 'kl' or 'tv'
    tv_weight   : spatial TV weight on P (0 = disabled)
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
    ):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim
        self._sqrt_dict_dim = math.sqrt(dict_dim)
        self.iters = iters
        self.tv_weight = tv_weight

        # Truncated BPTT: run (iters - n_grad_iters) steps without grad,
        # then n_grad_iters steps with grad.  Balances speed vs. gradient quality.
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

        # Learnable log-scale parameters so ε / τ stay positive during training
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.log_eps = nn.Parameter(torch.tensor(math.log(ot_eps)))

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

        # Eval-only storage — cleared every forward pass in training mode
        # to avoid retaining large (B, HW, N) tensors on the autograd graph.
        self.attn_probs: torch.Tensor | None = None

        # Row mass map for the unbalanced path — (B, HW) tensor, eval only.
        # Values > 1 indicate pixels strongly matched by a dictionary token;
        # values < 1 indicate low-confidence / outlier regions.
        self.last_row_mass: torch.Tensor | None = None

        # Divergence counter for diagnostic logging
        self.sinkhorn_divergence_count: int = 0

    # -----------------------------------------------------------------------
    # Cost matrix  C ∈ [0, 2]   (cosine distance)
    # -----------------------------------------------------------------------

    def _cost_matrix(
        self, x: torch.Tensor, k: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Cosine-distance cost matrix.

        C[b, hw, n] = 1 − <q_hw, k_n>  ∈ [0, 2]

        Both q and k are L2-normalised before the inner product so the
        distance is metric and bounded, which is required for the OT
        interpretation to be valid.

        Parameters
        ----------
        x : (B, input_dim, H, W)   — query feature map
        k : (B, N, dict_dim)       — dictionary keys (from k_proj)
        H, W : spatial dimensions of x

        Returns
        -------
        C_mat : (B, HW, N)  in float32
        """
        B = x.shape[0]
        HW = H * W

        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2)  # (B, HW, D)
        q_norm = F.normalize(q, p=2, dim=-1)  # unit sphere
        k_norm = F.normalize(k, p=2, dim=-1)  # (B, N, D)

        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, HW, N)
        return C_mat

    # -----------------------------------------------------------------------
    # Spatial TV regularisation on P  (optional, training only)
    # -----------------------------------------------------------------------

    def _spatial_tv(self, P: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Isotropic total-variation on the transport plan, encouraging
        spatially smooth token assignments.

        P : (B, HW, N)
        Returns scalar mean TV (mean over batch and token dims).
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
        Balanced entropic OT in log-domain.

        Marginals: uniform row a = 1/HW, uniform col b = 1/N.
            log_g ← log_b − logsumexp(log_f − M, dim=rows)   [column update]
            log_f ← log_a − logsumexp(log_g − M, dim=cols)   [row update]

        where M = C/ε and log_P = log_f + log_g − M.

        Initialisation: log_f = 0, log_g = 0  (absorbs uniform marginal).

        Parameters
        ----------
        C_mat : (B, HW, N)  cost matrix in float32, values in [0, 2]

        Returns
        -------
        P : (B, HW, N)  transport plan (NOT yet scaled by HW)
        """
        B, HW, N = C_mat.shape
        eps = F.softplus(self.log_eps) + 0.01
        log_a, log_b = -math.log(HW), -math.log(N)
        M = C_mat / eps
        log_f, log_g = C_mat.new_zeros(B, HW, 1), C_mat.new_zeros(B, 1, N)

        if self.training and self.n_nograd_iters > 0:
            with torch.no_grad():
                for _ in range(self.n_nograd_iters):
                    # Column update first
                    log_g = log_b - torch.logsumexp(log_f - M, dim=1, keepdim=True)
                    log_f = log_a - torch.logsumexp(log_g - M, dim=2, keepdim=True)

        # ── Grad iterations ──
        n_iters = self.n_grad_iters if self.training else self.iters
        for _ in range(n_iters):
            log_g = log_b - torch.logsumexp(log_f - M, dim=1, keepdim=True)
            log_f = log_a - torch.logsumexp(log_g - M, dim=2, keepdim=True)

        log_P = log_f + log_g - M  # (B, HW, N)

        # NaN guard fires at BOTH train and inference
        if torch.isnan(log_P).any():
            self.sinkhorn_divergence_count += 1
            warnings.warn(
                f"[BalancedEOT] Sinkhorn NaN detected "
                f"(eps={eps.item():.4f}, count={self.sinkhorn_divergence_count}). "
                "Falling back to softmax. Consider increasing ot_eps."
            )
            fallback_P = self._route_softmax(C_mat)
            if self.training:
                fallback_P = fallback_P + self.log_eps * 0.0
            return fallback_P

        # exp(-60) ≈ 1e-26 — negligible contribution to row sums
        return torch.exp(log_P.clamp(min=-60.0))

    # -----------------------------------------------------------------------
    # Shared Sinkhorn iteration runner
    # -----------------------------------------------------------------------

    def _sinkhorn_loop(
        self,
        log_f: torch.Tensor,
        log_g: torch.Tensor,
        M: torch.Tensor,
        log_a: float,
        log_b: float,
        col_fn,
        row_fn,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run warm-up (no-grad) + grad Sinkhorn iterations.

        col_fn(lse) and row_fn(lse) apply the marginal proximity operator to
        the respective logsumexp term.  For KL these are scalar multiplications;
        for TV they are symmetric clamps.
        """
        if self.training and self.n_nograd_iters > 0:
            with torch.no_grad():
                for _ in range(self.n_nograd_iters):
                    lse_g = torch.logsumexp(log_f - M, dim=1, keepdim=True)
                    log_g = log_b - col_fn(lse_g)
                    lse_f = torch.logsumexp(log_g - M, dim=2, keepdim=True)
                    log_f = log_a - row_fn(lse_f)

        n_iters = self.n_grad_iters if self.training else self.iters
        for _ in range(n_iters):
            lse_g = torch.logsumexp(log_f - M, dim=1, keepdim=True)
            log_g = log_b - col_fn(lse_g)
            lse_f = torch.logsumexp(log_g - M, dim=2, keepdim=True)
            log_f = log_a - row_fn(lse_f)

        return log_f, log_g

    # -----------------------------------------------------------------------
    # Unbalanced Sinkhorn with spatially-varying ρ (log-domain)
    # -----------------------------------------------------------------------

    def _route_unbalanced_eot(
        self, C_mat: torch.Tensor, rho_flat: torch.Tensor
    ) -> torch.Tensor:
        """
        Unbalanced entropic OT in log-domain.  Supports two marginal divergences
        controlled by `self.marginal_div`:

        'kl'  — KL proximity operator (Séjourné et al. 2019):
                    Aprox_KL(x) = ρ/(ρ+ε) · x
                    log_g ← log_b − shrink_col · lse(log_f − M, rows)
                    log_f ← log_a − shrink_row · lse(log_g − M, cols)
                Smooth spatial gating: row masses taper continuously ∈ (0,1).

        'tv'  — TV proximity operator (Séjourné et al. 2022):
                    Aprox_TV(x) = clamp(x, −ρ, +ρ)
                    log_g ← log_b − clamp(lse(log_f − M, rows), −ρ/ε, ρ/ε)
                    log_f ← log_a − clamp(lse(log_g − M, cols), −ρ/ε, ρ/ε)
                Sharp spatial gating: pixels either fully match the dictionary
                (row_sum ≈ 1) or are cleanly suppressed (row_sum ≈ 0).

        In the absorbed form the shrinkage / clamp applies ONLY to the logsumexp
        term, not to the marginal constants log_a / log_b.

        Parameters
        ----------
        C_mat    : (B, HW, N)  cost matrix float32, values in [0, 2]
        rho_flat : (B, HW)     per-pixel row marginal strength, > 0

        Returns
        -------
        P : (B, HW, N)  transport plan (NOT yet scaled by HW)
        """
        B, HW, N = C_mat.shape
        eps = F.softplus(self.log_eps) + 0.01  # strictly positive

        log_a = -math.log(HW)
        log_b = -math.log(N)
        M = C_mat / eps

        rho_mean = rho_flat.mean(dim=1, keepdim=True)  # (B, 1)
        log_f = C_mat.new_zeros(B, HW, 1)
        log_g = C_mat.new_zeros(B, 1, N)

        if self.marginal_div == "kl":
            # ρ→∞: shrink→1 (balanced limit)   ρ→0: shrink→0 (free transport)
            shrink_row = (rho_flat / (rho_flat + eps)).unsqueeze(2)  # (B, HW, 1)
            shrink_col = (rho_mean / (rho_mean + eps)).unsqueeze(2)  # (B, 1, 1)
            log_f, log_g = self._sinkhorn_loop(
                log_f,
                log_g,
                M,
                log_a,
                log_b,
                col_fn=lambda lse: shrink_col * lse,
                row_fn=lambda lse: shrink_row * lse,
            )
        else:  # tv — large ρ/ε → wide clamp (balanced), small ρ/ε → hard binary gating
            rpe_row = (rho_flat / eps).unsqueeze(2)  # (B, HW, 1)
            rpe_col = (rho_mean / eps).unsqueeze(2)  # (B, 1,  1)
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
                f"[UnbalancedEOT/{self.marginal_div.upper()}] Sinkhorn NaN detected "
                f"(eps={eps.item():.4f}, "
                f"rho_mean={rho_flat.mean().item():.4f}, "
                f"count={self.sinkhorn_divergence_count}). "
                "Falling back to softmax. Consider increasing ot_eps."
            )
            fallback_P = self._route_softmax(C_mat)
            if self.training:
                fallback_P = fallback_P + (rho_flat * 0.0).sum() + self.log_eps * 0.0
            return fallback_P

        return torch.exp(log_P.clamp(min=-60.0))

    # -----------------------------------------------------------------------
    # Softmax routing
    # -----------------------------------------------------------------------

    def _route_softmax(self, C_mat: torch.Tensor) -> torch.Tensor:
        """
        Temperature-scaled softmax routing.

        Used as the primary routing mode when routing_mode='softmax', and
        as a safe fallback when Sinkhorn diverges in either EOT mode.

        P[b, hw, :] = softmax(−C[b, hw, :] / τ)  — rows sum to 1.
        """
        tau = F.softplus(self.log_tau) + 0.01  # bounded away from 0
        return F.softmax(-C_mat / tau, dim=-1)

    # -----------------------------------------------------------------------
    # Dispersion loss  (returns −H in BITS)
    # -----------------------------------------------------------------------

    @staticmethod
    def _dispersion_loss(P: torch.Tensor) -> torch.Tensor:
        """
        Returns −H(m) where m is the column-marginal distribution of P.

            m_j = Σ_i P_ij / Z,    Z = Σ_{i,j} P_ij

        H(m) = −Σ_j m_j log₂ m_j   (Shannon entropy in bits)

        Entropy is in bits to be commensurate with bpp_loss units so that
        the adaptive EMA scaling in RateDistortionLoss remains well-conditioned.

        Returns negative entropy so callers can SUBTRACT it from the loss
        (minimising loss → maximising H → uniform dictionary usage).

        P : (B, HW, N) — already scaled by HW in the caller
        Returns: scalar tensor
        """
        marginal = P.sum(dim=1)  # (B, N)
        marginal = marginal / marginal.sum(dim=1, keepdim=True).clamp(min=1e-8)
        H = -(marginal * torch.log2(marginal.clamp(min=1e-8))).sum(dim=1).mean()
        return -H  # negative entropy in bits

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x           : (B, input_dim, H, W)   query feature map
        k           : (B, N, dict_dim)        dictionary keys  (from k_proj)
        v           : (B, N, dict_dim)        dictionary values (from v_proj)
        rho_spatial : (B, H, W) or None       per-pixel KL mass strength
                      Required for 'unbalanced_eot', ignored otherwise.
        calc_disp   : bool  — compute dispersion loss (True during training)

        Returns
        -------
        out      : (B, output_dim, H, W)  aggregated dictionary features
        disp_loss: scalar tensor          −H (bits), 0 if calc_disp=False
        """
        B, _, H, W = x.shape
        HW = H * W

        # Cost matrix always in float32 for numerical stability
        C_mat = self._cost_matrix(x, k, H, W).float()  # (B, HW, N)

        # ── Routing ──────────────────────────────────────────────────────────

        if self.routing_mode == "softmax":
            P = self._route_softmax(C_mat)
            # Row sums = 1 by softmax definition; no additional scaling needed.

        elif self.routing_mode == "balanced_eot":
            P = self._route_balanced_eot(C_mat)
            # Transport plan has uniform marginals 1/HW per row.
            # Scale by HW so row sums ≈ 1, matching the softmax convention
            # and keeping the aggregated feature magnitude stable.
            P = P * HW

        elif self.routing_mode == "unbalanced_eot":
            assert (
                rho_spatial is not None
            ), "rho_spatial must be provided for routing_mode='unbalanced_eot'"
            rho_flat = rho_spatial.view(B, HW).float().clamp(min=0.01)
            P = self._route_unbalanced_eot(C_mat, rho_flat)
            # Scale by HW as in the balanced case.
            # Row sums ≠ 1 in general — this is the spatial gating:
            #   row_sum_i ≈ shrink_i = ρ_i/(ρ_i+ε) · HW · (1/HW)
            #             = ρ_i/(ρ_i+ε)   ∈ (0, 1)
            # Pixels with no good dictionary match have low row_sum → attenuated
            # aggregated feature → effective soft-masking of irrelevant regions.
            P = P * HW

            # Store raw row mass (eval only) for ablation visualisation
            if not self.training:
                self.last_row_mass = P.sum(dim=-1).detach()  # (B, HW)
            else:
                self.last_row_mass = None

        else:
            raise RuntimeError(f"Unknown routing_mode: {self.routing_mode!r}")

        # Store full transport plan in eval mode for analysis.
        # Skipped during training to avoid O(B·HW·N·num_slices) memory overhead.
        if not self.training:
            self.attn_probs = P.detach()
        else:
            self.attn_probs = None

        # ── Spatial TV regularisation (optional, training only) ──────────────
        tv_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        if self.training and self.tv_weight > 0.0:
            tv_loss = self._spatial_tv(P, H, W) * self.tv_weight

        # ── Value aggregation ─────────────────────────────────────────────────
        # Normalise values and scale by sqrt(D) so the expected output magnitude
        # is O(1) when row sums of P are O(1) — standard attention normalisation.
        v_norm = F.normalize(v, p=2, dim=-1) * self._sqrt_dict_dim  # (B, N, D)
        # Weighted sum: each spatial position gets a convex (or near-convex)
        # combination of dictionary value vectors.
        # P : (B, HW, N)   v_norm : (B, N, D)  →  out : (B, HW, D)
        out = (
            torch.bmm(P, v_norm)  # (B, HW, D)
            .transpose(1, 2)  # (B, D, HW)
            .contiguous()
            .view(B, -1, H, W)  # (B, D, H, W)
        )

        # ── Dispersion loss ───────────────────────────────────────────────────
        if calc_disp:
            disp_loss = self._dispersion_loss(P) + tv_loss
            if self.training:
                # Keep unused parameter in the backward graph for DDP correctness
                # (find_unused_parameters=False requires all params to appear in loss).
                if self.routing_mode != "softmax":
                    disp_loss = disp_loss + self.log_tau * 0.0
                else:
                    disp_loss = disp_loss + self.log_eps * 0.0
        else:
            disp_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return self.out_proj(out), disp_loss
