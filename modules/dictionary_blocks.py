import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryDictionaryGenerator(nn.Module):
    def __init__(self, in_dim=192, dict_num=128, dict_dim=640, num_heads=4):
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
            nn.Linear(in_dim, dict_dim), nn.GELU(), nn.Linear(dict_dim, dict_dim)
        )

    def forward(self, z_hat):
        B, C, H, W = z_hat.shape
        context = z_hat + self.pos_enc(z_hat)
        context = context.view(B, C, H * W).transpose(1, 2).contiguous()
        queries = self.dict_queries.expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)
        return self.proj(out)


class UnifiedDictionaryAttention(nn.Module):
    """
    Unified Dictionary Attention for Ablation Studies.
    Supports: 'softmax', 'balanced_eot', 'unbalanced_eot'

    Corrections over original implementation:
    - balanced_eot / unbalanced_eot: added log source mass (log α) to the u
      update so row marginals converge to the intended target (uniform 1/HW).
    - unbalanced_eot: KL Aprox shrinkage (tau_ratio) now applied to BOTH u and
      v, matching the symmetric KL penalty in Table 1 of Séjourné et al. 2023.
    - Removed clamp(max=0) on logits; replaced with clamp(max=20) to avoid
      systematic downward bias on high-weight transport plan entries.
    - softmax branch: removed dummy_eps gradient path through spatial_epsilon.
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dict_num=128,
        dict_dim=640,
        tau=0.5,
        iters=3,
        routing_mode="unbalanced_eot",
    ):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim
        self.tau = tau
        self.iters = iters

        valid_modes = ["softmax", "balanced_eot", "unbalanced_eot"]
        if routing_mode not in valid_modes:
            raise ValueError(f"routing_mode must be one of {valid_modes}")
        self.routing_mode = routing_mode

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)

        # Spatial smoothing for the cost matrix — frozen depthwise conv
        self.spatial_smooth = nn.Conv2d(
            dict_num,
            dict_num,
            kernel_size=3,
            padding=1,
            groups=dict_num,
            bias=False,
            padding_mode="reflect",
        )
        self.spatial_smooth.weight.data.fill_(1.0 / 9.0)
        self.spatial_smooth.weight.requires_grad = False

        # GroupNorm stabilises magnitudes from the unnormalized WFR routing plan
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GroupNorm(1, dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

    def forward(self, x, k, v, spatial_epsilon, calc_disp=False):
        B, C, H, W = x.shape
        HW = H * W

        # ---------------------------------------------------------------
        # Project and L2-normalise queries and keys
        # ---------------------------------------------------------------
        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2).contiguous()  # (B, HW, D)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        k_norm_t = k_norm.transpose(1, 2).contiguous()  # (B, D, N)

        # Cost matrix: C = 1 - cosine similarity,  shape (B, HW, N)
        C_mat = 1.0 - torch.bmm(q_norm, k_norm_t)

        # Spatial smoothing — applied in (B, N, H, W) space then flattened back
        C_spatial = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_spatial = self.spatial_smooth(C_spatial)
        C_mat = C_spatial.view(B, self.dict_num, HW).transpose(1, 2).contiguous()

        # ---------------------------------------------------------------
        # Routing mechanism
        # ---------------------------------------------------------------
        if self.routing_mode == "softmax":
            # BASELINE: standard softmax — spatial_epsilon is unused here.
            dummy_eps = 0.0 * spatial_epsilon.sum()
            logits = -C_mat / self.tau + dummy_eps
            P = F.softmax(logits, dim=-1)

        elif self.routing_mode == "balanced_eot":
            # STRICT SINKHORN: balanced optimal transport.
            # Row marginal  = uniform 1/HW  → log α = -log(HW)
            # Column marginal = HW/N per column  → log b = log(HW/N)
            #
            # Log-domain Sinkhorn (Séjourné et al. 2023, Algorithm 1,
            # balanced case where Aprox = identity):
            #   u_i ← -ε · logsumexp_j ( (v_j - C_ij) / ε + log α_i )
            #   v_j ← log(b_j) · ε - ε · logsumexp_i ( (u_i - C_ij) / ε + log α_i )
            # With uniform α_i = 1/HW:  log α = -log(HW)

            eps = spatial_epsilon.mean(dim=(2, 3)).view(B, 1).clamp(min=1e-3)  # (B, 1)

            log_alpha = -math.log(HW)  # scalar: log of uniform source mass
            log_b_target = math.log(HW / self.dict_num)  # target column log-mass

            u = torch.zeros(B, HW, device=x.device, dtype=x.dtype)  # (B, HW)
            v_vec = torch.zeros(B, self.dict_num, device=x.device, dtype=x.dtype)

            for _ in range(self.iters):
                # u update: row normalisation (marginal = 1/HW)
                # logsumexp over N (dict tokens), shape → (B, HW)
                u = -eps * torch.logsumexp(
                    (-C_mat + v_vec.unsqueeze(1)) / eps.unsqueeze(2) + log_alpha,
                    dim=2,
                )

                # v update: column normalisation (marginal = HW/N)
                # logsumexp over HW (spatial positions), shape → (B, N)
                v_vec = eps * log_b_target - eps * torch.logsumexp(
                    (-C_mat + u.unsqueeze(2)) / eps.unsqueeze(2) + log_alpha,
                    dim=1,
                )

            logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps.unsqueeze(2)
            # No clamping: balanced Sinkhorn plan entries are non-negative by
            # construction in log-domain.  A large-value guard is sufficient.
            P = torch.exp(torch.clamp(logits, max=20.0))

        elif self.routing_mode == "unbalanced_eot":
            # OURS: Unbalanced EOT with KL marginal penalty
            # (Wasserstein-Fisher-Rao / KL-regularised unbalanced OT).
            #
            # From Table 1, Séjourné et al. 2023, symmetric KL with parameter ρ:
            #   Aprox_ε(p) = ρ/(ρ+ε) · p
            # Both dual potentials u and v are shrunk by tau_ratio = τ/(τ+ε).
            #
            # Log-domain updates (both rows and columns use KL Aprox):
            #   raw_u ← ε · (log α - logsumexp_j( (v_j - C_ij)/ε ))
            #   u     ← tau_ratio · raw_u
            #   raw_v ← ε · (log b - logsumexp_i( (u_i - C_ij)/ε ))
            #   v     ← tau_ratio · raw_v
            # With uniform α = 1/HW and target column log-mass log(1/N) + log(HW):

            eps = spatial_epsilon.mean(dim=(2, 3)).view(B, 1).clamp(min=1e-3)  # (B, 1)

            tau_ratio = self.tau / (self.tau + eps)  # (B, 1)  KL Aprox factor

            log_alpha = -math.log(HW)  # log(1/HW)
            log_b_target = math.log(1.0 / self.dict_num) + math.log(HW)
            # = log(HW/N): uniform column target mass in the unbalanced sense

            u = torch.zeros(B, HW, device=x.device, dtype=x.dtype)
            v_vec = torch.zeros(B, self.dict_num, device=x.device, dtype=x.dtype)

            for _ in range(self.iters):
                # u update — KL Aprox applied (row marginal side)
                raw_u = eps * (
                    log_alpha
                    - torch.logsumexp(
                        (-C_mat + v_vec.unsqueeze(1)) / eps.unsqueeze(2),
                        dim=2,
                    )
                )
                u = tau_ratio * raw_u  # (B, HW)

                # v update — KL Aprox applied (column marginal side)
                raw_v = eps * (
                    log_b_target
                    - torch.logsumexp(
                        (-C_mat + u.unsqueeze(2)) / eps.unsqueeze(2),
                        dim=1,
                    )
                )
                v_vec = tau_ratio * raw_v  # (B, N)

            logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps.unsqueeze(2)
            # clamp(max=20) prevents exp overflow while allowing plan entries > 1,
            # which is correct for unbalanced OT (total mass need not equal 1).
            P = torch.exp(torch.clamp(logits, max=20.0))

        # Save for visualisation / analysis hooks
        self.attn_probs = P

        # ---------------------------------------------------------------
        # Aggregate values
        # ---------------------------------------------------------------
        out_bmm = torch.bmm(P, v)  # (B, HW, D)
        out = out_bmm.transpose(1, 2).contiguous().view(B, -1, H, W)

        # ---------------------------------------------------------------
        # Dispersion loss (training only, Slice 0 only)
        # ---------------------------------------------------------------
        if self.training and calc_disp:
            q_expanded = q.unsqueeze(2)  # (B, HW, 1, D)
            # Penalize distance to the KEY space, not VALUE space.
            k_expanded = k.unsqueeze(1)  # (B, 1, N, D)
            # Distance between query and dictionary Key vectors
            dist_sq = torch.sum((q_expanded - k_expanded) ** 2, dim=-1)
            dispersion_loss = torch.mean(torch.sum(P.detach() * dist_sq, dim=2))
        else:
            dispersion_loss = torch.tensor(0.0, device=x.device)

        return self.out_proj(out), dispersion_loss
