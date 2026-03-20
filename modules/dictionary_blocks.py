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
    Unified Dictionary Attention supporting three routing modes:
      'softmax'        — standard softmax baseline
      'balanced_eot'   — strict balanced optimal transport (Sinkhorn)
      'unbalanced_eot' — KL-regularised unbalanced OT (WFR / Séjourné et al. 2023)

    Sinkhorn convention (Cuturi / Peyré log-domain formulation):
      Rows  = HW spatial query positions   (source measure α, uniform 1/HW)
      Cols  = N  dictionary tokens         (target measure β, uniform 1/N )

      Row update (u / f):  f_i = ε·log(a_i) - ε·LSE_j[(g_j - C_ij)/ε]
                                = -ε·log(HW) - ε·LSE_cols((-C + v)/ε)
      Col update (v / g):  g_j = ε·log(b_j) - ε·LSE_i[(f_i - C_ij)/ε]
                                = -ε·log(N)  - ε·LSE_rows((-C + u)/ε)

      For unbalanced OT: both potentials are additionally shrunk by τ/(τ+ε)
      (KL Aprox from Table 1, Séjourné et al. 2023).
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

        # Frozen depthwise conv for spatial smoothing of the cost matrix
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

        # GroupNorm stabilises magnitudes from the unnormalised WFR routing plan
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
        # 1. Project and L2-normalise queries and keys
        # ---------------------------------------------------------------
        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2).contiguous()  # (B, HW, D)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        k_norm_t = k_norm.transpose(1, 2).contiguous()  # (B, D, N)

        # Cost matrix C = 1 − cosine_similarity, shape (B, HW, N)
        C_mat = 1.0 - torch.bmm(q_norm, k_norm_t)

        # Apply spatial smoothing in (B, N, H, W) space then flatten back
        C_spatial = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_spatial = self.spatial_smooth(C_spatial)
        C_mat = C_spatial.view(B, self.dict_num, HW).transpose(1, 2).contiguous()

        # ---------------------------------------------------------------
        # 2. Routing
        # ---------------------------------------------------------------

        # Log-masses for source (row) and target (col) measures.
        # Source: HW spatial positions, uniform mass 1/HW → log(a_i) = -log(HW)
        # Target: N  dictionary tokens, uniform mass 1/N  → log(b_j) = -log(N)
        log_a = -math.log(HW)  # scalar
        log_b = -math.log(self.dict_num)  # scalar

        if self.routing_mode == "softmax":
            # BASELINE: standard softmax (spatial_epsilon not used here)
            dummy_eps = 0.0 * spatial_epsilon.sum()
            logits = -C_mat / self.tau + dummy_eps
            P = F.softmax(logits, dim=-1)

        elif self.routing_mode == "balanced_eot":
            # STRICT SINKHORN — balanced OT.
            # Log-domain updates (Algorithm 1):
            #   f_i ← ε·log(a_i) - ε·LSE_j[(g_j - C_ij)/ε]
            #        = -ε·log(HW) - ε·LSE_cols((-C + v)/ε)
            #   g_j ← ε·log(b_j) - ε·LSE_i[(f_i - C_ij)/ε]
            #        = -ε·log(N)  - ε·LSE_rows((-C + u)/ε)
            eps = spatial_epsilon.mean(dim=(2, 3)).view(B, 1).clamp(min=1e-3)  # (B, 1)

            u = torch.zeros(B, HW, device=x.device, dtype=x.dtype)
            v_vec = torch.zeros(B, self.dict_num, device=x.device, dtype=x.dtype)

            for _ in range(self.iters):
                # Row update: uses log(a_i) = log_a = -log(HW)
                u = log_a * eps - eps * torch.logsumexp(
                    (-C_mat + v_vec.unsqueeze(1)) / eps.unsqueeze(2),
                    dim=2,
                )  # (B, HW)

                # Col update: uses log(b_j) = log_b = -log(N)
                v_vec = log_b * eps - eps * torch.logsumexp(
                    (-C_mat + u.unsqueeze(2)) / eps.unsqueeze(2),
                    dim=1,
                )  # (B, N)

            logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps.unsqueeze(2)
            P = torch.exp(torch.clamp(logits, max=20.0))

        elif self.routing_mode == "unbalanced_eot":
            # KL-REGULARISED UNBALANCED OT (Séjourné et al. 2023, Table 1).
            # KL Aprox: Aprox_ε(p) = τ/(τ+ε)·p  applied to BOTH potentials.
            #
            # Updates:
            #   raw_u = ε·log(a_i) - ε·LSE_j[(v_j - C_ij)/ε]
            #   u     = τ/(τ+ε) · raw_u
            #   raw_v = ε·log(b_j) - ε·LSE_i[(u_i - C_ij)/ε]
            #   v     = τ/(τ+ε) · raw_v
            eps = spatial_epsilon.mean(dim=(2, 3)).view(B, 1).clamp(min=1e-3)  # (B, 1)

            tau_ratio = self.tau / (self.tau + eps)  # (B, 1)

            u = torch.zeros(B, HW, device=x.device, dtype=x.dtype)
            v_vec = torch.zeros(B, self.dict_num, device=x.device, dtype=x.dtype)

            for _ in range(self.iters):
                # Row update — KL Aprox on source side
                raw_u = log_a * eps - eps * torch.logsumexp(
                    (-C_mat + v_vec.unsqueeze(1)) / eps.unsqueeze(2),
                    dim=2,
                )  # (B, HW)
                u = tau_ratio * raw_u

                # Col update — KL Aprox on target side
                raw_v = log_b * eps - eps * torch.logsumexp(
                    (-C_mat + u.unsqueeze(2)) / eps.unsqueeze(2),
                    dim=1,
                )  # (B, N)
                v_vec = tau_ratio * raw_v

            logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps.unsqueeze(2)
            # clamp(max=20) prevents exp overflow while allowing plan entries > 1
            # (correct for unbalanced OT where total mass need not equal 1)
            P = torch.exp(torch.clamp(logits, max=20.0))

        # Store for visualisation / analysis hooks
        self.attn_probs = P

        # ---------------------------------------------------------------
        # 3. Aggregate values
        # ---------------------------------------------------------------
        out_bmm = torch.bmm(P, v)  # (B, HW, D)
        out = out_bmm.transpose(1, 2).contiguous().view(B, -1, H, W)

        # ---------------------------------------------------------------
        # 4. Dispersion loss (training, Slice 0 only)
        # ---------------------------------------------------------------
        if self.training and calc_disp:
            # Use torch.cdist to avoid allocating the (B, HW, N, D) intermediate
            # that the naive (q_expanded - k_expanded)**2 broadcast would create.
            # For B=8, HW=256, N=128, D=640 that intermediate is ~2.1 GB.
            dist_sq = torch.cdist(q, k, p=2).pow(2)  # (B, HW, N)
            dispersion_loss = torch.mean(torch.sum(P.detach() * dist_sq, dim=2))
        else:
            dispersion_loss = torch.tensor(0.0, device=x.device)

        return self.out_proj(out), dispersion_loss
