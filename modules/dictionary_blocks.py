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

        # Spatial smoothing for the cost matrix
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

        #  GroupNorm to stabilize magnitudes from the unnormalized WFR routing plan
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GroupNorm(1, dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

    def forward(self, x, k, v, spatial_epsilon, calc_disp=False):
        B, C, H, W = x.shape
        HW = H * W

        # 1. Project and Normalize Queries and Keys
        q = self.q_proj(x).view(B, -1, HW).transpose(1, 2).contiguous()  # (B, HW, D)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        k_norm_t = k_norm.transpose(1, 2).contiguous()

        # Cost Matrix C = 1 - Cosine Similarity
        C_mat = 1.0 - torch.bmm(q_norm, k_norm_t)  # (B, HW, N)

        # Apply Spatial Smoothing to Cost Matrix
        C_spatial = C_mat.transpose(1, 2).contiguous().view(B, self.dict_num, H, W)
        C_spatial = self.spatial_smooth(C_spatial)
        C_mat = C_spatial.view(B, self.dict_num, HW).transpose(1, 2).contiguous()

        # ---------------------------------------------------------------------
        # ROUTING MECHANISM ABLATION
        # ---------------------------------------------------------------------
        if self.routing_mode == "softmax":
            # BASELINE: Standard Softmax (Unconstrained)
            # P = Softmax(-C / tau)
            dummy_eps = 0.0 * spatial_epsilon.sum()
            logits = -C_mat / self.tau + dummy_eps
            P = F.softmax(logits, dim=-1)

        elif self.routing_mode == "balanced_eot":
            # STRICT SINKHORN: Balanced Optimal Transport
            # Forces exact marginals: Rows sum to 1, Columns sum to HW/N
            eps = spatial_epsilon.mean(dim=(2, 3)).view(B, 1).clamp(min=1e-3)  # (B, 1)

            u = torch.zeros_like(C_mat[:, :, 0])  # (B, HW)
            v_vec = torch.zeros_like(C_mat[:, 0, :])  # (B, N)

            # For uniform column usage, each of the N columns must receive exactly HW/N mass.
            log_b_target = math.log(HW / self.dict_num)

            for _ in range(self.iters):
                # Update u (Row constraint: sum to 1 -> log(1) = 0)
                u = -eps * torch.logsumexp(
                    (-C_mat + v_vec.unsqueeze(1)) / eps.unsqueeze(2), dim=2
                )
                # Update v (Column constraint: sum to HW/N -> log(HW/N))
                v_vec = eps * log_b_target - eps * torch.logsumexp(
                    (-C_mat + u.unsqueeze(2)) / eps.unsqueeze(2), dim=1
                )

            logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps.unsqueeze(2)
            P = torch.exp(logits)  # Exact marginals achieved

        elif self.routing_mode == "unbalanced_eot":
            # OURS: Unbalanced EOT (Wasserstein-Fisher-Rao)
            # Relaxes column constraints to allow adaptive splitting
            eps = spatial_epsilon.mean(dim=(2, 3)).view(B, 1).clamp(min=1e-3)

            u = torch.zeros_like(C_mat[:, :, 0])
            v_vec = torch.zeros_like(C_mat[:, 0, :])

            target_marginal = math.log(1.0 / self.dict_num)
            hw_log = math.log(HW)
            tau_ratio = self.tau / (self.tau + eps)  # (B, 1)

            for _ in range(self.iters):
                u = eps * (
                    -torch.logsumexp(
                        (-C_mat + v_vec.unsqueeze(1)) / eps.unsqueeze(2), dim=2
                    )
                )

                v_unbalanced = target_marginal * eps + eps * (
                    -(
                        torch.logsumexp(
                            (-C_mat + u.unsqueeze(2)) / eps.unsqueeze(2), dim=1
                        )
                        - hw_log
                    )
                )
                v_vec = tau_ratio * v_unbalanced

            # =====================================================================
            # Sinkhorn Marginal Normalization and Safe Exp
            # =====================================================================
            # Use eps_global consistently for KKT optimality.
            logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps.unsqueeze(2)

            # We clamp to prevent blowups, allowing row sums to be < 1 or > 1.
            P = torch.exp(torch.clamp(logits, max=0.0))

        # Save for visualization/analysis hook
        self.attn_probs = P

        # Apply Attention to Values
        out_bmm = torch.bmm(P, v)  # (B, HW, D)
        out = out_bmm.transpose(1, 2).contiguous().view(B, -1, H, W)

        # ---------------------------------------------------------------------
        # DISPERSION LOSS (Only active during training)
        # ---------------------------------------------------------------------
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
