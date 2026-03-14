import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryDictionaryGenerator(nn.Module):
    """DETR-style Query-based Dictionary Generator."""

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
        context = context.view(B, C, H * W).transpose(1, 2)  # (B, H*W, C)
        queries = self.dict_queries.expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)

        return self.proj(out)  # (B, 128, 640)


class SpatialDispersionEOTAttention(nn.Module):
    """
    Integrates Spatial Cost Regularization (prevents spatial bitrate leakage)
    and Dispersion Penalization (prevents 'drab'/blurry latents).
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dict_num=128,
        dict_dim=640,
        epsilon=0.05,
        tau=0.5,
        iters=3,
    ):
        super().__init__()
        self.dict_num = dict_num
        self.epsilon = epsilon
        self.tau = tau
        self.iters = iters

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)
        self.k_proj = nn.Linear(dict_dim, dict_dim)
        self.v_proj = nn.Linear(dict_dim, dict_dim)

        # Enforces local spatial coherence in dictionary assignments, slashing BPP
        self.spatial_smooth = nn.Conv2d(
            dict_num, dict_num, kernel_size=3, padding=1, groups=dict_num, bias=False
        )
        self.spatial_smooth.weight.data.fill_(1.0 / 9.0)
        self.spatial_smooth.weight.requires_grad = False

        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

    def forward(self, x, dt):
        B, C, H, W = x.shape

        q = self.q_proj(x).view(B, -1, H * W).transpose(1, 2)  # (B, HW, D)
        k = self.k_proj(dt)  # (B, N, D)
        v = self.v_proj(dt)  # (B, N, D)

        # 1. Cost Matrix (L2 Normalized)
        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, HW, N)

        # Apply Spatial Graph Regularization
        C_spatial = C_mat.transpose(1, 2).view(B, self.dict_num, H, W)
        C_spatial = self.spatial_smooth(C_spatial)
        C_mat = C_spatial.view(B, self.dict_num, H * W).transpose(1, 2)

        # 2. Semi-Unbalanced Sinkhorn Iterations
        u = torch.zeros_like(C_mat[:, :, 0])  # (B, HW)
        v_vec = torch.zeros_like(C_mat[:, 0, :])  # (B, N)

        # RESOLUTION INVARIANT: Target marginal must be an intensive property
        target_marginal = math.log(1.0 / self.dict_num)
        tau_ratio = self.tau / (self.tau + self.epsilon)

        for _ in range(self.iters):
            # Query
            u = self.epsilon * (
                -torch.logsumexp((-C_mat + v_vec.unsqueeze(1)) / self.epsilon, dim=2)
            )

            # Key
            v_unbalanced = self.epsilon * target_marginal + self.epsilon * (
                -torch.logsumexp((-C_mat + u.unsqueeze(2)) / self.epsilon, dim=1)
            )
            v_vec = tau_ratio * v_unbalanced

        # 3. Optimal Transport Plan
        P = torch.exp((-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / self.epsilon)

        self.attn_probs = P.detach()

        # 4. Gather Dictionary Values & Project
        out_bmm = torch.bmm(P, v)  # (B, HW, D)
        out = out_bmm.transpose(1, 2).view(B, -1, H, W)

        # 5. DISPERSION PENALIZATION (Calculated during training only)
        # Prevents "drab" mixing by penalizing intra-cluster variance
        if self.training:
            out_detached = out_bmm.detach()
            v_expanded = v.unsqueeze(1)  # (B, 1, N, D)
            out_expanded = out_detached.unsqueeze(2)  # (B, HW, 1, D)

            dist_sq = torch.sum((v_expanded - out_expanded) ** 2, dim=-1)  # (B, HW, N)
            dispersion_loss = torch.mean(torch.sum(P * dist_sq, dim=2))
        else:
            dispersion_loss = torch.tensor(0.0, device=x.device)

        return self.out_proj(out), dispersion_loss
