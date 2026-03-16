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
        context = context.view(B, C, H * W).transpose(1, 2)
        queries = self.dict_queries.expand(B, -1, -1)

        attn_out, _ = self.cross_attn(query=queries, key=context, value=context)
        out = self.norm(queries + attn_out)
        return self.proj(out)


class SpatialDispersionEOTAttention(nn.Module):
    def __init__(
        self, input_dim, output_dim, dict_num=128, dict_dim=640, tau=0.5, iters=3
    ):
        super().__init__()
        self.dict_num = dict_num
        self.tau = tau
        self.iters = iters

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)

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

        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )

    def forward(self, x, k, v, spatial_epsilon):
        """
        Accepts precomputed `k` and `v` to save redundant linear projections during slice iteration.
        """
        B, C, H, W = x.shape
        q = self.q_proj(x).view(B, -1, H * W).transpose(1, 2)  # (B, HW, D)

        q_norm = F.normalize(q, p=2, dim=-1)
        k_norm = F.normalize(k, p=2, dim=-1)
        C_mat = 1.0 - torch.bmm(q_norm, k_norm.transpose(1, 2))  # (B, HW, N)

        C_spatial = C_mat.transpose(1, 2).view(B, self.dict_num, H, W)
        C_spatial = self.spatial_smooth(C_spatial)
        C_mat = C_spatial.view(B, self.dict_num, H * W).transpose(1, 2)

        eps_spatial = spatial_epsilon.view(B, H * W, 1).clamp(min=1e-4)
        eps_mean = spatial_epsilon.mean(dim=(2, 3)).clamp(min=1e-4)

        u = torch.zeros_like(C_mat[:, :, 0])
        v_vec = torch.zeros_like(C_mat[:, 0, :])

        target_marginal = math.log(1.0 / self.dict_num)
        hw_log = math.log(H * W)
        tau_ratio = self.tau / (self.tau + eps_mean)

        for _ in range(self.iters):
            u = eps_spatial.squeeze(-1) * (
                -torch.logsumexp((-C_mat + v_vec.unsqueeze(1)) / eps_spatial, dim=2)
            )
            v_unbalanced = target_marginal * eps_mean + eps_mean * (
                -(
                    torch.logsumexp((-C_mat + u.unsqueeze(2)) / eps_spatial, dim=1)
                    - hw_log
                )
            )
            v_vec = tau_ratio * v_unbalanced

        # =====================================================================
        # Sinkhorn Marginal Normalization and Safe Exp
        # =====================================================================
        logits = (-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / eps_spatial
        # Clamp guards against float16/float32 explosion
        P = torch.exp(logits.clamp(max=0.0))
        # Force strict sum-to-1 along the dictionary dimension to prevent unscaled features
        P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
        self.attn_probs = P.detach()

        out_bmm = torch.bmm(P, v)  # (B, HW, D)
        out = out_bmm.transpose(1, 2).view(B, -1, H, W)

        if self.training:
            # Detach P here to avoid sending gradients through the unstable Sinkhorn loop
            v_expanded = v.unsqueeze(1)
            out_expanded = out_bmm.detach().unsqueeze(2)
            dist_sq = torch.sum((v_expanded - out_expanded) ** 2, dim=-1)
            dispersion_loss = torch.mean(torch.sum(P.detach() * dist_sq, dim=2))
        else:
            dispersion_loss = torch.tensor(0.0, device=x.device)

        return self.out_proj(out), dispersion_loss
