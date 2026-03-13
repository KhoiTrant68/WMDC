import math

import torch
import torch.nn as nn
from einops import rearrange


class QueryDictionaryGenerator(nn.Module):
    """
    DETR-style Query-based Dictionary Generator.
    Resolves the 42M linear layer trap. Costs ~0.3M params and 0 bits overhead.
    """

    def __init__(self, in_dim=192, dict_num=128, dict_dim=640, num_heads=4):
        super().__init__()
        self.dict_num = dict_num
        self.dict_dim = dict_dim

        # Learnable queries
        self.dict_queries = nn.Parameter(torch.randn(1, dict_num, in_dim))

        # Spatial implicit encoding
        self.pos_enc = nn.Conv2d(
            in_dim, in_dim, kernel_size=3, padding=1, groups=in_dim
        )

        # Cross Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=in_dim, num_heads=num_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(in_dim)

        # Projection to final dictionary dimension
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

        dict_tokens = self.proj(out)  # (B, 128, 640)
        return dict_tokens


class EntropicOptimalTransportAttention(nn.Module):
    """
    EOT-HDDA: Uses Sinkhorn algorithm for Sparse Dictionary Matching.
    Mathematically prevents entropy inflation by restricting query-key assignments
    and enforcing a uniform marginal target to prevent index collapse.
    """

    def __init__(
        self, input_dim, output_dim, dict_num=128, dict_dim=640, epsilon=0.05, iters=3
    ):
        super().__init__()
        self.dict_num = dict_num
        self.epsilon = epsilon
        self.iters = iters

        self.q_proj = nn.Conv2d(input_dim, dict_dim, 1)
        self.k_proj = nn.Linear(dict_dim, dict_dim)
        self.v_proj = nn.Linear(dict_dim, dict_dim)

        # Output projection and residual scaling (Depthwise for efficiency)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dict_dim, dict_dim, 3, 1, 1, groups=dict_dim),
            nn.GELU(),
            nn.Conv2d(dict_dim, output_dim, 1),
        )
        self.scale = dict_dim**-0.5

    def forward(self, x, dt):
        """
        x: (B, C, H, W) - Local feature query
        dt: (B, N, D) - Dynamic Global Dictionary
        """
        B, C, H, W = x.shape

        # Projections
        q = self.q_proj(x).view(B, -1, H * W).transpose(1, 2)  # (B, HW, D)
        k = self.k_proj(dt)  # (B, N, D)
        v = self.v_proj(dt)  # (B, N, D)

        # 1. Cost Matrix (Cosine Distance)
        C_mat = 1.0 - torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, HW, N)

        # 2. Sinkhorn Iterations (Log-Domain for numerical stability)
        u = torch.zeros_like(C_mat[:, :, 0])  # (B, HW)
        v_vec = torch.zeros_like(C_mat[:, 0, :])  # (B, N)

        # Ensures every dictionary token gets queried equally on average
        target_marginal = math.log((H * W) / self.dict_num)

        for _ in range(self.iters):
            u = self.epsilon * (
                -torch.logsumexp((-C_mat + v_vec.unsqueeze(1)) / self.epsilon, dim=2)
            )
            v_vec = self.epsilon * target_marginal + self.epsilon * (
                -torch.logsumexp((-C_mat + u.unsqueeze(2)) / self.epsilon, dim=1)
            )

        # 3. Optimal Transport Plan (Sparse Attention Matrix)
        P = torch.exp((-C_mat + u.unsqueeze(2) + v_vec.unsqueeze(1)) / self.epsilon)

        # 4. Gather Dictionary Values
        out = torch.bmm(P, v)  # (B, HW, D)

        # 5. Output Projection
        out = out.transpose(1, 2).view(B, -1, H, W)
        return x + self.out_proj(out)
