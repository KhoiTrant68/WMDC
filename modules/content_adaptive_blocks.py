"""
content_adaptive_blocks.py
==========================
Content-adaptive Mamba token permutation using K-means clustering.

Key Idea (from CMIC):
  Instead of processing tokens in raster (spatial) order through the selective scan,
  cluster tokens semantically and reorder them so the scan operates on similar
  features consecutively. Expected gain: +0.5–1.0 dB on Kodak.

Classes:
  TokenClustering — K-means with EMA-updated centroids, lazy bootstrap
  ContentAdaptiveVSSBlock — VSSBlock subclass wrapping the permutation logic
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock


class TokenClustering(nn.Module):
    """
    K-means clustering for 2D feature map tokens.

    Maintains EMA-updated centroids initialized from the first batch.
    On small feature maps (HW < 8*N), falls back to no clustering.

    Attributes:
      cluster_num (int): Number of clusters (dictionary size)
      feature_dim (int): Feature dimension per token
      momentum (float): EMA momentum for centroid updates
      initted (bool buffer): Whether centroids have been initialized
      means (Tensor buffer): Centroid vectors, shape (N,)
    """

    def __init__(
        self, cluster_num: int = 8, feature_dim: int = None, momentum: float = 0.99
    ):
        super().__init__()
        self.cluster_num = cluster_num
        self.feature_dim = feature_dim
        self.momentum = momentum

        # Register buffers (not in state_dict until first forward)
        self.register_buffer("initted", torch.tensor(False, dtype=torch.bool))
        if feature_dim is not None:
            self.register_buffer("means", torch.zeros(cluster_num, feature_dim))
        else:
            self.means = None

    def _bootstrap_from_batch(self, x_flat: torch.Tensor) -> None:
        """Initialize centroids from batch using random selection."""
        # x_flat: (B*HW, D)
        B_HW = x_flat.shape[0]
        idx = torch.randperm(B_HW, device=x_flat.device)[: self.cluster_num]
        self.means.data = x_flat[idx].clone().detach()
        self.initted.data = torch.tensor(True, dtype=torch.bool)

    def _center_iter(self, x_flat: torch.Tensor, assign: torch.Tensor) -> None:
        """Single K-means iteration: update centroids via hard assignment."""
        # assign: (B*HW,) integer cluster IDs
        for k in range(self.cluster_num):
            mask = assign == k
            if mask.sum() > 0:
                new_mean = x_flat[mask].mean(dim=0)
                # EMA update
                self.means.data[k] = (
                    self.momentum * self.means.data[k] + (1 - self.momentum) * new_mean
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Assign tokens to clusters.

        Parameters
        ----------
        x : (B, HW, D) feature map flattened to tokens

        Returns
        -------
        assign : (B, HW) cluster assignments (integers 0..N-1)
        """
        B, HW, D = x.shape

        # Fallback: if too few tokens, return identity
        if HW < 8 * self.cluster_num:
            return torch.arange(HW, device=x.device).unsqueeze(0).expand(B, -1)

        # Initialize on first forward
        if not self.initted:
            x_flat = x.reshape(-1, D)
            self._bootstrap_from_batch(x_flat)

        # Hard assignment: argmin distance to nearest centroid
        x_flat = x.reshape(B * HW, D)
        dists = torch.cdist(x_flat, self.means)  # (B*HW, N)
        assign_flat = dists.argmin(dim=1)  # (B*HW,)

        # Update centroids via EMA
        if self.training:
            self._center_iter(x_flat, assign_flat)

        return assign_flat.view(B, HW)


class ContentAdaptiveVSSBlock(VSSBlock):
    """
    VSSBlock subclass that reorders tokens via K-means clustering before SS2D.

    Preserves all parent weights at the same state_dict keys; cluster buffers
    are initialized lazily on first forward.

    Attributes:
      cluster_num (int): Number of clusters
      token_clustering (TokenClustering): Clustering module
    """

    def __init__(
        self,
        hidden_dim: int = 0,
        drop_path: float = 0,
        cluster_num: int = 8,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            drop_path=drop_path,
            norm_layer=norm_layer,
            **kwargs,
        )
        self.cluster_num = cluster_num
        self.token_clustering = TokenClustering(
            cluster_num=cluster_num, feature_dim=hidden_dim, momentum=0.99
        )

    def _gather_tokens(
        self, x: torch.Tensor, assign: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder tokens by cluster assignment, returning permuted and unsort indices.

        Parameters
        ----------
        x : (B, HW, D) tokens
        assign : (B, HW) cluster assignments

        Returns
        -------
        x_sorted : (B, HW, D) tokens sorted by cluster ID
        unsort_idx : (B, HW) indices to undo permutation
        """
        B, HW, D = x.shape
        x_sorted = x.clone()

        for b in range(B):
            # Stable sort by cluster ID within this batch element
            sort_idx = torch.argsort(assign[b], stable=True)
            unsort_idx = torch.argsort(sort_idx)
            x_sorted[b] = x[b, sort_idx]
            if b == 0:
                unsort_indices = unsort_idx.unsqueeze(0)
            else:
                unsort_indices = torch.cat([unsort_indices, unsort_idx.unsqueeze(0)])

        return x_sorted, unsort_indices

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with content-adaptive token reordering.

        Pipeline:
          1. Permute (0,2,3,1) to (B, H, W, D) → (B, HW, D)
          2. Cluster & sort tokens
          3. Reshape to (B, HW, D) and apply SS2D
          4. Unsort and reshape back to (B, H, W, D)
          5. Permute back to (B, D, H, W)
        """
        B, D, H, W = input.shape
        HW = H * W

        # Step 1: Convert to token format (B, HW, D)
        x = input.permute(0, 2, 3, 1)  # (B, H, W, D)
        x = x.reshape(B, HW, D)  # (B, HW, D)

        # Step 2: Cluster and sort
        assign = self.token_clustering(x)  # (B, HW)
        x_sorted, unsort_indices = self._gather_tokens(x, assign)  # (B, HW, D), (B, HW)

        # Step 3: Apply normalization and SS2D (inherited _forward)
        x_proc = self._forward(x_sorted.reshape(B, H, W, D))  # (B, H, W, D)
        x_proc_flat = x_proc.reshape(B, HW, D)  # (B, HW, D)

        # Step 4: Unsort tokens
        x_unsorted = x_proc_flat.clone()
        for b in range(B):
            x_unsorted[b] = x_proc_flat[b, unsort_indices[b]]

        # Step 5: Convert back to image format and permute to (B, D, H, W)
        x_out = x_unsorted.reshape(B, H, W, D)  # (B, H, W, D)
        x_out = x_out.permute(0, 3, 1, 2)  # (B, D, H, W)

        return x_out