from __future__ import annotations

from functools import partial
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from modules.VSS_module import VSSBlock


def _dist_active() -> bool:
    """True only when a real NCCL/Gloo process group is up. Cheap and safe to
    call on every forward — avoids the AttributeError that `dist.is_initialized`
    raises in environments where torch.distributed was never imported."""
    return dist.is_available() and dist.is_initialized()


class TokenClustering(nn.Module):
    """
    K-means clustering for 2D feature map tokens with EMA centroid updates.

    Falls back to identity ordering when HW < 8*cluster_num (too few tokens).
    Centroids are bootstrapped lazily from the first batch seen.

    DDP semantics
    -------------
    `means` is registered as a buffer. PyTorch DDP broadcasts buffers from
    rank 0 at the start of every forward (broadcast_buffers=True by default),
    so per-rank EMA updates on local shards would be silently discarded on
    every step except rank 0's — wasting (N-1)/N of the effective batch for
    centroid estimation. To use the full global batch:

      * Bootstrap is performed on rank 0 only, then broadcast to all ranks
        before the buffer flag is flipped (guarantees byte-identical init).
      * `_center_iter` all-reduces the per-cluster sum and count across ranks
        before applying the EMA update, so every rank applies the SAME global
        update — matching DDP's broadcast-from-rank-0 contract.
    """

    def __init__(
        self,
        cluster_num: int = 8,
        feature_dim: int = None,
        momentum: float = 0.99,
    ):
        super().__init__()
        self.cluster_num = cluster_num
        self.feature_dim = feature_dim
        self.momentum = momentum

        self.register_buffer("initted", torch.tensor(False, dtype=torch.bool))
        if feature_dim is not None:
            self.register_buffer("means", torch.zeros(cluster_num, feature_dim))
        else:
            self.means = None

    @torch.no_grad()
    def _bootstrap_from_batch(self, x_flat: torch.Tensor) -> None:
        if _dist_active():
            if dist.get_rank() == 0:
                n = x_flat.shape[0]
                idx = torch.randperm(n, device=x_flat.device)[: self.cluster_num]
                self.means.copy_(x_flat[idx].detach())
            dist.broadcast(self.means, src=0)
        else:
            n = x_flat.shape[0]
            idx = torch.randperm(n, device=x_flat.device)[: self.cluster_num]
            self.means.copy_(x_flat[idx].detach())
        self.initted.fill_(True)

    @torch.no_grad()
    def _center_iter(self, x_flat: torch.Tensor, assign_flat: torch.Tensor) -> None:
        """Single hard-assignment EMA centroid update — in-place, no graph.

        Aggregates sums and counts across ranks so the EMA uses the full
        global batch rather than only rank 0's local shard (which is what
        broadcast_buffers=True would otherwise enforce).
        """
        D = x_flat.shape[1]
        device = x_flat.device
        K = self.cluster_num

        sums = x_flat.new_zeros(K, D)
        counts = torch.zeros(K, device=device, dtype=torch.float32)
        sums.index_add_(0, assign_flat, x_flat)
        counts.index_add_(
            0, assign_flat, torch.ones_like(assign_flat, dtype=torch.float32)
        )

        if _dist_active():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)

        nonempty = counts > 0
        if nonempty.any():
            new_means = sums[nonempty] / counts[nonempty].unsqueeze(1)
            self.means[nonempty] = self.means[nonempty] * self.momentum + new_means * (
                1.0 - self.momentum
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, HW, D) → assign : (B, HW) int64 cluster IDs in [0, cluster_num)
        """
        B, HW, D = x.shape

        if HW < 8 * self.cluster_num:
            return torch.arange(HW, device=x.device).unsqueeze(0).expand(B, -1)

        x_flat = x.reshape(B * HW, D)

        if not self.initted:
            self._bootstrap_from_batch(x_flat)

        with torch.no_grad():
            dists = torch.cdist(x_flat.float(), self.means.float())  # (B*HW, K)
            assign_flat = dists.argmin(dim=1)  # (B*HW,)

        if self.training:
            self._center_iter(x_flat.detach(), assign_flat)

        return assign_flat.view(B, HW)


class ContentAdaptiveVSSBlock(VSSBlock):
    """
    VSSBlock subclass that reorders tokens via K-means before SS2D scan.

    All parent weights are preserved at identical state_dict keys.
    Token sort/unsort is fully vectorised — no Python loops over batch.

    Design note
    -----------
    After permutation the (H, W) spatial grid fed to SS2D has no meaningful
    spatial correspondence — this is intentional.  Mamba processes tokens
    sequentially; placing similar tokens consecutively improves hidden-state
    transitions.  The residual connection is un-permuted before returning,
    preserving correct spatial output.
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
            cluster_num=cluster_num,
            feature_dim=hidden_dim,
            momentum=0.99,
        )

    @staticmethod
    def _permute_tokens(
        x: torch.Tensor, assign: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sort tokens by cluster ID (vectorised over batch).

        Parameters
        ----------
        x      : (B, HW, D)
        assign : (B, HW)  int64 cluster IDs

        Returns
        -------
        x_sorted   : (B, HW, D)
        unsort_idx : (B, HW)  — apply again to undo the permutation
        """
        B, HW, D = x.shape
        # stable=True: tokens within a cluster keep their original spatial order
        sort_idx = torch.argsort(assign, dim=1, stable=True)  # (B, HW)
        unsort_idx = torch.argsort(sort_idx, dim=1)  # (B, HW)
        # vectorised gather: x_sorted[b, i] = x[b, sort_idx[b, i]]
        idx_exp = sort_idx.unsqueeze(-1).expand(-1, -1, D)
        x_sorted = torch.gather(x, dim=1, index=idx_exp)
        return x_sorted, unsort_idx

    @staticmethod
    def _unpermute_tokens(x: torch.Tensor, unsort_idx: torch.Tensor) -> torch.Tensor:
        """Undo token permutation (vectorised over batch)."""
        idx_exp = unsort_idx.unsqueeze(-1).expand_as(x)
        return torch.gather(x, dim=1, index=idx_exp)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Pipeline:
          1. BCHW → (B, HW, D)
          2. K-means cluster assignment
          3. Sort tokens by cluster (vectorised)
          4. Reshape to (B, H, W, D), apply inherited SS2D via _forward
          5. Flatten, un-sort (vectorised)
          6. Reshape back to BCHW
        """
        B, D, H, W = input.shape
        HW = H * W

        x = input.permute(0, 2, 3, 1).reshape(B, HW, D)  # (B, HW, D)
        assign = self.token_clustering(x)  # (B, HW)

        x_sorted, unsort_idx = self._permute_tokens(x, assign)  # (B, HW, D)

        # SS2D on permuted "grid" — spatial layout is intentionally meaningless
        x_proc = self._forward(x_sorted.reshape(B, H, W, D))  # (B, H, W, D)
        x_out_flat = self._unpermute_tokens(x_proc.reshape(B, HW, D), unsort_idx)

        return x_out_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)  # BCHW
