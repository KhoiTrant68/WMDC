"""
ablation_models — drop-in architectural variants used by the Table 2
backbone ablation. See `ablation_models/backbone_variants.py`.
"""

from ablation_models.backbone_variants import (
    FDMReversedBlock,
    ResidualCNNBlock,
    SS2DBlock,
    SwinBlock,
    build_backbone,
)

__all__ = [
    "ResidualCNNBlock",
    "SwinBlock",
    "SS2DBlock",
    "FDMReversedBlock",
    "build_backbone",
]
