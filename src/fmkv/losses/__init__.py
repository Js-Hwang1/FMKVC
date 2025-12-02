"""
Loss Functions for Force-Matched KV Compression
================================================

The core innovation: training the Sidecar to preserve gradient dynamics
rather than just geometric similarity.

Key components:
    - ForceMatchingLoss: Main loss matching attention Jacobians
    - ConsistencyLoss: Regularization preserving attention output magnitude
    - AttentionJacobian: Efficient computation of ∂Attn/∂q
"""

from fmkv.losses.force_matching import ForceMatchingLoss
from fmkv.losses.consistency import ConsistencyLoss
from fmkv.losses.jacobian import (
    AttentionJacobian,
    compute_attention_jacobian,
    compute_attention_jacobian_batched,
)

__all__ = [
    "ForceMatchingLoss",
    "ConsistencyLoss",
    "AttentionJacobian",
    "compute_attention_jacobian",
    "compute_attention_jacobian_batched",
]

