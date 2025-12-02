"""
Sidecar Network Module
======================

The Sidecar is a lightweight auxiliary network that learns to compress
N KV pairs into a single "super-token" that preserves gradient dynamics.

Architecture:
    Input: (B, N, 2*d_head) - concatenated K,V vectors for window
    Encoder: GIN or Transformer to capture intra-window dependencies
    Aggregator: Learned attention pooling (Set Transformer style)
    Output: (B, 2*d_head) - compressed K_CG, V_CG
"""

from fmkv.sidecar.config import SidecarConfig
from fmkv.sidecar.network import Sidecar
from fmkv.sidecar.encoder import (
    TransformerEncoder,
    GINEncoder,
)
from fmkv.sidecar.aggregator import SetTransformerAggregator

__all__ = [
    "Sidecar",
    "SidecarConfig",
    "TransformerEncoder",
    "GINEncoder",
    "SetTransformerAggregator",
]

