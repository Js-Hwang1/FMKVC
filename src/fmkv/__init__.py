"""
Force-Matched KV Cache Compression (FMKV)
==========================================

A physics-inspired framework for KV cache compression in Large Language Models,
based on Force Matching / Multiscale Coarse-Graining from Molecular Dynamics.

Instead of selecting tokens based on heuristics (attention scores, similarity),
we learn to synthesize "super-tokens" that preserve the gradient force field
of the original token cluster.

Paper: "Force-Matched KV Cache Compression: A Physics-Inspired Framework"
"""

__version__ = "0.1.0"
__author__ = "Anonymous"

from fmkv.sidecar import Sidecar, SidecarConfig
from fmkv.compression import CompressedKVCache, CompressionPolicy
from fmkv.losses import ForceMatchingLoss

__all__ = [
    "Sidecar",
    "SidecarConfig",
    "CompressedKVCache",
    "CompressionPolicy",
    "ForceMatchingLoss",
    "__version__",
]

