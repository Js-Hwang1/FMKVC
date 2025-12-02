"""
KV Cache Compression
====================

Components for managing compressed KV caches during inference:
- CompressedKVCache: Hybrid cache with compressed + dense tokens
- CompressionPolicy: When and how to trigger compression
- CacheIntegration: Hooks for integrating with model forward passes
"""

from fmkv.compression.cache import CompressedKVCache
from fmkv.compression.policy import CompressionPolicy, WindowPolicy, AdaptivePolicy

__all__ = [
    "CompressedKVCache",
    "CompressionPolicy",
    "WindowPolicy",
    "AdaptivePolicy",
]

