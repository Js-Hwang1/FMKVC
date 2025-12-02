"""
Compressed KV Cache
===================

Implementation of hybrid KV cache that maintains both:
1. Compressed "super-tokens" from older context
2. Dense (uncompressed) tokens from recent context

This allows efficient long-context inference while preserving
recent token granularity for accurate generation.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class LayerCache:
    """Cache for a single transformer layer."""
    
    # Compressed KV tokens (from Sidecar)
    # Each tensor is (batch, num_heads, d_head) representing one super-token
    compressed_keys: List[torch.Tensor] = field(default_factory=list)
    compressed_values: List[torch.Tensor] = field(default_factory=list)
    
    # Dense (uncompressed) KV tokens for recent context
    # Shape: (batch, num_heads, seq_len, d_head)
    dense_keys: Optional[torch.Tensor] = None
    dense_values: Optional[torch.Tensor] = None
    
    # Metadata
    num_compressed_tokens: int = 0  # Original tokens represented by compressed
    window_size: int = 64  # Compression window size
    
    def append_compressed(
        self,
        k_compressed: torch.Tensor,
        v_compressed: torch.Tensor,
        original_window_size: int,
    ):
        """Add a compressed super-token to the cache."""
        self.compressed_keys.append(k_compressed)
        self.compressed_values.append(v_compressed)
        self.num_compressed_tokens += original_window_size
    
    def append_dense(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Add dense tokens to the cache."""
        if self.dense_keys is None:
            self.dense_keys = keys
            self.dense_values = values
        else:
            self.dense_keys = torch.cat([self.dense_keys, keys], dim=2)
            self.dense_values = torch.cat([self.dense_values, values], dim=2)
    
    def get_full_cache(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the full KV cache combining compressed and dense.
        
        Returns:
            Tuple of (keys, values), each of shape (batch, heads, total_len, d_head)
        """
        all_keys = []
        all_values = []
        
        # Add compressed tokens
        if self.compressed_keys:
            # Stack compressed: (num_compressed, batch, heads, d_head) -> (batch, heads, num_compressed, d_head)
            comp_keys = torch.stack(self.compressed_keys, dim=0)
            comp_keys = comp_keys.permute(1, 2, 0, 3)
            comp_values = torch.stack(self.compressed_values, dim=0)
            comp_values = comp_values.permute(1, 2, 0, 3)
            
            all_keys.append(comp_keys)
            all_values.append(comp_values)
        
        # Add dense tokens
        if self.dense_keys is not None:
            all_keys.append(self.dense_keys)
            all_values.append(self.dense_values)
        
        if not all_keys:
            return None, None
        
        full_keys = torch.cat(all_keys, dim=2)
        full_values = torch.cat(all_values, dim=2)
        
        return full_keys, full_values
    
    @property
    def total_effective_length(self) -> int:
        """Total sequence length including compressed tokens."""
        length = len(self.compressed_keys)  # Each compressed = 1 effective token
        if self.dense_keys is not None:
            length += self.dense_keys.size(2)
        return length
    
    @property
    def total_original_length(self) -> int:
        """Original sequence length before compression."""
        length = self.num_compressed_tokens
        if self.dense_keys is not None:
            length += self.dense_keys.size(2)
        return length
    
    @property
    def compression_ratio(self) -> float:
        """Current compression ratio."""
        if self.total_original_length == 0:
            return 1.0
        return self.total_effective_length / self.total_original_length
    
    def clear(self):
        """Clear all cached data."""
        self.compressed_keys.clear()
        self.compressed_values.clear()
        self.dense_keys = None
        self.dense_values = None
        self.num_compressed_tokens = 0


class CompressedKVCache:
    """
    Hybrid KV cache supporting both compressed and dense storage.
    
    Architecture:
        [Compressed Super-Tokens] | [Dense Recent Tokens]
        [====== History ========] | [=== Recent Context ===]
    
    The Sidecar compresses windows of old tokens into super-tokens,
    while recent tokens are kept dense for accurate next-token prediction.
    
    Example:
        >>> cache = CompressedKVCache(
        ...     num_layers=32,
        ...     num_heads=32,
        ...     head_dim=128,
        ...     window_size=64,
        ... )
        >>> 
        >>> # During inference, update cache after each forward pass
        >>> cache.update(layer_idx=0, keys=k, values=v)
        >>> 
        >>> # Get full cache for attention computation
        >>> full_k, full_v = cache.get_layer_cache(layer_idx=0)
    """
    
    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        window_size: int = 64,
        max_dense_length: int = 128,
        dtype: torch.dtype = torch.bfloat16,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize compressed KV cache.
        
        Args:
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            head_dim: Dimension of each head
            window_size: Window size for compression
            max_dense_length: Maximum dense cache length before compression
            dtype: Data type for cache tensors
            device: Device for cache tensors
        """
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.max_dense_length = max_dense_length
        self.dtype = dtype
        self.device = device
        
        # Per-layer caches
        self._layer_caches: Dict[int, LayerCache] = {
            i: LayerCache(window_size=window_size) for i in range(num_layers)
        }
        
        # Sidecar for compression (set externally)
        self._sidecar: Optional[nn.Module] = None
    
    def set_sidecar(self, sidecar: nn.Module):
        """Set the Sidecar network for compression."""
        self._sidecar = sidecar
    
    def update(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
        compress_if_needed: bool = True,
    ):
        """
        Update cache for a layer with new KV states.
        
        Args:
            layer_idx: Layer index
            keys: New keys of shape (batch, heads, seq_len, d_head)
            values: New values of shape (batch, heads, seq_len, d_head)
            compress_if_needed: Whether to trigger compression if cache is full
        """
        cache = self._layer_caches[layer_idx]
        
        # Append to dense cache
        cache.append_dense(keys, values)
        
        # Check if we should compress
        if compress_if_needed and cache.dense_keys is not None:
            dense_len = cache.dense_keys.size(2)
            
            if dense_len >= self.max_dense_length and self._sidecar is not None:
                self._compress_oldest_window(layer_idx)
    
    def _compress_oldest_window(self, layer_idx: int):
        """Compress the oldest window in a layer's dense cache."""
        cache = self._layer_caches[layer_idx]
        
        if cache.dense_keys is None:
            return
        
        dense_len = cache.dense_keys.size(2)
        if dense_len < self.window_size:
            return
        
        # Extract window to compress
        window_keys = cache.dense_keys[:, :, :self.window_size, :]
        window_values = cache.dense_values[:, :, :self.window_size, :]
        
        # Compress using Sidecar
        # Sidecar expects (batch * heads, seq, 2*d_head) format
        batch_size, num_heads, seq_len, head_dim = window_keys.shape
        
        # Reshape for Sidecar: merge batch and heads
        k_flat = window_keys.reshape(batch_size * num_heads, seq_len, head_dim)
        v_flat = window_values.reshape(batch_size * num_heads, seq_len, head_dim)
        
        # Compress
        with torch.no_grad():
            k_cg, v_cg = self._sidecar.compress_cache(k_flat, v_flat)
        
        # Reshape back: (batch * heads, d_head) -> (batch, heads, d_head)
        k_cg = k_cg.reshape(batch_size, num_heads, head_dim)
        v_cg = v_cg.reshape(batch_size, num_heads, head_dim)
        
        # Add to compressed cache
        cache.append_compressed(k_cg, v_cg, self.window_size)
        
        # Remove compressed tokens from dense cache
        cache.dense_keys = cache.dense_keys[:, :, self.window_size:, :]
        cache.dense_values = cache.dense_values[:, :, self.window_size:, :]
        
        # Handle empty dense cache
        if cache.dense_keys.size(2) == 0:
            cache.dense_keys = None
            cache.dense_values = None
    
    def get_layer_cache(
        self,
        layer_idx: int,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get the full cache for a layer.
        
        Returns:
            Tuple of (keys, values), each of shape (batch, heads, total_len, d_head)
        """
        return self._layer_caches[layer_idx].get_full_cache()
    
    def get_all_caches(self) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Get caches for all layers."""
        return {
            i: cache.get_full_cache()
            for i, cache in self._layer_caches.items()
        }
    
    @property
    def effective_length(self) -> int:
        """Current effective sequence length (after compression)."""
        if not self._layer_caches:
            return 0
        # All layers should have same length, use layer 0
        return self._layer_caches[0].total_effective_length
    
    @property
    def original_length(self) -> int:
        """Original sequence length before compression."""
        if not self._layer_caches:
            return 0
        return self._layer_caches[0].total_original_length
    
    @property
    def compression_ratio(self) -> float:
        """Overall compression ratio."""
        if not self._layer_caches:
            return 1.0
        return self._layer_caches[0].compression_ratio
    
    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        return {
            "effective_length": self.effective_length,
            "original_length": self.original_length,
            "compression_ratio": self.compression_ratio,
            "num_compressed_windows": len(self._layer_caches[0].compressed_keys),
            "dense_length": (
                self._layer_caches[0].dense_keys.size(2)
                if self._layer_caches[0].dense_keys is not None
                else 0
            ),
        }
    
    def clear(self):
        """Clear all caches."""
        for cache in self._layer_caches.values():
            cache.clear()
    
    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Move cache to device/dtype."""
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        
        for cache in self._layer_caches.values():
            cache.compressed_keys = [
                k.to(device=device, dtype=dtype) for k in cache.compressed_keys
            ]
            cache.compressed_values = [
                v.to(device=device, dtype=dtype) for v in cache.compressed_values
            ]
            if cache.dense_keys is not None:
                cache.dense_keys = cache.dense_keys.to(device=device, dtype=dtype)
                cache.dense_values = cache.dense_values.to(device=device, dtype=dtype)
        
        return self

