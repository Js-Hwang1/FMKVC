"""
Compression Policies
====================

Policies for deciding when and how to compress KV cache tokens.

Available policies:
- WindowPolicy: Fixed window-based compression (compress every N tokens)
- AdaptivePolicy: Compression based on cache memory budget
- HybridPolicy: Combines attention-based importance with force matching
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class CompressionPolicy(ABC):
    """Base class for compression policies."""
    
    @abstractmethod
    def should_compress(
        self,
        cache_length: int,
        new_tokens: int,
        **kwargs,
    ) -> bool:
        """
        Determine if compression should be triggered.
        
        Args:
            cache_length: Current cache length
            new_tokens: Number of new tokens being added
            **kwargs: Additional context
        
        Returns:
            True if compression should happen
        """
        pass
    
    @abstractmethod
    def get_compression_window(
        self,
        cache_length: int,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Get the window of tokens to compress.
        
        Args:
            cache_length: Current cache length
            **kwargs: Additional context
        
        Returns:
            Tuple of (start_idx, end_idx) for the window to compress
        """
        pass


@dataclass
class WindowPolicy(CompressionPolicy):
    """
    Fixed window-based compression policy.
    
    Compresses the oldest `window_size` tokens whenever the cache
    exceeds `max_length` tokens. Simple but effective.
    
    Example:
        window_size=64, max_length=128
        
        Cache grows: [0...127] (length 128)
        Compress: [0...63] -> super-token
        Cache after: [super-token, 64...127] (length 65)
    """
    
    window_size: int = 64
    max_length: int = 128
    min_dense_tokens: int = 32  # Keep at least this many dense tokens
    
    def should_compress(
        self,
        cache_length: int,
        new_tokens: int = 1,
        **kwargs,
    ) -> bool:
        """Compress when cache exceeds max_length."""
        return cache_length + new_tokens > self.max_length
    
    def get_compression_window(
        self,
        cache_length: int,
        **kwargs,
    ) -> Tuple[int, int]:
        """Return the oldest window of size window_size."""
        return (0, self.window_size)


@dataclass
class AdaptivePolicy(CompressionPolicy):
    """
    Memory-budget based adaptive compression.
    
    Compresses more aggressively when approaching memory limits,
    less aggressively when there's headroom.
    
    Uses a sliding scale:
    - cache < soft_limit: No compression
    - soft_limit <= cache < hard_limit: Compress oldest window
    - cache >= hard_limit: Aggressive compression (larger windows)
    """
    
    window_size: int = 64
    soft_limit: int = 1024  # Start considering compression
    hard_limit: int = 2048  # Force aggressive compression
    aggressive_window_multiplier: float = 2.0  # Larger windows when urgent
    
    def should_compress(
        self,
        cache_length: int,
        new_tokens: int = 1,
        **kwargs,
    ) -> bool:
        """Compress when cache exceeds soft limit."""
        return cache_length + new_tokens > self.soft_limit
    
    def get_compression_window(
        self,
        cache_length: int,
        **kwargs,
    ) -> Tuple[int, int]:
        """Adaptive window size based on memory pressure."""
        if cache_length >= self.hard_limit:
            # Urgent: use larger window
            effective_window = int(self.window_size * self.aggressive_window_multiplier)
        else:
            effective_window = self.window_size
        
        # Don't compress more than available
        effective_window = min(effective_window, cache_length)
        
        return (0, effective_window)


@dataclass
class BudgetPolicy(CompressionPolicy):
    """
    Strict memory budget policy.
    
    Maintains a fixed effective cache size by compressing
    whenever the budget would be exceeded.
    
    Useful for fixed-memory inference scenarios.
    """
    
    budget_tokens: int = 512  # Maximum effective tokens in cache
    window_size: int = 64
    
    def should_compress(
        self,
        cache_length: int,
        new_tokens: int = 1,
        **kwargs,
    ) -> bool:
        """Compress to stay within budget."""
        # Account for the fact that window_size -> 1 super-token
        effective_new_length = cache_length + new_tokens
        return effective_new_length > self.budget_tokens
    
    def get_compression_window(
        self,
        cache_length: int,
        **kwargs,
    ) -> Tuple[int, int]:
        """Compress enough to get back under budget."""
        # Each compression of window_size saves (window_size - 1) tokens
        return (0, self.window_size)


class ImportanceBasedPolicy(CompressionPolicy):
    """
    Importance-weighted compression policy.
    
    Unlike attention-score based methods (H2O), this policy
    uses the learned Sidecar's confidence to decide what to compress.
    
    Tokens that are harder to compress (higher reconstruction error)
    are kept dense longer.
    """
    
    def __init__(
        self,
        sidecar: nn.Module,
        window_size: int = 64,
        max_length: int = 128,
        importance_threshold: float = 0.5,
    ):
        self.sidecar = sidecar
        self.window_size = window_size
        self.max_length = max_length
        self.importance_threshold = importance_threshold
    
    def should_compress(
        self,
        cache_length: int,
        new_tokens: int = 1,
        **kwargs,
    ) -> bool:
        return cache_length + new_tokens > self.max_length
    
    def get_compression_window(
        self,
        cache_length: int,
        keys: Optional[torch.Tensor] = None,
        values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[int, int]:
        """
        Find the most compressible window.
        
        Instead of always compressing the oldest tokens, find the window
        that the Sidecar can compress best (lowest expected error).
        """
        if keys is None or values is None:
            # Fall back to oldest window
            return (0, self.window_size)
        
        # For now, use simple oldest-first
        # TODO: Implement importance scoring using Sidecar confidence
        return (0, self.window_size)
    
    def _compute_window_importance(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        start: int,
        end: int,
    ) -> float:
        """
        Compute importance score for a window.
        
        Higher score = harder to compress = more important.
        """
        # Extract window
        k_window = keys[:, :, start:end, :]
        v_window = values[:, :, start:end, :]
        
        # One approach: use variance as proxy for importance
        # High variance = diverse tokens = harder to compress
        k_var = k_window.var(dim=2).mean().item()
        v_var = v_window.var(dim=2).mean().item()
        
        return k_var + v_var


def create_policy(
    policy_type: str = "window",
    **kwargs,
) -> CompressionPolicy:
    """
    Factory function to create compression policies.
    
    Args:
        policy_type: Type of policy ("window", "adaptive", "budget")
        **kwargs: Policy-specific arguments
    
    Returns:
        CompressionPolicy instance
    """
    policies = {
        "window": WindowPolicy,
        "adaptive": AdaptivePolicy,
        "budget": BudgetPolicy,
    }
    
    if policy_type not in policies:
        raise ValueError(f"Unknown policy type: {policy_type}. Available: {list(policies.keys())}")
    
    return policies[policy_type](**kwargs)

