"""
Gradient Cache
==============

Computes and caches the "True Force" vectors for training.

For each window, we compute:
    F_true = Σ_q ∂Attn(q, K_window, V_window)/∂q

This gradient represents how the window affects query directions,
and is what the Sidecar learns to preserve.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from fmkv.losses.jacobian import (
    compute_attention_jacobian_batched,
    compute_aggregate_jacobian,
)


@dataclass
class CachedGradient:
    """Cached gradient (force) for a single window."""
    
    layer_idx: int
    window_idx: int
    
    # The "true force" Jacobian: (d_head, d_head)
    aggregate_jacobian: torch.Tensor
    
    # Attention outputs for consistency loss: (num_queries, d_head)
    attention_outputs: torch.Tensor
    
    # Reference to original window data
    sample_id: str
    position_offset: int


class GradientCache:
    """
    Computes and caches gradient forces for training data.
    
    This precomputes the "target" forces that the Sidecar should match,
    avoiding the need to compute them on-the-fly during training.
    
    Example:
        >>> cache = GradientCache(d_head=128)
        >>> 
        >>> # Compute gradients for collected trajectories
        >>> for window in trajectories:
        ...     grad = cache.compute_gradient(
        ...         window.keys, window.values, window.future_queries
        ...     )
        ...     cache.add(grad)
        >>> 
        >>> # Save for training
        >>> cache.save("gradients/")
    """
    
    def __init__(
        self,
        d_head: int,
        output_dir: Optional[str] = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.dtype = dtype
        
        self._cache: List[CachedGradient] = []
        
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.output_dir = None
    
    def compute_gradient(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        queries: torch.Tensor,
        layer_idx: int = 0,
        window_idx: int = 0,
        sample_id: str = "",
        position_offset: int = 0,
    ) -> CachedGradient:
        """
        Compute the aggregate Jacobian (force) for a window.
        
        Args:
            keys: Key tensor (window_size, d_head)
            values: Value tensor (window_size, d_head)
            queries: Query tensor (num_queries, d_query)
            layer_idx: Layer index for metadata
            window_idx: Window index for metadata
            sample_id: Sample ID for metadata
            position_offset: Position offset for metadata
        
        Returns:
            CachedGradient with computed force
        """
        # Add batch dimension if needed
        if keys.dim() == 2:
            keys = keys.unsqueeze(0)
            values = values.unsqueeze(0)
        if queries.dim() == 2:
            queries = queries.unsqueeze(0)
        
        # Ensure correct dtype
        keys = keys.to(dtype=self.dtype)
        values = values.to(dtype=self.dtype)
        queries = queries.to(dtype=self.dtype)
        
        # Handle dimension mismatch between queries and keys
        # If queries come from hidden states, they may need projection
        # For now, we'll truncate/pad to match
        q_dim = queries.size(-1)
        k_dim = keys.size(-1)
        
        if q_dim != k_dim:
            # Simple projection: truncate or pad
            if q_dim > k_dim:
                queries = queries[..., :k_dim]
            else:
                padding = torch.zeros(*queries.shape[:-1], k_dim - q_dim, 
                                     dtype=queries.dtype, device=queries.device)
                queries = torch.cat([queries, padding], dim=-1)
        
        # Compute aggregate Jacobian
        jacobians = compute_attention_jacobian_batched(
            queries, keys, values, self.scale
        )  # (1, num_q, d, d)
        
        aggregate_jacobian = jacobians.sum(dim=1).squeeze(0)  # (d, d)
        
        # Compute attention outputs for consistency
        import torch.nn.functional as F
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attention_outputs = torch.matmul(attn_weights, values).squeeze(0)  # (num_q, d)
        
        return CachedGradient(
            layer_idx=layer_idx,
            window_idx=window_idx,
            aggregate_jacobian=aggregate_jacobian.cpu(),
            attention_outputs=attention_outputs.cpu(),
            sample_id=sample_id,
            position_offset=position_offset,
        )
    
    def add(self, gradient: CachedGradient):
        """Add a computed gradient to the cache."""
        self._cache.append(gradient)
    
    def compute_from_trajectories(
        self,
        trajectories: List["TrajectoryWindow"],
        device: torch.device = torch.device("cpu"),
        show_progress: bool = True,
    ) -> List[CachedGradient]:
        """
        Compute gradients for a list of trajectory windows.
        
        Args:
            trajectories: List of TrajectoryWindow objects
            device: Device for computation
            show_progress: Show progress bar
        
        Returns:
            List of CachedGradient objects
        """
        iterator = tqdm(trajectories, desc="Computing gradients") if show_progress else trajectories
        
        gradients = []
        for window in iterator:
            # Move tensors to device
            keys = window.keys.to(device)
            values = window.values.to(device)
            queries = window.future_queries.to(device)
            
            grad = self.compute_gradient(
                keys=keys,
                values=values,
                queries=queries,
                layer_idx=window.layer_idx,
                window_idx=window.window_idx,
                sample_id=window.sample_id,
                position_offset=window.position_offset,
            )
            
            gradients.append(grad)
            self._cache.append(grad)
        
        return gradients
    
    def save(self, path: Optional[Union[str, Path]] = None):
        """Save cached gradients to disk."""
        if path is None:
            if self.output_dir is None:
                raise ValueError("No output path specified")
            path = self.output_dir / "gradients.pt"
        else:
            path = Path(path)
        
        data = {
            "gradients": [
                {
                    "layer_idx": g.layer_idx,
                    "window_idx": g.window_idx,
                    "aggregate_jacobian": g.aggregate_jacobian,
                    "attention_outputs": g.attention_outputs,
                    "sample_id": g.sample_id,
                    "position_offset": g.position_offset,
                }
                for g in self._cache
            ],
            "d_head": self.d_head,
            "num_gradients": len(self._cache),
        }
        
        torch.save(data, path)
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "GradientCache":
        """Load cached gradients from disk."""
        path = Path(path)
        data = torch.load(path)
        
        cache = cls(d_head=data["d_head"])
        
        for g in data["gradients"]:
            cache._cache.append(CachedGradient(**g))
        
        return cache
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __getitem__(self, idx: int) -> CachedGradient:
        return self._cache[idx]
    
    def get_by_layer(self, layer_idx: int) -> List[CachedGradient]:
        """Get all gradients for a specific layer."""
        return [g for g in self._cache if g.layer_idx == layer_idx]

