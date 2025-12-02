"""
Attention Jacobian Computation
==============================

Efficient computation of the Jacobian of the attention output with respect
to the query vector: ∂Attn(q, K, V)/∂q

This gradient represents how the query "pulls" on the attention mechanism,
and matching this Jacobian ensures the compressed cache exerts the same
"force" on future queries.

Mathematical derivation:
    Attn(q, K, V) = softmax(q K^T / √d) V = α V
    
    where α = softmax(q K^T / √d) ∈ R^{1×n}
    
    ∂Attn/∂q = ∂(αV)/∂q
             = V^T ∂α/∂q
    
    For softmax: ∂α_i/∂z_j = α_i (δ_{ij} - α_j)
    And z = q K^T / √d, so ∂z/∂q = K / √d
    
    Therefore:
    ∂Attn/∂q = V^T diag(α) (I - 1 α^T) K / √d
             = (1/√d) V^T [diag(α) K - α^T α K]
             = (1/√d) [Σ_i α_i v_i k_i^T - (Σ_i α_i v_i)(Σ_j α_j k_j)^T]
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange, einsum


def compute_attention_output(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output and weights.
    
    Args:
        query: Query vector of shape (batch, d) or (batch, 1, d)
        keys: Key matrix of shape (batch, seq_len, d)
        values: Value matrix of shape (batch, seq_len, d)
        scale: Scaling factor (default: 1/√d)
    
    Returns:
        Tuple of:
            - attention_output: Shape (batch, d)
            - attention_weights: Shape (batch, seq_len)
    """
    # Handle query shape
    if query.dim() == 2:
        query = query.unsqueeze(1)  # (batch, 1, d)
    
    d = query.size(-1)
    if scale is None:
        scale = d ** -0.5
    
    # Compute attention scores
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale  # (batch, 1, seq_len)
    
    # Softmax
    attn_weights = F.softmax(scores, dim=-1)  # (batch, 1, seq_len)
    
    # Compute output
    attn_output = torch.matmul(attn_weights, values)  # (batch, 1, d)
    
    return attn_output.squeeze(1), attn_weights.squeeze(1)


def compute_attention_jacobian(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
    return_attention: bool = False,
) -> torch.Tensor:
    """
    Compute the Jacobian of attention output w.r.t. query.
    
    ∂Attn(q, K, V)/∂q ∈ R^{d_v × d_q}
    
    This is the "force" that the KV cache exerts on the query direction.
    
    Args:
        query: Query vector of shape (batch, d_q) or (batch, 1, d_q)
        keys: Key matrix of shape (batch, seq_len, d_k)
        values: Value matrix of shape (batch, seq_len, d_v)
        scale: Scaling factor (default: 1/√d_k)
        return_attention: If True, also return attention weights and output
    
    Returns:
        Jacobian tensor of shape (batch, d_v, d_q)
        
        If return_attention=True, returns tuple of:
            - jacobian: Shape (batch, d_v, d_q)
            - attn_output: Shape (batch, d_v)
            - attn_weights: Shape (batch, seq_len)
    
    Mathematical formula:
        J = (1/√d) [Σ_i α_i (v_i ⊗ k_i) - (Σ_i α_i v_i) ⊗ (Σ_j α_j k_j)]
        
        where ⊗ denotes outer product.
    """
    # Handle query shape
    if query.dim() == 2:
        query = query.unsqueeze(1)
    
    batch_size, _, d_q = query.shape
    _, seq_len, d_v = values.shape
    d_k = keys.size(-1)
    
    if scale is None:
        scale = d_k ** -0.5
    
    # Compute attention weights
    scores = torch.matmul(query, keys.transpose(-2, -1)) * scale  # (batch, 1, seq_len)
    attn_weights = F.softmax(scores, dim=-1).squeeze(1)  # (batch, seq_len)
    
    # Compute attention output
    attn_output = torch.matmul(attn_weights.unsqueeze(1), values).squeeze(1)  # (batch, d_v)
    
    # Compute weighted values and keys
    # α_i v_i for each i, then weighted key
    weighted_values = attn_weights.unsqueeze(-1) * values  # (batch, seq_len, d_v)
    weighted_keys = attn_weights.unsqueeze(-1) * keys  # (batch, seq_len, d_k)
    
    # First term: Σ_i α_i (v_i ⊗ k_i)
    # This is: (batch, seq_len, d_v, 1) @ (batch, seq_len, 1, d_k)
    # = (batch, seq_len, d_v, d_k) -> sum over seq_len -> (batch, d_v, d_k)
    term1 = einsum(weighted_values, keys, "b n v, b n k -> b v k")
    
    # Second term: (Σ_i α_i v_i) ⊗ (Σ_j α_j k_j)
    # attn_output: (batch, d_v), weighted_keys.sum(1): (batch, d_k)
    sum_weighted_keys = weighted_keys.sum(dim=1)  # (batch, d_k)
    term2 = einsum(attn_output, sum_weighted_keys, "b v, b k -> b v k")
    
    # Jacobian
    jacobian = scale * (term1 - term2)  # (batch, d_v, d_k)
    
    if return_attention:
        return jacobian, attn_output, attn_weights
    return jacobian


def compute_attention_jacobian_batched(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute Jacobians for multiple queries efficiently.
    
    Args:
        queries: Query vectors of shape (batch, num_queries, d_q)
        keys: Key matrix of shape (batch, seq_len, d_k)
        values: Value matrix of shape (batch, seq_len, d_v)
        scale: Scaling factor
    
    Returns:
        Jacobians of shape (batch, num_queries, d_v, d_q)
    """
    batch_size, num_queries, d_q = queries.shape
    _, seq_len, d_v = values.shape
    d_k = keys.size(-1)
    
    if scale is None:
        scale = d_k ** -0.5
    
    # Compute all attention weights at once
    scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale  # (batch, num_q, seq_len)
    attn_weights = F.softmax(scores, dim=-1)  # (batch, num_q, seq_len)
    
    # Attention outputs
    attn_outputs = torch.matmul(attn_weights, values)  # (batch, num_q, d_v)
    
    # For each query, compute Jacobian
    # Expand for broadcasting
    # attn_weights: (batch, num_q, seq_len) -> (batch, num_q, seq_len, 1, 1)
    weights_expanded = attn_weights.unsqueeze(-1).unsqueeze(-1)
    
    # values: (batch, seq_len, d_v) -> (batch, 1, seq_len, d_v, 1)
    values_expanded = values.unsqueeze(1).unsqueeze(-1)
    
    # keys: (batch, seq_len, d_k) -> (batch, 1, seq_len, 1, d_k)
    keys_expanded = keys.unsqueeze(1).unsqueeze(-2)
    
    # Term 1: Σ_i α_i (v_i ⊗ k_i)
    outer_products = values_expanded * keys_expanded  # (batch, 1, seq_len, d_v, d_k)
    term1 = (weights_expanded * outer_products).sum(dim=2)  # (batch, num_q, d_v, d_k)
    
    # Term 2: (Σ α_i v_i) ⊗ (Σ α_j k_j)
    weighted_keys_sum = torch.matmul(attn_weights, keys)  # (batch, num_q, d_k)
    term2 = einsum(attn_outputs, weighted_keys_sum, "b q v, b q k -> b q v k")
    
    jacobians = scale * (term1 - term2)
    
    return jacobians


class AttentionJacobian(torch.nn.Module):
    """
    Module for computing attention Jacobians with optional caching.
    
    Provides a clean interface for Jacobian computation in training loops.
    """
    
    def __init__(self, d_head: int, cache_intermediates: bool = False):
        super().__init__()
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.cache_intermediates = cache_intermediates
        
        # Cached intermediates for debugging/visualization
        self._cached_attn_weights = None
        self._cached_attn_output = None
    
    def forward(
        self,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian ∂Attn/∂q.
        
        Args:
            query: Shape (batch, d_head) or (batch, num_q, d_head)
            keys: Shape (batch, seq_len, d_head)
            values: Shape (batch, seq_len, d_head)
        
        Returns:
            Jacobian of shape (batch, d_head, d_head) or (batch, num_q, d_head, d_head)
        """
        if query.dim() == 2:
            jacobian, attn_output, attn_weights = compute_attention_jacobian(
                query, keys, values, self.scale, return_attention=True
            )
        else:
            jacobian = compute_attention_jacobian_batched(
                query, keys, values, self.scale
            )
            attn_output = None
            attn_weights = None
        
        if self.cache_intermediates:
            self._cached_attn_weights = attn_weights
            self._cached_attn_output = attn_output
        
        return jacobian
    
    def get_cached_attention(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Return cached attention weights and output."""
        return self._cached_attn_weights, self._cached_attn_output


def compute_aggregate_jacobian(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Compute the aggregate Jacobian over multiple queries.
    
    This is the "True Aggregate Force" in the Force Matching loss:
    Σ_q ∂Attn(q, K, V)/∂q
    
    Args:
        queries: Shape (batch, num_queries, d)
        keys: Shape (batch, seq_len, d)
        values: Shape (batch, seq_len, d)
        scale: Scaling factor
    
    Returns:
        Aggregate Jacobian of shape (batch, d, d)
    """
    jacobians = compute_attention_jacobian_batched(queries, keys, values, scale)
    return jacobians.sum(dim=1)  # Sum over queries


def jacobian_frobenius_distance(
    jacobian1: torch.Tensor,
    jacobian2: torch.Tensor,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Compute Frobenius distance between Jacobians.
    
    Args:
        jacobian1: Shape (batch, d, d) or (batch, num_q, d, d)
        jacobian2: Same shape as jacobian1
        normalize: If True, normalize by matrix size
    
    Returns:
        Distance tensor of shape (batch,) or (batch, num_q)
    """
    diff = jacobian1 - jacobian2
    
    # Frobenius norm: sqrt(sum of squared elements)
    if diff.dim() == 3:
        frob_sq = (diff ** 2).sum(dim=(-2, -1))
    else:
        frob_sq = (diff ** 2).sum(dim=(-2, -1))
    
    if normalize:
        # Normalize by sqrt(d^2) = d
        d = diff.size(-1)
        frob_sq = frob_sq / (d * d)
    
    return frob_sq  # Return squared distance (for loss)

