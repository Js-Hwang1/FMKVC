"""
Sidecar Aggregators
===================

Aggregation mechanisms that compress N encoded tokens into a single
coarse-grained representation. This is the N -> 1 bottleneck.

Supported methods:
    - SetTransformerAggregator: Learned attention pooling with inducing points
    - AttentionPooling: Single learned query attending to all tokens
    - MeanPooling: Simple averaging baseline
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fmkv.sidecar.config import SidecarConfig, AggregatorType


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention for cross-attention in aggregators.
    
    Unlike self-attention, Q comes from a different source than K, V.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cross-attention forward pass.
        
        Args:
            query: Query tensor of shape (batch, q_len, d_model)
            key: Key tensor of shape (batch, kv_len, d_model)
            value: Value tensor of shape (batch, kv_len, d_model)
            attention_mask: Optional mask of shape (batch, kv_len)
        
        Returns:
            Output tensor of shape (batch, q_len, d_model)
        """
        batch_size = query.size(0)
        
        # Project Q, K, V
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head attention
        q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads)
        k = rearrange(k, "b k (h d) -> b h k d", h=self.num_heads)
        v = rearrange(v, "b k (h d) -> b h k d", h=self.num_heads)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            # Expand mask for broadcasting
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, "b h q d -> b q (h d)")
        out = self.out_proj(out)
        
        return out


class SetAttentionBlock(nn.Module):
    """
    Set Attention Block (SAB) from "Set Transformer" paper.
    
    Applies self-attention to a set of elements.
    SAB(X) = LayerNorm(X + MHA(X, X, X))
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Self-attention block."""
        # Pre-norm self-attention
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set Attention Block (ISAB) from "Set Transformer" paper.

    Uses inducing points to reduce attention complexity from O(n^2) to O(nm),
    where m is the number of inducing points.

    ISAB(X) = MAB(X, MAB(I, X))
    where MAB(X, Y) = LayerNorm(X + MHA(X, Y, Y))

    v5 Fix: Inducing points are initialized orthogonally to encourage diverse representations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        # v5: Learnable inducing points with orthogonal initialization
        self.inducing_points = nn.Parameter(torch.empty(1, num_inducing, d_model))
        if num_inducing > 1 and num_inducing <= d_model:
            nn.init.orthogonal_(self.inducing_points.squeeze(0))
        else:
            nn.init.normal_(self.inducing_points, mean=0.0, std=0.5)
        
        # First attention: inducing points attend to input
        self.norm1 = nn.LayerNorm(d_model)
        self.attn1 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Second attention: input attends to updated inducing points
        self.norm2 = nn.LayerNorm(d_model)
        self.attn2 = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ISAB forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size = x.size(0)
        
        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)
        
        # First attention: I attends to X
        I_normed = self.norm1(I)
        x_normed = self.norm1(x)
        H = I + self.attn1(I_normed, x_normed, x_normed, attention_mask)
        
        # Second attention: X attends to H
        x_normed = self.norm2(x)
        H_normed = self.norm2(H)
        out = x + self.attn2(x_normed, H_normed, H_normed)
        
        # Feed-forward
        out = out + self.ffn(self.norm3(out))
        
        return out


class PoolingByMultiheadAttention(nn.Module):
    """
    Pooling by Multihead Attention (PMA) from "Set Transformer" paper.

    Uses learnable seed vectors to pool set elements into fixed-size output.
    PMA(X) = MAB(S, X) where S are learnable seed vectors.

    v5 Fix: Seeds are initialized orthogonally to break symmetry and encourage
    the network to produce diverse super-tokens from the start.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_outputs: int,  # Number of output vectors (1 for our use case)
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.num_outputs = num_outputs

        # v5: Learnable seed vectors with orthogonal initialization
        # Orthogonal init breaks symmetry and encourages diverse outputs
        self.seeds = nn.Parameter(torch.empty(1, num_outputs, d_model))
        if num_outputs > 1 and num_outputs <= d_model:
            # Use orthogonal init when we have multiple seeds
            nn.init.orthogonal_(self.seeds.squeeze(0))
        else:
            # Fallback for single seed or when num_outputs > d_model
            nn.init.normal_(self.seeds, mean=0.0, std=0.5)
        
        # Attention from seeds to input
        self.norm1 = nn.LayerNorm(d_model)
        self.norm_seeds = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool input set to fixed number of outputs.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Output tensor of shape (batch, num_outputs, d_model)
        """
        batch_size = x.size(0)
        
        # Expand seeds for batch
        S = self.seeds.expand(batch_size, -1, -1)
        
        # Attention: seeds attend to input
        S_normed = self.norm_seeds(S)
        x_normed = self.norm1(x)
        out = S + self.attn(S_normed, x_normed, x_normed, attention_mask)
        
        # Feed-forward
        out = out + self.ffn(self.norm2(out))
        
        return out


class SetTransformerAggregator(nn.Module):
    """
    Set Transformer-based aggregator for N -> 1 compression.
    
    Architecture:
        1. Optional ISAB layers for set-to-set transformation
        2. PMA layer to pool to single output
    
    This learns to aggregate a variable-size set into a single vector
    in a permutation-invariant way (order within window doesn't matter
    after positional encoding in the encoder).
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        
        d_model = config.encoder_hidden_dim
        d_ff = int(d_model * config.encoder_ffn_ratio)
        
        # Optional ISAB layers before pooling
        self.isab_layers = nn.ModuleList([
            InducedSetAttentionBlock(
                d_model=d_model,
                num_heads=config.aggregator_num_heads,
                num_inducing=config.aggregator_num_inducing,
                d_ff=d_ff,
                dropout=config.encoder_dropout,
            )
            for _ in range(1)  # Single ISAB layer is usually sufficient
        ])
        
        # Pooling layer: N -> 1
        self.pooling = PoolingByMultiheadAttention(
            d_model=d_model,
            num_heads=config.aggregator_num_heads,
            num_outputs=1,  # Single coarse-grained token
            d_ff=d_ff,
            dropout=config.encoder_dropout,
        )
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Aggregate window tokens to single representation.
        
        Args:
            x: Encoded tokens of shape (batch, window_size, encoder_hidden_dim)
            attention_mask: Optional mask of shape (batch, window_size)
        
        Returns:
            Aggregated tensor of shape (batch, encoder_hidden_dim)
        """
        # Apply ISAB layers
        for isab in self.isab_layers:
            x = isab(x, attention_mask)
        
        # Pool to single output
        out = self.pooling(x, attention_mask)  # (batch, 1, d_model)
        
        return out.squeeze(1)  # (batch, d_model)


class AttentionPooling(nn.Module):
    """
    Simple attention pooling with a single learned query.
    
    A lightweight alternative to Set Transformer.
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        
        d_model = config.encoder_hidden_dim
        
        # Learnable query vector
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.5)
        
        # Attention
        self.attn = MultiHeadAttention(
            d_model=d_model,
            num_heads=config.aggregator_num_heads,
            dropout=config.encoder_dropout,
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Pool via attention with learned query.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            attention_mask: Optional mask
        
        Returns:
            Pooled output of shape (batch, d_model)
        """
        batch_size = x.size(0)
        
        # Expand query for batch
        q = self.query.expand(batch_size, -1, -1)
        
        # Attend to input
        out = self.attn(q, x, x, attention_mask)
        out = self.norm(out)
        
        return out.squeeze(1)


class MeanPooling(nn.Module):
    """
    Simple mean pooling baseline.
    
    Averages all token representations. Used as a baseline to
    verify that learned aggregation provides benefits.
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.norm = nn.LayerNorm(config.encoder_hidden_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Mean pool over sequence dimension.
        
        Args:
            x: Input of shape (batch, seq_len, d_model)
            attention_mask: Optional mask of shape (batch, seq_len)
        
        Returns:
            Pooled output of shape (batch, d_model)
        """
        if attention_mask is not None:
            # Masked mean pooling
            mask = attention_mask.unsqueeze(-1).float()
            x = x * mask
            out = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            out = x.mean(dim=1)
        
        return self.norm(out)


def create_aggregator(config: SidecarConfig) -> nn.Module:
    """Factory function to create aggregator based on config."""
    if config.aggregator_type == AggregatorType.SET_TRANSFORMER:
        return SetTransformerAggregator(config)
    elif config.aggregator_type == AggregatorType.ATTENTION_POOL:
        return AttentionPooling(config)
    elif config.aggregator_type == AggregatorType.MEAN_POOL:
        return MeanPooling(config)
    else:
        raise ValueError(f"Unknown aggregator type: {config.aggregator_type}")

