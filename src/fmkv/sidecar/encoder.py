"""
Sidecar Encoders
================

Encoder architectures for processing intra-window token dependencies.
These encoders transform a window of KV pairs to capture relationships
(e.g., subject-verb, coreference) before aggregation.

Supported architectures:
    - TransformerEncoder: Standard self-attention encoder
    - GINEncoder: Graph Isomorphism Network (treats window as fully-connected graph)
    - MLPEncoder: Simple MLP baseline
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from fmkv.sidecar.config import SidecarConfig, EncoderType


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence elements.
    
    Supports:
        - Sinusoidal (fixed, from "Attention is All You Need")
        - Learned (trainable embeddings)
        - RoPE (Rotary Position Embedding, applied in attention)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        encoding_type: str = "sinusoidal",
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(p=dropout)
        
        if encoding_type == "sinusoidal":
            pe = self._create_sinusoidal_encoding(d_model, max_len)
            self.register_buffer("pe", pe)
            self.embedding = None
        elif encoding_type == "learned":
            self.embedding = nn.Embedding(max_len, d_model)
            self.register_buffer("pe", None)
        elif encoding_type == "rope":
            # RoPE is applied in attention, not here
            self.register_buffer("pe", None)
            self.embedding = None
        elif encoding_type == "none":
            self.register_buffer("pe", None)
            self.embedding = None
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def _create_sinusoidal_encoding(self, d_model: int, max_len: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added.
        """
        seq_len = x.size(1)
        
        if self.encoding_type == "sinusoidal":
            x = x + self.pe[:, :seq_len, :]
        elif self.encoding_type == "learned":
            positions = torch.arange(seq_len, device=x.device)
            x = x + self.embedding(positions)
        # For "rope" and "none", return x unchanged (RoPE applied in attention)
        
        return self.dropout(x)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional RoPE support.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_rope = use_rope
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        if use_rope:
            self._init_rope()
    
    def _init_rope(self, base: float = 10000.0, max_len: int = 512):
        """Initialize RoPE frequency tensor."""
        inv_freq = 1.0 / (base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        t = torch.arange(max_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0))
    
    def _apply_rope(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Apply rotary position embedding."""
        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]
        
        # Rotate pairs of dimensions
        x1 = x[..., : self.head_dim // 2]
        x2 = x[..., self.head_dim // 2 :]
        
        cos = cos[..., : self.head_dim // 2]
        sin = sin[..., : self.head_dim // 2]
        
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of multi-head self-attention.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            attention_mask: Optional mask of shape (batch, seq_len) or (batch, 1, seq_len, seq_len)
        
        Returns:
            Output tensor of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv_proj(x)
        qkv = rearrange(qkv, "b s (three h d) -> three b h s d", three=3, h=self.num_heads)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Apply RoPE if enabled
        if self.use_rope:
            q = self._apply_rope(q, seq_len)
            k = self._apply_rope(k, seq_len)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                # Expand (batch, seq_len) -> (batch, 1, 1, seq_len)
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(attention_mask == 0, float("-inf"))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Compute output
        out = torch.matmul(attn_weights, v)
        out = rearrange(out, "b h s d -> b s (h d)")
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation (SwiGLU optional)."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        use_swiglu: bool = False,
    ):
        super().__init__()
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            # SwiGLU: gate * swish(x) where gate and x are separate projections
            self.w1 = nn.Linear(d_model, d_ff, bias=False)
            self.w2 = nn.Linear(d_model, d_ff, bias=False)
            self.w3 = nn.Linear(d_ff, d_model, bias=False)
        else:
            self.fc1 = nn.Linear(d_model, d_ff)
            self.fc2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            return self.w3(F.silu(self.w1(x)) * self.w2(x))
        else:
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.dropout(x)
            x = self.fc2(x)
            x = self.dropout(x)
            return x


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with pre-norm architecture."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_rope: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads, dropout, use_rope)
        self.ffn = FeedForward(d_model, d_ff, dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm architecture (more stable training)
        x = x + self.self_attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for processing intra-window dependencies.
    
    This encoder uses self-attention to capture relationships between
    tokens within a compression window (e.g., subject-verb, coreference).
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        
        d_model = config.encoder_hidden_dim
        d_ff = int(d_model * config.encoder_ffn_ratio)
        use_rope = config.position_encoding == "rope"
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, d_model)
        
        # Positional encoding (if not RoPE)
        self.pos_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=config.window_size * 2,  # Buffer for variable window sizes
            encoding_type=config.position_encoding if not use_rope else "none",
            dropout=config.encoder_dropout,
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=config.encoder_num_heads,
                d_ff=d_ff,
                dropout=config.encoder_dropout,
                use_rope=use_rope,
            )
            for _ in range(config.encoder_num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model) if config.use_layer_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a window of KV pairs.
        
        Args:
            x: Input tensor of shape (batch, window_size, 2*d_head)
            attention_mask: Optional mask of shape (batch, window_size)
        
        Returns:
            Encoded tensor of shape (batch, window_size, encoder_hidden_dim)
        """
        # Project input to encoder dimension
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attention_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class GINLayer(nn.Module):
    """
    Graph Isomorphism Network layer.
    
    Treats the token window as a fully-connected graph and applies
    the GIN update: h_v = MLP((1 + eps) * h_v + sum_{u in N(v)} h_u)
    
    For fully-connected graphs, this is equivalent to:
    h_v = MLP((1 + eps) * h_v + mean(h))
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        eps: float = 0.0,
        learn_eps: bool = True,
    ):
        super().__init__()
        
        if learn_eps:
            self.eps = nn.Parameter(torch.tensor(eps))
        else:
            self.register_buffer("eps", torch.tensor(eps))
        
        # MLP for GIN update
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        GIN layer forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
        
        Returns:
            Updated tensor of shape (batch, seq_len, d_model)
        """
        # Aggregate neighbors (fully-connected: all tokens are neighbors)
        neighbor_sum = x.mean(dim=1, keepdim=True).expand_as(x)
        
        # GIN update
        out = self.mlp((1 + self.eps) * x + neighbor_sum)
        
        # Residual + LayerNorm
        return self.norm(x + out)


class GINEncoder(nn.Module):
    """
    Graph Isomorphism Network encoder.
    
    Treats the compression window as a fully-connected graph,
    using GIN layers to capture token relationships.
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        
        d_model = config.encoder_hidden_dim
        d_ff = int(d_model * config.encoder_ffn_ratio)
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, d_model)
        
        # GIN layers
        self.layers = nn.ModuleList([
            GINLayer(
                d_model=d_model,
                d_ff=d_ff,
                dropout=config.encoder_dropout,
            )
            for _ in range(config.encoder_num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model) if config.use_layer_norm else nn.Identity()
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode a window of KV pairs using GIN.
        
        Args:
            x: Input tensor of shape (batch, window_size, 2*d_head)
            attention_mask: Optional mask (applied as weighting in aggregation)
        
        Returns:
            Encoded tensor of shape (batch, window_size, encoder_hidden_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Apply GIN layers
        for layer in self.layers:
            x = layer(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class MLPEncoder(nn.Module):
    """
    Simple MLP encoder baseline.
    
    Processes each position independently without explicit interaction modeling.
    Useful as a baseline to verify that encoder capacity matters.
    """
    
    def __init__(self, config: SidecarConfig):
        super().__init__()
        self.config = config
        
        d_model = config.encoder_hidden_dim
        d_ff = int(d_model * config.encoder_ffn_ratio)
        
        layers = [nn.Linear(config.input_dim, d_model), nn.GELU()]
        
        for _ in range(config.encoder_num_layers - 1):
            layers.extend([
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Dropout(config.encoder_dropout),
                nn.Linear(d_ff, d_model),
                nn.LayerNorm(d_model),
            ])
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        MLP encoder forward pass.
        
        Args:
            x: Input tensor of shape (batch, window_size, 2*d_head)
            attention_mask: Ignored (no cross-position interaction)
        
        Returns:
            Encoded tensor of shape (batch, window_size, encoder_hidden_dim)
        """
        return self.mlp(x)


def create_encoder(config: SidecarConfig) -> nn.Module:
    """Factory function to create encoder based on config."""
    if config.encoder_type == EncoderType.TRANSFORMER:
        return TransformerEncoder(config)
    elif config.encoder_type == EncoderType.GIN:
        return GINEncoder(config)
    elif config.encoder_type == EncoderType.MLP:
        return MLPEncoder(config)
    else:
        raise ValueError(f"Unknown encoder type: {config.encoder_type}")

