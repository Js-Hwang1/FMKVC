"""
Sidecar Configuration
=====================

Defines the configuration for the Sidecar network architecture.
Designed to be model-agnostic and configurable via Hydra.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
from enum import Enum


class EncoderType(str, Enum):
    """Encoder architecture type."""
    TRANSFORMER = "transformer"
    GIN = "gin"  # Graph Isomorphism Network
    MLP = "mlp"  # Simple MLP baseline


class AggregatorType(str, Enum):
    """Aggregation method for N -> 1 compression."""
    SET_TRANSFORMER = "set_transformer"  # Learned attention pooling
    MEAN_POOL = "mean_pool"  # Simple averaging (baseline)
    ATTENTION_POOL = "attention_pool"  # Single-head attention pooling


@dataclass
class SidecarConfig:
    """
    Configuration for the Sidecar network.
    
    The Sidecar maps a window of N KV pairs to a single coarse-grained
    super-token that preserves gradient dynamics.
    
    Attributes:
        d_head: Dimension of each attention head in the base model.
        n_heads: Number of attention heads in the base model (for multi-head support).
        window_size: Number of tokens (N) in each compression window.
        encoder_type: Type of encoder architecture.
        encoder_hidden_dim: Hidden dimension of the encoder.
        encoder_num_layers: Number of encoder layers.
        encoder_num_heads: Number of attention heads in encoder (if Transformer).
        encoder_dropout: Dropout rate in encoder.
        aggregator_type: Type of aggregation method.
        aggregator_num_heads: Number of heads in aggregator (if attention-based).
        aggregator_num_inducing: Number of inducing points (Set Transformer).
        use_layer_norm: Whether to use layer normalization.
        use_residual: Whether to use residual connections in encoder.
        position_encoding: Type of positional encoding ("sinusoidal", "learned", "rope", "none").
        output_projection: Whether to use output projection layer.
        dtype: Data type for computations ("float32", "float16", "bfloat16").
    """
    
    # Base model dimensions (set at runtime based on target LLM)
    d_head: int = 128
    n_heads: int = 32
    
    # Compression window
    window_size: int = 64
    
    # Encoder configuration
    encoder_type: EncoderType = EncoderType.TRANSFORMER
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 3
    encoder_num_heads: int = 4
    encoder_dropout: float = 0.1
    encoder_ffn_ratio: float = 4.0  # FFN hidden = encoder_hidden_dim * ffn_ratio
    
    # Aggregator configuration
    aggregator_type: AggregatorType = AggregatorType.SET_TRANSFORMER
    aggregator_num_heads: int = 4
    aggregator_num_inducing: int = 8  # Inducing points for Set Transformer
    
    # Architecture options
    use_layer_norm: bool = True
    use_residual: bool = True
    position_encoding: Literal["sinusoidal", "learned", "rope", "none"] = "learned"
    output_projection: bool = True
    
    # Numerical precision
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    
    # Regularization
    weight_decay: float = 0.01
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        assert self.d_head > 0, "d_head must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.window_size >= 2, "window_size must be at least 2"
        assert self.encoder_num_layers >= 1, "encoder_num_layers must be at least 1"
        
        # Convert string enums if needed
        if isinstance(self.encoder_type, str):
            self.encoder_type = EncoderType(self.encoder_type)
        if isinstance(self.aggregator_type, str):
            self.aggregator_type = AggregatorType(self.aggregator_type)
    
    @property
    def input_dim(self) -> int:
        """Input dimension: concatenated K and V vectors."""
        return 2 * self.d_head
    
    @property
    def output_dim(self) -> int:
        """Output dimension: K_CG and V_CG concatenated."""
        return 2 * self.d_head
    
    @classmethod
    def from_model(
        cls,
        model_name_or_config,
        window_size: int = 64,
        **kwargs
    ) -> "SidecarConfig":
        """
        Create SidecarConfig from a HuggingFace model name or config.
        
        Args:
            model_name_or_config: HuggingFace model name or config object.
            window_size: Compression window size.
            **kwargs: Additional config overrides.
        
        Returns:
            SidecarConfig configured for the target model.
        """
        from transformers import AutoConfig
        
        if isinstance(model_name_or_config, str):
            hf_config = AutoConfig.from_pretrained(model_name_or_config)
        else:
            hf_config = model_name_or_config
        
        # Extract dimensions from various model architectures
        if hasattr(hf_config, "head_dim"):
            d_head = hf_config.head_dim
        elif hasattr(hf_config, "hidden_size") and hasattr(hf_config, "num_attention_heads"):
            d_head = hf_config.hidden_size // hf_config.num_attention_heads
        else:
            raise ValueError(f"Cannot infer head dimension from config: {type(hf_config)}")
        
        n_heads = getattr(hf_config, "num_attention_heads", 32)
        
        return cls(
            d_head=d_head,
            n_heads=n_heads,
            window_size=window_size,
            **kwargs
        )
    
    def estimate_parameters(self) -> int:
        """Estimate total number of parameters in the Sidecar network."""
        input_proj = self.input_dim * self.encoder_hidden_dim
        
        if self.encoder_type == EncoderType.TRANSFORMER:
            # Self-attention + FFN per layer
            attn_params = 4 * self.encoder_hidden_dim ** 2  # Q, K, V, O projections
            ffn_params = 2 * self.encoder_hidden_dim * int(self.encoder_hidden_dim * self.encoder_ffn_ratio)
            encoder_params = self.encoder_num_layers * (attn_params + ffn_params)
        elif self.encoder_type == EncoderType.GIN:
            # MLP per layer
            encoder_params = self.encoder_num_layers * 2 * self.encoder_hidden_dim ** 2
        else:
            encoder_params = self.encoder_num_layers * 2 * self.encoder_hidden_dim ** 2
        
        if self.aggregator_type == AggregatorType.SET_TRANSFORMER:
            # Inducing points + attention
            aggregator_params = (
                self.aggregator_num_inducing * self.encoder_hidden_dim +
                4 * self.encoder_hidden_dim ** 2
            )
        else:
            aggregator_params = self.encoder_hidden_dim ** 2
        
        output_proj = self.encoder_hidden_dim * self.output_dim if self.output_projection else 0
        
        total = input_proj + encoder_params + aggregator_params + output_proj
        return total


@dataclass
class TrainingConfig:
    """Configuration for Sidecar training."""
    
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    max_steps: int = 100000
    
    # Batch size
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    
    # Loss weights
    force_matching_weight: float = 1.0
    consistency_weight: float = 0.1
    
    # Regularization
    gradient_clip_norm: float = 1.0
    
    # Checkpointing
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Logging
    log_steps: int = 10
    wandb_project: Optional[str] = "fmkv"
    wandb_entity: Optional[str] = None
    
    # Data
    num_workers: int = 4
    prefetch_factor: int = 2

