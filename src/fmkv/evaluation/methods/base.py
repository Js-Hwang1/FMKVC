"""
Base class for KV cache compression methods.

This provides a unified interface for benchmarking different methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class MethodConfig:
    """Configuration for evaluation methods."""
    
    # Model settings
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    
    # Cache budget (for compression methods)
    # None means unlimited (dense)
    cache_budget: Optional[int] = None
    compression_ratio: Optional[float] = None  # Alternative: specify ratio
    
    # Method-specific settings
    method_kwargs: dict = field(default_factory=dict)
    
    def __post_init__(self):
        # Parse dtype
        self.torch_dtype_parsed = getattr(torch, self.torch_dtype, torch.float32)


@dataclass
class GenerationOutput:
    """Output from generation with metadata."""
    
    sequences: torch.Tensor
    text: list[str]
    
    # Performance metrics
    prefill_time: float = 0.0
    decode_time: float = 0.0
    total_time: float = 0.0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    cache_memory_mb: float = 0.0
    
    # Cache stats
    cache_size: int = 0  # Number of tokens in cache
    original_size: int = 0  # Original sequence length
    compression_ratio: float = 1.0


class BaseMethod(ABC):
    """
    Abstract base class for KV cache methods.
    
    All methods must implement:
    - setup(): Load model and prepare method
    - generate(): Generate with the method's KV cache strategy
    - get_cache_stats(): Return cache statistics
    """
    
    def __init__(self, config: MethodConfig):
        self.config = config
        self.model: Optional[Any] = None  # PreTrainedModel
        self.tokenizer: Optional[Any] = None  # PreTrainedTokenizer
        self._is_setup = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Method name for logging."""
        pass
    
    @property
    def is_compression_method(self) -> bool:
        """Whether this method compresses the KV cache."""
        return False
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the method (load model, initialize components).
        
        This is called once before evaluation.
        """
        pass
    
    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> GenerationOutput:
        """
        Generate tokens using this method's KV cache strategy.
        
        Args:
            input_ids: (batch, seq_len) input token IDs
            attention_mask: Optional attention mask
            max_new_tokens: Number of tokens to generate
            **kwargs: Additional generation arguments
        
        Returns:
            GenerationOutput with sequences and metrics
        """
        pass
    
    @abstractmethod
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute language modeling loss for perplexity.
        
        Args:
            input_ids: Input token IDs
            labels: Target token IDs (usually same as input_ids shifted)
            attention_mask: Optional attention mask
        
        Returns:
            Loss tensor
        """
        pass
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics after generation."""
        return {
            "method": self.name,
            "is_compression": self.is_compression_method,
        }
    
    def reset_cache(self) -> None:
        """Reset any cached state between evaluations."""
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

