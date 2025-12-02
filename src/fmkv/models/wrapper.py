"""
Model Wrapper
=============

High-level wrapper for HuggingFace models that provides:
1. Unified interface for different model architectures
2. KV cache extraction and manipulation
3. Integration with Sidecar for compression
4. Gradient capture for training
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fmkv.models.registry import (
    get_model_config,
    get_model_dimensions,
    ModelConfig,
)
from fmkv.models.hooks import KVCacheHook, GradientCaptureHook, CacheManager


@dataclass
class ModelInfo:
    """Information about a wrapped model."""
    
    name: str
    family: str
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    head_dim: int
    vocab_size: int
    max_position_embeddings: int
    uses_gqa: bool
    uses_rope: bool
    dtype: torch.dtype
    device: torch.device


class ModelWrapper:
    """
    Wrapper for HuggingFace causal language models.
    
    Provides a unified interface for:
    - Loading and configuring models
    - Extracting KV states during inference
    - Capturing gradients for training
    - Integrating with Sidecar compression
    
    Example:
        >>> wrapper = ModelWrapper.from_pretrained(
        ...     "meta-llama/Llama-2-7b-hf",
        ...     torch_dtype=torch.bfloat16,
        ...     device_map="auto",
        ... )
        >>> 
        >>> # Get model info
        >>> print(wrapper.info)
        >>> 
        >>> # Forward pass with KV capture
        >>> with wrapper.capture_kv():
        ...     outputs = wrapper.forward(input_ids)
        ...     kv_states = wrapper.get_kv_states()
        >>> 
        >>> # Generate with compression
        >>> wrapper.set_sidecar(sidecar)
        >>> outputs = wrapper.generate(input_ids, max_new_tokens=100)
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: Optional[PreTrainedTokenizer] = None,
        model_config: Optional[ModelConfig] = None,
    ):
        """
        Initialize wrapper with a loaded model.
        
        Args:
            model: Pre-trained HuggingFace model
            tokenizer: Optional tokenizer
            model_config: Model architecture config (auto-detected if not provided)
        """
        self.model = model
        self.tokenizer = tokenizer
        
        # Auto-detect model config
        if model_config is None:
            model_name = getattr(model.config, "_name_or_path", "unknown")
            model_config = get_model_config(model_name)
            
            if model_config is None:
                raise ValueError(
                    f"Could not auto-detect config for model: {model_name}. "
                    f"Please provide model_config explicitly."
                )
        
        self.model_config = model_config
        
        # Extract dimensions
        dims = get_model_dimensions(model_config, model.config)
        
        # Build model info
        self.info = ModelInfo(
            name=getattr(model.config, "_name_or_path", "unknown"),
            family=model_config.family,
            hidden_size=dims["hidden_size"],
            num_heads=dims["num_heads"],
            num_kv_heads=dims["num_kv_heads"],
            num_layers=dims["num_layers"],
            head_dim=dims["head_dim"],
            vocab_size=model.config.vocab_size,
            max_position_embeddings=getattr(model.config, "max_position_embeddings", 4096),
            uses_gqa=model_config.uses_gqa,
            uses_rope=model_config.uses_rope,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        
        # Hooks (initialized lazily)
        self._kv_hook: Optional[KVCacheHook] = None
        self._grad_hook: Optional[GradientCaptureHook] = None
        self._cache_manager: Optional[CacheManager] = None
        
        # Sidecar for compression
        self._sidecar: Optional[nn.Module] = None
    
    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = False,
        attn_implementation: Optional[str] = None,
        **kwargs,
    ) -> "ModelWrapper":
        """
        Load a model from HuggingFace Hub or local path.
        
        Args:
            model_name_or_path: Model identifier or path
            torch_dtype: Data type (e.g., torch.bfloat16)
            device_map: Device placement strategy
            trust_remote_code: Allow custom model code
            attn_implementation: Attention implementation ("eager", "sdpa", "flash_attention_2")
            **kwargs: Additional arguments for AutoModelForCausalLM
        
        Returns:
            ModelWrapper instance
        """
        # Determine dtype
        if torch_dtype is None:
            # Default to bfloat16 if available
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        
        # Load config first to check model type
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        
        # Prepare model kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
            "trust_remote_code": trust_remote_code,
            **kwargs,
        }
        
        # Set attention implementation if specified
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            **model_kwargs,
        )
        
        # Load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            tokenizer = None
        
        return cls(model, tokenizer)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        output_attentions: bool = False,
        **kwargs,
    ) -> Any:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            past_key_values: Cached KV states
            use_cache: Whether to return updated cache
            output_attentions: Whether to return attention weights
            **kwargs: Additional model arguments
        
        Returns:
            Model outputs (logits, cache, etc.)
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Generate tokens autoregressively.
        
        Args:
            input_ids: Input token IDs
            max_new_tokens: Maximum tokens to generate
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            **kwargs: Additional generation arguments
        
        Returns:
            Generated token IDs
        """
        return self.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else None,
            **kwargs,
        )
    
    def capture_kv(self, layers: Optional[List[int]] = None):
        """
        Context manager for capturing KV states.
        
        Args:
            layers: Layer indices to capture (None = all)
        
        Returns:
            KVCacheHook context manager
        """
        if self._kv_hook is not None:
            self._kv_hook.remove()
        
        self._kv_hook = KVCacheHook(
            self.model,
            capture_layers=layers,
            model_config=self.model_config,
        )
        return self._kv_hook
    
    def get_kv_states(self, layer_idx: Optional[int] = None):
        """
        Get captured KV states.
        
        Args:
            layer_idx: Specific layer to get (None = all layers)
        
        Returns:
            KV states dict or single layer states
        """
        if self._kv_hook is None:
            raise RuntimeError("No KV capture active. Use capture_kv() first.")
        
        if layer_idx is not None:
            return self._kv_hook.get_layer_states(layer_idx)
        return self._kv_hook.get_captured_states()
    
    def set_sidecar(self, sidecar: nn.Module):
        """Set the Sidecar network for KV compression."""
        self._sidecar = sidecar
    
    def enable_compression(
        self,
        window_size: int = 64,
        compression_threshold: int = 128,
    ):
        """
        Enable KV cache compression.
        
        Args:
            window_size: Compression window size
            compression_threshold: Cache size to trigger compression
        """
        if self._sidecar is None:
            raise RuntimeError("No Sidecar set. Use set_sidecar() first.")
        
        self._cache_manager = CacheManager(
            self.model,
            self._sidecar,
            window_size=window_size,
            compression_threshold=compression_threshold,
        )
    
    def disable_compression(self):
        """Disable KV cache compression."""
        if self._cache_manager is not None:
            self._cache_manager.remove()
            self._cache_manager = None
    
    def extract_kv_for_training(
        self,
        input_ids: torch.Tensor,
        layers: Optional[List[int]] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Extract KV states for Sidecar training.
        
        Runs a forward pass and returns KV states for each layer.
        
        Args:
            input_ids: Input token IDs
            layers: Layers to extract (None = all)
        
        Returns:
            Dict mapping layer_idx -> (keys, values)
        """
        with self.capture_kv(layers) as hook:
            with torch.no_grad():
                _ = self.forward(input_ids, use_cache=True)
            
            states = hook.get_captured_states()
        
        result = {}
        for layer_idx, state in states.items():
            if state.keys is not None and state.values is not None:
                result[layer_idx] = (state.keys.clone(), state.values.clone())
        
        return result
    
    def get_hidden_states(
        self,
        input_ids: torch.Tensor,
        layer_idx: int = -1,
    ) -> torch.Tensor:
        """
        Get hidden states from a specific layer.
        
        Args:
            input_ids: Input token IDs
            layer_idx: Layer index (-1 = last layer)
        
        Returns:
            Hidden state tensor
        """
        outputs = self.forward(
            input_ids,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states
        return hidden_states[layer_idx]
    
    def cleanup(self):
        """Clean up all hooks and resources."""
        if self._kv_hook is not None:
            self._kv_hook.remove()
            self._kv_hook = None
        
        if self._grad_hook is not None:
            self._grad_hook.remove()
            self._grad_hook = None
        
        if self._cache_manager is not None:
            self._cache_manager.remove()
            self._cache_manager = None
    
    def __repr__(self) -> str:
        return (
            f"ModelWrapper(\n"
            f"  name={self.info.name},\n"
            f"  family={self.info.family},\n"
            f"  hidden_size={self.info.hidden_size},\n"
            f"  num_heads={self.info.num_heads},\n"
            f"  num_layers={self.info.num_layers},\n"
            f"  head_dim={self.info.head_dim},\n"
            f"  dtype={self.info.dtype},\n"
            f"  device={self.info.device}\n"
            f")"
        )


def create_model_wrapper(
    model_name_or_path: str,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: str = "auto",
    **kwargs,
) -> ModelWrapper:
    """
    Convenience function to create a ModelWrapper.
    
    Args:
        model_name_or_path: Model identifier
        torch_dtype: Data type
        device_map: Device placement
        **kwargs: Additional arguments
    
    Returns:
        ModelWrapper instance
    """
    return ModelWrapper.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **kwargs,
    )

