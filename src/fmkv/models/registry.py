"""
Model Registry
==============

Registry for supported model architectures with their specific configurations
for KV cache extraction and attention hooking.

Each entry specifies:
- How to access attention layers
- KV cache format and dimensions
- Layer naming conventions
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type
import re


@dataclass
class ModelConfig:
    """Configuration for a specific model architecture."""
    
    # Model family name
    family: str
    
    # Pattern to match model names (regex)
    name_pattern: str
    
    # Path to attention layers (e.g., "model.layers.{i}.self_attn")
    attention_layer_pattern: str
    
    # Names of K, V projection modules within attention
    k_proj_name: str = "k_proj"
    v_proj_name: str = "v_proj"
    q_proj_name: str = "q_proj"
    o_proj_name: str = "o_proj"
    
    # KV cache format: "separate" (K, V as separate tensors) or "tuple"
    kv_cache_format: str = "separate"
    
    # Dimension order: "bshd" (batch, seq, heads, dim) or "bhsd"
    kv_dim_order: str = "bshd"
    
    # Whether model uses grouped-query attention (GQA)
    uses_gqa: bool = False
    num_kv_heads: Optional[int] = None  # If GQA, number of KV heads
    
    # RoPE configuration
    uses_rope: bool = True
    rope_theta: float = 10000.0
    
    # Attribute names for extracting dimensions
    hidden_size_attr: str = "hidden_size"
    num_heads_attr: str = "num_attention_heads"
    num_layers_attr: str = "num_hidden_layers"
    head_dim_attr: Optional[str] = "head_dim"
    
    def get_attention_layer_name(self, layer_idx: int) -> str:
        """Get the full path to an attention layer."""
        return self.attention_layer_pattern.format(i=layer_idx)
    
    def matches(self, model_name: str) -> bool:
        """Check if this config matches a model name."""
        return bool(re.search(self.name_pattern, model_name.lower()))


# Registry of supported models
MODEL_REGISTRY: Dict[str, ModelConfig] = {}


def register_model(config: ModelConfig) -> None:
    """Register a model configuration."""
    MODEL_REGISTRY[config.family] = config


def get_model_config(model_name_or_path: str) -> Optional[ModelConfig]:
    """
    Get the configuration for a model based on its name.
    
    Args:
        model_name_or_path: HuggingFace model name or path
    
    Returns:
        ModelConfig if found, None otherwise
    """
    model_name = model_name_or_path.lower()
    
    for config in MODEL_REGISTRY.values():
        if config.matches(model_name):
            return config
    
    return None


# ============================================================================
# Register supported model families
# ============================================================================

# LLaMA family (Llama-2, Llama-3, Code Llama)
register_model(ModelConfig(
    family="llama",
    name_pattern=r"(llama|meta-llama)",
    attention_layer_pattern="model.layers.{i}.self_attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="o_proj",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=True,  # Llama-2 70B and Llama-3 use GQA
    uses_rope=True,
    rope_theta=10000.0,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
    head_dim_attr="head_dim",
))

# Mistral family
register_model(ModelConfig(
    family="mistral",
    name_pattern=r"mistral",
    attention_layer_pattern="model.layers.{i}.self_attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="o_proj",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=True,  # Mistral uses GQA
    uses_rope=True,
    rope_theta=10000.0,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))

# Qwen family (Qwen, Qwen2)
register_model(ModelConfig(
    family="qwen",
    name_pattern=r"qwen",
    attention_layer_pattern="transformer.h.{i}.attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="c_proj",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=False,
    uses_rope=True,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))

# Qwen2 uses different structure
register_model(ModelConfig(
    family="qwen2",
    name_pattern=r"qwen2",
    attention_layer_pattern="model.layers.{i}.self_attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="o_proj",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=True,
    uses_rope=True,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))

# Phi family (Phi-2, Phi-3)
register_model(ModelConfig(
    family="phi",
    name_pattern=r"phi",
    attention_layer_pattern="model.layers.{i}.self_attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="dense",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=False,
    uses_rope=True,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))

# GPT-2 (for testing with small models)
register_model(ModelConfig(
    family="gpt2",
    name_pattern=r"gpt2",
    attention_layer_pattern="transformer.h.{i}.attn",
    k_proj_name="c_attn",  # GPT-2 uses fused QKV projection
    v_proj_name="c_attn",
    q_proj_name="c_attn",
    o_proj_name="c_proj",
    kv_cache_format="tuple",  # GPT-2 returns (K, V) tuple
    kv_dim_order="bhsd",  # GPT-2 uses batch, heads, seq, dim
    uses_gqa=False,
    uses_rope=False,  # GPT-2 uses learned positional embeddings
    hidden_size_attr="n_embd",
    num_heads_attr="n_head",
    num_layers_attr="n_layer",
))

# GPT-NeoX / Pythia
register_model(ModelConfig(
    family="gpt_neox",
    name_pattern=r"(gpt-neox|pythia)",
    attention_layer_pattern="gpt_neox.layers.{i}.attention",
    k_proj_name="query_key_value",  # Fused QKV
    v_proj_name="query_key_value",
    q_proj_name="query_key_value",
    o_proj_name="dense",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=False,
    uses_rope=True,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))

# OPT family
register_model(ModelConfig(
    family="opt",
    name_pattern=r"opt-",
    attention_layer_pattern="model.decoder.layers.{i}.self_attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="out_proj",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=False,
    uses_rope=False,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))

# Gemma family
register_model(ModelConfig(
    family="gemma",
    name_pattern=r"gemma",
    attention_layer_pattern="model.layers.{i}.self_attn",
    k_proj_name="k_proj",
    v_proj_name="v_proj",
    q_proj_name="q_proj",
    o_proj_name="o_proj",
    kv_cache_format="separate",
    kv_dim_order="bshd",
    uses_gqa=True,
    uses_rope=True,
    hidden_size_attr="hidden_size",
    num_heads_attr="num_attention_heads",
    num_layers_attr="num_hidden_layers",
))


def list_supported_models() -> List[str]:
    """List all supported model families."""
    return list(MODEL_REGISTRY.keys())


def get_model_dimensions(model_config, hf_config) -> Dict[str, int]:
    """
    Extract model dimensions from HuggingFace config.
    
    Returns dict with: hidden_size, num_heads, num_layers, head_dim
    """
    hidden_size = getattr(hf_config, model_config.hidden_size_attr)
    num_heads = getattr(hf_config, model_config.num_heads_attr)
    num_layers = getattr(hf_config, model_config.num_layers_attr)
    
    if model_config.head_dim_attr and hasattr(hf_config, model_config.head_dim_attr):
        head_dim = getattr(hf_config, model_config.head_dim_attr)
    else:
        head_dim = hidden_size // num_heads
    
    # Handle GQA
    num_kv_heads = num_heads
    if model_config.uses_gqa:
        if hasattr(hf_config, "num_key_value_heads"):
            num_kv_heads = hf_config.num_key_value_heads
        elif model_config.num_kv_heads is not None:
            num_kv_heads = model_config.num_kv_heads
    
    return {
        "hidden_size": hidden_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "num_layers": num_layers,
        "head_dim": head_dim,
    }

