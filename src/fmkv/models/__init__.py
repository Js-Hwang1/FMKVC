"""
Model Wrappers
==============

Model-agnostic wrappers for HuggingFace transformers that provide:
- KV cache extraction and manipulation
- Attention hook registration
- Gradient capture for training
- Compressed cache injection

Supported model families:
- LLaMA / Llama-2 / Llama-3
- Mistral
- Qwen / Qwen2
- Phi-2 / Phi-3
- GPT-2 / GPT-NeoX (for testing)
"""

from fmkv.models.wrapper import (
    ModelWrapper,
    create_model_wrapper,
)
from fmkv.models.hooks import (
    KVCacheHook,
    AttentionHook,
    GradientCaptureHook,
)
from fmkv.models.registry import (
    MODEL_REGISTRY,
    register_model,
    get_model_config,
)

__all__ = [
    "ModelWrapper",
    "create_model_wrapper",
    "KVCacheHook",
    "AttentionHook",
    "GradientCaptureHook",
    "MODEL_REGISTRY",
    "register_model",
    "get_model_config",
]

