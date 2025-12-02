"""
Evaluation method wrappers for KV cache compression comparison.

Each method implements a common interface for fair benchmarking:
- Dense (uncompressed baseline)
- FMKV (our method)
- H2O (Heavy-Hitter Oracle) - TODO
- StreamingLLM - TODO
- SnapKV - TODO
- PyramidKV - TODO
"""

from .base import BaseMethod, MethodConfig
from .dense import DenseMethod
from .fmkv import FMKVMethod

__all__ = [
    "BaseMethod",
    "MethodConfig",
    "DenseMethod",
    "FMKVMethod",
    "get_method",
    "list_methods",
]

_METHOD_REGISTRY = {
    "dense": DenseMethod,
    "fmkv": FMKVMethod,
    # Future baselines:
    # "h2o": H2OMethod,
    # "streamingllm": StreamingLLMMethod,
    # "snapkv": SnapKVMethod,
    # "pyramidkv": PyramidKVMethod,
}


def get_method(name: str, **kwargs) -> BaseMethod:
    """Get method by name."""
    if name not in _METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method: {name}. Available: {list(_METHOD_REGISTRY.keys())}"
        )
    return _METHOD_REGISTRY[name](**kwargs)


def list_methods() -> list[str]:
    """List available methods."""
    return list(_METHOD_REGISTRY.keys())

