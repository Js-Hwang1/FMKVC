"""
Benchmark implementations for KV cache compression evaluation.

Standard benchmarks:
- Perplexity: Language modeling quality (WikiText-2, WikiText-103, C4)
- Passkey Retrieval: Long-context retrieval accuracy
- LongBench: Comprehensive long-context benchmark suite
- Needle-in-Haystack: Single fact retrieval in long context
- Memory/Latency: Efficiency metrics
"""

from .base import BaseBenchmark, BenchmarkResult
from .perplexity import PerplexityBenchmark
from .passkey import PasskeyRetrievalBenchmark
from .needle import NeedleInHaystackBenchmark
from .longbench import LongBenchBenchmark

__all__ = [
    "BaseBenchmark",
    "BenchmarkResult",
    "PerplexityBenchmark",
    "PasskeyRetrievalBenchmark",
    "NeedleInHaystackBenchmark",
    "LongBenchBenchmark",
    "get_benchmark",
    "list_benchmarks",
]

_BENCHMARK_REGISTRY = {
    "perplexity": PerplexityBenchmark,
    "passkey": PasskeyRetrievalBenchmark,
    "needle": NeedleInHaystackBenchmark,
    "longbench": LongBenchBenchmark,
}


def get_benchmark(name: str, **kwargs) -> BaseBenchmark:
    """Get benchmark by name."""
    if name not in _BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark: {name}. Available: {list(_BENCHMARK_REGISTRY.keys())}"
        )
    return _BENCHMARK_REGISTRY[name](**kwargs)


def list_benchmarks() -> list[str]:
    """List available benchmarks."""
    return list(_BENCHMARK_REGISTRY.keys())

