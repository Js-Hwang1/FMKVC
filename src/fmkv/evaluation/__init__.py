"""
Evaluation Module
=================

Comprehensive evaluation framework for KV cache compression methods.

Benchmarks:
- Perplexity: Language modeling quality on WikiText, C4
- PasskeyRetrieval: Long-context retrieval accuracy
- NeedleInHaystack: Single fact retrieval across depths
- LongBench: Comprehensive multi-task evaluation

Methods:
- Dense: Uncompressed baseline (upper bound quality)
- FMKV: Force-Matched KV compression (our method)
- H2O, StreamingLLM, etc. (coming soon)
"""

# Benchmarks
from fmkv.evaluation.benchmarks import (
    BaseBenchmark,
    BenchmarkResult,
    PerplexityBenchmark,
    PasskeyRetrievalBenchmark,
    NeedleInHaystackBenchmark,
    LongBenchBenchmark,
    get_benchmark,
    list_benchmarks,
)

# Methods
from fmkv.evaluation.methods import (
    BaseMethod,
    MethodConfig,
    DenseMethod,
    FMKVMethod,
    get_method,
    list_methods,
)

# Legacy imports for backwards compatibility
from fmkv.evaluation.passkey import PasskeyRetrievalBenchmark as PasskeyRetrievalBenchmarkOld
from fmkv.evaluation.perplexity import PerplexityEvaluator

__all__ = [
    # Benchmarks
    "BaseBenchmark",
    "BenchmarkResult",
    "PerplexityBenchmark",
    "PasskeyRetrievalBenchmark",
    "NeedleInHaystackBenchmark",
    "LongBenchBenchmark",
    "get_benchmark",
    "list_benchmarks",
    # Methods
    "BaseMethod",
    "MethodConfig",
    "DenseMethod",
    "FMKVMethod",
    "get_method",
    "list_methods",
    # Legacy
    "PerplexityEvaluator",
]
