"""
Evaluation Module
=================

Benchmarks and evaluation utilities for Force-Matched KV compression:
- PasskeyRetrieval: Find a random number hidden in long context
- PerplexityEvaluator: PPL vs compression ratio curves
- NeedleInHaystack: Anomaly preservation test
"""

from fmkv.evaluation.passkey import PasskeyRetrievalBenchmark
from fmkv.evaluation.perplexity import PerplexityEvaluator

__all__ = [
    "PasskeyRetrievalBenchmark",
    "PerplexityEvaluator",
]

