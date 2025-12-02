"""
Base class for benchmarks.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import torch

from ..methods.base import BaseMethod


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""
    
    benchmark_name: str
    method_name: str
    
    # Primary metrics
    metrics: dict[str, float] = field(default_factory=dict)
    
    # Per-sample results (for detailed analysis)
    sample_results: list[dict] = field(default_factory=list)
    
    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: dict = field(default_factory=dict)
    
    # Error tracking
    errors: list[str] = field(default_factory=list)
    num_samples: int = 0
    num_successful: int = 0
    
    def __repr__(self) -> str:
        metrics_str = ", ".join(f"{k}={v:.4f}" for k, v in self.metrics.items())
        return f"BenchmarkResult({self.benchmark_name}/{self.method_name}: {metrics_str})"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "benchmark_name": self.benchmark_name,
            "method_name": self.method_name,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "config": self.config,
            "num_samples": self.num_samples,
            "num_successful": self.num_successful,
            "errors": self.errors,
        }


class BaseBenchmark(ABC):
    """
    Abstract base class for benchmarks.
    
    All benchmarks must implement:
    - setup(): Load data and prepare benchmark
    - evaluate(): Run evaluation and return results
    """
    
    def __init__(
        self,
        num_samples: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        self.num_samples = num_samples
        self.seed = seed
        self.verbose = verbose
        self.kwargs = kwargs
        self._is_setup = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Benchmark name."""
        pass
    
    @property
    def description(self) -> str:
        """Benchmark description."""
        return ""
    
    @abstractmethod
    def setup(self) -> None:
        """
        Set up the benchmark (load data, prepare examples).
        
        Called once before evaluation.
        """
        pass
    
    @abstractmethod
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """
        Run benchmark evaluation on a method.
        
        Args:
            method: The KV cache method to evaluate
        
        Returns:
            BenchmarkResult with metrics and details
        """
        pass
    
    def log(self, message: str) -> None:
        """Log a message if verbose."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

