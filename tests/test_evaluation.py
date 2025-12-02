"""
Tests for evaluation framework.

Note: These tests require transformers and datasets libraries.
Run with: pip install transformers datasets
"""

import pytest
import torch

# Check for optional dependencies
try:
    import transformers
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import datasets
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

requires_transformers = pytest.mark.skipif(
    not HAS_TRANSFORMERS,
    reason="transformers library not installed"
)

requires_datasets = pytest.mark.skipif(
    not HAS_DATASETS,
    reason="datasets library not installed"
)

requires_eval_deps = pytest.mark.skipif(
    not (HAS_TRANSFORMERS and HAS_DATASETS),
    reason="transformers and datasets libraries required"
)


class TestMethodConfig:
    """Test method configuration."""
    
    @requires_transformers
    def test_default_config(self):
        from fmkv.evaluation.methods.base import MethodConfig
        
        config = MethodConfig()
        
        assert config.model_name == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        assert config.torch_dtype == "bfloat16"
        assert config.device == "cuda"
        assert config.cache_budget is None
        assert config.compression_ratio is None
    
    @requires_transformers
    def test_custom_config(self):
        from fmkv.evaluation.methods.base import MethodConfig
        
        config = MethodConfig(
            model_name="test/model",
            torch_dtype="float16",
            device="cpu",
            compression_ratio=0.5,
        )
        
        assert config.model_name == "test/model"
        assert config.torch_dtype_parsed == torch.float16
        assert config.compression_ratio == 0.5


class TestBenchmarkResult:
    """Test benchmark result container."""
    
    @requires_eval_deps
    def test_benchmark_result_creation(self):
        from fmkv.evaluation.benchmarks.base import BenchmarkResult
        
        result = BenchmarkResult(
            benchmark_name="test_bench",
            method_name="test_method",
            metrics={"accuracy": 0.95, "f1": 0.92},
            num_samples=100,
            num_successful=98,
        )
        
        assert result.benchmark_name == "test_bench"
        assert result.method_name == "test_method"
        assert result.metrics["accuracy"] == 0.95
        assert result.num_samples == 100
        assert len(result.errors) == 0
    
    @requires_eval_deps
    def test_result_to_dict(self):
        from fmkv.evaluation.benchmarks.base import BenchmarkResult
        
        result = BenchmarkResult(
            benchmark_name="test",
            method_name="dense",
            metrics={"ppl": 10.5},
        )
        
        d = result.to_dict()
        
        assert "benchmark_name" in d
        assert "method_name" in d
        assert "metrics" in d
        assert d["metrics"]["ppl"] == 10.5


class TestMethodRegistry:
    """Test method registry."""
    
    @requires_eval_deps
    def test_list_methods(self):
        from fmkv.evaluation.methods import list_methods
        
        methods = list_methods()
        
        assert "dense" in methods
        assert "fmkv" in methods
    
    @requires_eval_deps
    def test_get_unknown_method(self):
        from fmkv.evaluation.methods import get_method
        
        with pytest.raises(ValueError, match="Unknown method"):
            get_method("nonexistent_method")


class TestBenchmarkRegistry:
    """Test benchmark registry."""
    
    @requires_eval_deps
    def test_list_benchmarks(self):
        from fmkv.evaluation.benchmarks import list_benchmarks
        
        benchmarks = list_benchmarks()
        
        assert "perplexity" in benchmarks
        assert "passkey" in benchmarks
        assert "needle" in benchmarks
        assert "longbench" in benchmarks
    
    @requires_eval_deps
    def test_get_unknown_benchmark(self):
        from fmkv.evaluation.benchmarks import get_benchmark
        
        with pytest.raises(ValueError, match="Unknown benchmark"):
            get_benchmark("nonexistent_benchmark")


class TestPerplexityBenchmark:
    """Test perplexity benchmark."""
    
    @requires_eval_deps
    def test_benchmark_creation(self):
        from fmkv.evaluation.benchmarks import PerplexityBenchmark
        
        bench = PerplexityBenchmark(
            dataset_name="wikitext-2",
            max_length=512,
            num_samples=10,
        )
        
        assert bench.name == "perplexity_wikitext-2"
        assert bench.max_length == 512
        assert bench.num_samples == 10
    
    @requires_eval_deps
    def test_invalid_dataset(self):
        from fmkv.evaluation.benchmarks import PerplexityBenchmark
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            PerplexityBenchmark(dataset_name="invalid_dataset")


class TestPasskeyBenchmark:
    """Test passkey retrieval benchmark."""
    
    @requires_eval_deps
    def test_benchmark_creation(self):
        from fmkv.evaluation.benchmarks import PasskeyRetrievalBenchmark
        
        bench = PasskeyRetrievalBenchmark(
            context_lengths=[512, 1024],
            passkey_positions=[0.25, 0.5, 0.75],
            num_samples_per_config=5,
        )
        
        assert bench.name == "passkey_retrieval"
        assert bench.context_lengths == [512, 1024]
        assert len(bench.passkey_positions) == 3
    
    @requires_eval_deps
    def test_passkey_generation(self):
        from fmkv.evaluation.benchmarks.passkey import PasskeyRetrievalBenchmark
        import random
        
        bench = PasskeyRetrievalBenchmark(passkey_length=5)
        rng = random.Random(42)
        
        passkey = bench._generate_passkey(rng)
        
        assert len(passkey) == 5
        assert passkey.isdigit()


class TestNeedleBenchmark:
    """Test needle-in-haystack benchmark."""
    
    @requires_eval_deps
    def test_benchmark_creation(self):
        from fmkv.evaluation.benchmarks import NeedleInHaystackBenchmark
        
        bench = NeedleInHaystackBenchmark(
            context_lengths=[1024, 2048],
            depth_percents=[0, 50, 100],
        )
        
        assert bench.name == "needle_in_haystack"
        assert len(bench.context_lengths) == 2
        assert len(bench.depth_percents) == 3
    
    @requires_eval_deps
    def test_answer_checking(self):
        from fmkv.evaluation.benchmarks.needle import NeedleInHaystackBenchmark
        
        bench = NeedleInHaystackBenchmark()
        
        # Exact match
        is_correct, score = bench._check_answer(
            "The answer is truffle oil",
            "truffle oil"
        )
        assert is_correct
        assert score == 1.0
        
        # Partial match
        is_correct, score = bench._check_answer(
            "I think it's truffle",
            "truffle oil"
        )
        assert score > 0


class TestLongBenchBenchmark:
    """Test LongBench benchmark."""
    
    @requires_eval_deps
    def test_benchmark_creation(self):
        from fmkv.evaluation.benchmarks import LongBenchBenchmark
        
        bench = LongBenchBenchmark(
            datasets=["narrativeqa", "hotpotqa"],
            num_samples_per_dataset=10,
        )
        
        assert bench.name == "longbench"
        assert len(bench.selected_datasets) == 2
    
    @requires_eval_deps
    def test_invalid_dataset(self):
        from fmkv.evaluation.benchmarks import LongBenchBenchmark
        
        with pytest.raises(ValueError, match="Unknown dataset"):
            LongBenchBenchmark(datasets=["invalid_dataset"])
    
    @requires_eval_deps
    def test_f1_computation(self):
        from fmkv.evaluation.benchmarks.longbench import LongBenchBenchmark
        
        bench = LongBenchBenchmark()
        
        # Perfect match
        f1 = bench._compute_f1("the quick brown fox", "the quick brown fox")
        assert f1 == 1.0
        
        # Partial match
        f1 = bench._compute_f1("the quick brown", "the quick brown fox")
        assert 0 < f1 < 1
        
        # No match
        f1 = bench._compute_f1("hello world", "goodbye universe")
        assert f1 == 0.0


class TestProfiling:
    """Test profiling utilities (standalone, no external deps)."""
    
    def test_profile_result(self):
        """Profile result doesn't need external deps."""
        # Import directly to avoid triggering evaluation.__init__
        import sys
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "profiling", 
            "src/fmkv/evaluation/profiling.py"
        )
        profiling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiling)
        
        result = profiling.ProfileResult(
            peak_memory_bytes=1024 * 1024 * 100,  # 100MB
            total_time=2.0,
            output_tokens=100,
        )
        
        assert result.peak_memory_mb == 100.0
        assert result.tokens_per_second == 50.0
        assert result.time_per_token_ms == 20.0
    
    def test_kv_cache_memory_estimation(self):
        """KV cache memory estimation doesn't need external deps."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "profiling", 
            "src/fmkv/evaluation/profiling.py"
        )
        profiling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiling)
        
        # Estimate for a small model
        memory = profiling.estimate_kv_cache_memory(
            batch_size=1,
            num_layers=12,
            num_heads=12,
            head_dim=64,
            seq_len=512,
            dtype=torch.float16,
        )
        
        # Expected: 1 * 12 * 512 * 64 * 12 * 2 (K+V) * 2 (bytes) = 18,874,368 bytes
        expected = 1 * 12 * 12 * 512 * 64 * 2 * 2
        assert memory == expected
    
    def test_memory_tracker(self):
        """Memory tracker doesn't need external deps."""
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "profiling", 
            "src/fmkv/evaluation/profiling.py"
        )
        profiling = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(profiling)
        
        # CPU tracker should work but return 0
        tracker = profiling.MemoryTracker(device="cpu")
        tracker.reset()
        tracker.snapshot("test")
        
        assert tracker.get_peak() == 0  # No CUDA
        assert tracker.get_current() == 0


class TestGenerationOutput:
    """Test generation output container."""
    
    @requires_transformers
    def test_output_creation(self):
        from fmkv.evaluation.methods.base import GenerationOutput
        
        output = GenerationOutput(
            sequences=torch.zeros(1, 100),
            text=["Hello world"],
            total_time=1.5,
            peak_memory_mb=100.0,
            cache_size=100,
            original_size=50,
            compression_ratio=0.5,
        )
        
        assert output.total_time == 1.5
        assert output.compression_ratio == 0.5
        assert len(output.text) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
