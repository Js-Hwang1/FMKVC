"""
Memory and latency profiling utilities.

Provides detailed metrics for KV cache compression efficiency:
- Peak GPU memory usage
- KV cache memory estimation
- Throughput (tokens/second)
- Latency breakdown (prefill vs decode)
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch


@dataclass
class ProfileResult:
    """Results from profiling a forward/generate pass."""
    
    # Memory metrics (bytes)
    peak_memory_bytes: int = 0
    allocated_memory_bytes: int = 0
    reserved_memory_bytes: int = 0
    
    # Estimated KV cache memory
    kv_cache_memory_bytes: int = 0
    
    # Timing metrics (seconds)
    total_time: float = 0.0
    prefill_time: float = 0.0
    decode_time: float = 0.0
    
    # Throughput metrics
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_bytes / 1024 / 1024
    
    @property
    def kv_cache_memory_mb(self) -> float:
        return self.kv_cache_memory_bytes / 1024 / 1024
    
    @property
    def tokens_per_second(self) -> float:
        if self.total_time > 0:
            return self.output_tokens / self.total_time
        return 0.0
    
    @property
    def time_per_token_ms(self) -> float:
        if self.output_tokens > 0:
            return (self.total_time / self.output_tokens) * 1000
        return 0.0
    
    def to_dict(self) -> dict:
        return {
            "peak_memory_mb": self.peak_memory_mb,
            "kv_cache_memory_mb": self.kv_cache_memory_mb,
            "total_time_s": self.total_time,
            "prefill_time_s": self.prefill_time,
            "decode_time_s": self.decode_time,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_per_second": self.tokens_per_second,
            "time_per_token_ms": self.time_per_token_ms,
        }


class MemoryTracker:
    """Track GPU memory usage during operations."""
    
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.enabled = torch.cuda.is_available() and "cuda" in device
        self.snapshots = []
    
    def reset(self):
        """Reset memory tracking."""
        if self.enabled:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        self.snapshots = []
    
    def snapshot(self, label: str = ""):
        """Take a memory snapshot."""
        if self.enabled:
            torch.cuda.synchronize()
            self.snapshots.append({
                "label": label,
                "allocated": torch.cuda.memory_allocated(),
                "reserved": torch.cuda.memory_reserved(),
                "peak": torch.cuda.max_memory_allocated(),
                "timestamp": time.perf_counter(),
            })
    
    def get_peak(self) -> int:
        """Get peak memory usage in bytes."""
        if self.enabled:
            return torch.cuda.max_memory_allocated()
        return 0
    
    def get_current(self) -> int:
        """Get current memory usage in bytes."""
        if self.enabled:
            return torch.cuda.memory_allocated()
        return 0


@contextmanager
def memory_profile(device: str = "cuda"):
    """Context manager for memory profiling."""
    tracker = MemoryTracker(device)
    tracker.reset()
    tracker.snapshot("start")
    
    try:
        yield tracker
    finally:
        tracker.snapshot("end")


@contextmanager
def timer():
    """Simple timing context manager."""
    start = time.perf_counter()
    result = {"elapsed": 0.0}
    
    try:
        yield result
    finally:
        result["elapsed"] = time.perf_counter() - start


def estimate_kv_cache_memory(
    batch_size: int,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    dtype: torch.dtype = torch.float16,
) -> int:
    """
    Estimate KV cache memory usage in bytes.
    
    KV cache stores:
    - K: (batch, num_heads, seq_len, head_dim) per layer
    - V: (batch, num_heads, seq_len, head_dim) per layer
    """
    # Bytes per element
    if dtype in (torch.float16, torch.bfloat16):
        bytes_per_elem = 2
    elif dtype == torch.float32:
        bytes_per_elem = 4
    else:
        bytes_per_elem = 2  # default
    
    # Total elements
    elements_per_layer = batch_size * num_heads * seq_len * head_dim * 2  # K + V
    total_elements = elements_per_layer * num_layers
    
    return total_elements * bytes_per_elem


def profile_generation(
    model,
    tokenizer,
    input_text: str,
    max_new_tokens: int = 100,
    device: str = "cuda",
    warmup: bool = True,
) -> ProfileResult:
    """
    Profile a generation pass with detailed metrics.
    
    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        input_text: Input prompt
        max_new_tokens: Tokens to generate
        device: Device string
        warmup: Whether to run a warmup pass
    
    Returns:
        ProfileResult with timing and memory metrics
    """
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    input_len = input_ids.shape[1]
    
    # Warmup pass (to JIT compile, etc.)
    if warmup and torch.cuda.is_available():
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids[:, :min(32, input_len)],
                max_new_tokens=5,
                do_sample=False,
            )
        torch.cuda.synchronize()
    
    # Reset memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Profile prefill
    prefill_start = time.perf_counter()
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    prefill_end = time.perf_counter()
    prefill_time = prefill_end - prefill_start
    
    # Get KV cache from first pass
    past_key_values = outputs.past_key_values
    
    # Estimate KV cache memory
    if past_key_values:
        # Get dimensions from first layer
        first_k = past_key_values[0][0]  # (batch, heads, seq, dim)
        batch_size = first_k.shape[0]
        num_heads = first_k.shape[1]
        head_dim = first_k.shape[3]
        num_layers = len(past_key_values)
        
        kv_cache_memory = estimate_kv_cache_memory(
            batch_size=batch_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            seq_len=input_len,
            dtype=first_k.dtype,
        )
    else:
        kv_cache_memory = 0
    
    # Profile decode (token-by-token generation)
    decode_start = time.perf_counter()
    
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            return_dict_in_generate=True,
        )
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    decode_end = time.perf_counter()
    
    # Calculate metrics
    output_len = generated.sequences.shape[1] - input_len
    total_time = decode_end - prefill_start
    decode_time = decode_end - decode_start
    
    # Memory metrics
    peak_memory = 0
    allocated_memory = 0
    reserved_memory = 0
    
    if torch.cuda.is_available():
        peak_memory = torch.cuda.max_memory_allocated()
        allocated_memory = torch.cuda.memory_allocated()
        reserved_memory = torch.cuda.memory_reserved()
    
    return ProfileResult(
        peak_memory_bytes=peak_memory,
        allocated_memory_bytes=allocated_memory,
        reserved_memory_bytes=reserved_memory,
        kv_cache_memory_bytes=kv_cache_memory,
        total_time=total_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        input_tokens=input_len,
        output_tokens=output_len,
        total_tokens=input_len + output_len,
    )


def profile_method(
    method,
    test_prompts: list[str],
    max_new_tokens: int = 100,
    num_runs: int = 3,
) -> dict:
    """
    Profile a method across multiple prompts.
    
    Returns aggregated statistics.
    """
    all_results = []
    
    for prompt in test_prompts:
        for _ in range(num_runs):
            # Tokenize
            inputs = method.tokenizer(prompt, return_tensors="pt")
            
            # Generate
            output = method.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=max_new_tokens,
            )
            
            all_results.append({
                "input_tokens": inputs["input_ids"].shape[1],
                "output_tokens": len(method.tokenizer.encode(output.text[0])),
                "total_time": output.total_time,
                "peak_memory_mb": output.peak_memory_mb,
            })
            
            method.reset_cache()
    
    # Aggregate
    if not all_results:
        return {}
    
    import statistics
    
    return {
        "avg_tokens_per_second": statistics.mean(
            r["output_tokens"] / r["total_time"] for r in all_results if r["total_time"] > 0
        ),
        "avg_peak_memory_mb": statistics.mean(r["peak_memory_mb"] for r in all_results),
        "avg_time_per_token_ms": statistics.mean(
            r["total_time"] / r["output_tokens"] * 1000
            for r in all_results if r["output_tokens"] > 0
        ),
        "num_runs": len(all_results),
    }


def benchmark_throughput(
    method,
    input_lengths: list[int] = None,
    output_length: int = 100,
    num_runs: int = 3,
) -> list[dict]:
    """
    Benchmark throughput across different input lengths.
    
    Returns list of results for each input length.
    """
    if input_lengths is None:
        input_lengths = [128, 256, 512, 1024, 2048, 4096]
    
    results = []
    
    for input_len in input_lengths:
        # Generate dummy input of specified length
        dummy_tokens = list(range(100, 100 + input_len))
        prompt = method.tokenizer.decode(dummy_tokens)
        
        # Truncate to exact length
        inputs = method.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=input_len,
            truncation=True,
        )
        
        actual_len = inputs["input_ids"].shape[1]
        
        times = []
        memories = []
        
        for _ in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            
            output = method.generate(
                input_ids=inputs["input_ids"],
                max_new_tokens=output_length,
            )
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            
            if torch.cuda.is_available():
                memories.append(torch.cuda.max_memory_allocated() / 1024 / 1024)
            
            method.reset_cache()
        
        import statistics
        
        results.append({
            "input_length": actual_len,
            "output_length": output_length,
            "avg_time_s": statistics.mean(times),
            "std_time_s": statistics.stdev(times) if len(times) > 1 else 0,
            "avg_memory_mb": statistics.mean(memories) if memories else 0,
            "tokens_per_second": output_length / statistics.mean(times),
        })
    
    return results

