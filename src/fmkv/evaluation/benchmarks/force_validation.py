"""
Force Matching Validation Benchmark
====================================

This benchmark explicitly validates that the compressed cache preserves
gradient forces as claimed by the theoretical framework.

Instead of just measuring perplexity, this directly computes the
Jacobian difference between dense and compressed attention to verify
that force matching was successful.

This addresses Bug #10: Need for explicit force-matching validation.
"""

import torch
from typing import Optional, Dict
from tqdm import tqdm

from ..methods.base import BaseMethod
from .base import BaseBenchmark, BenchmarkResult
from fmkv.losses.jacobian import compute_attention_jacobian_batched


class ForceValidationBenchmark(BaseBenchmark):
    """
    Validate that compressed cache preserves gradient forces.
    
    This directly measures the force-matching objective that the
    Sidecar was trained to minimize:
    
    L_FM = || ∂Attn(q, K_dense, V_dense)/∂q - ∂Attn(q, K_cg, V_cg)/∂q ||_F^2
    
    Lower error means better preservation of gradient dynamics.
    """
    
    def __init__(
        self,
        dataset_name: str = "wikitext-2",
        split: str = "validation",
        num_samples: int = 100,
        window_size: int = 64,
        num_future_queries: int = 16,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(num_samples=num_samples, seed=seed, verbose=verbose, **kwargs)
        
        self.dataset_name = dataset_name
        self.split = split
        self.window_size = window_size
        self.num_future_queries = num_future_queries
        
        self.dataset = None
    
    @property
    def name(self) -> str:
        return f"force_validation_{self.dataset_name}"
    
    @property
    def description(self) -> str:
        return "Validate force-matching: Jacobian error between dense and compressed"
    
    def setup(self) -> None:
        """Load dataset."""
        if self._is_setup:
            return
        
        from datasets import load_dataset
        
        self.log(f"Loading dataset: {self.dataset_name}")
        
        if self.dataset_name == "wikitext-2":
            self.dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=self.split,
                trust_remote_code=True,
            )
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")
        
        self.log(f"Loaded {len(self.dataset)} examples")
        self._is_setup = True
    
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """
        Evaluate force-matching error.
        
        Args:
            method: Must be an FMKV method with Sidecar
        
        Returns:
            BenchmarkResult with force-matching metrics
        """
        if not self._is_setup:
            self.setup()
        
        method.setup()
        
        # Check if method has Sidecar
        if not hasattr(method, 'sidecar') or method.sidecar is None:
            self.log("Method does not have Sidecar - skipping force validation")
            return BenchmarkResult(
                benchmark_name=self.name,
                method_name=method.name,
                metrics={"error": "no_sidecar"},
                sample_results=[],
                config={},
                errors=["Method does not have Sidecar"],
                num_samples=0,
                num_successful=0,
            )
        
        self.log(f"Evaluating force preservation for {method.name}")
        
        # Get model dimensions
        hidden_size = method.model.config.hidden_size
        num_heads = method.model.config.num_attention_heads
        head_dim = hidden_size // num_heads
        scale = head_dim ** -0.5
        
        # Collect metrics
        total_frobenius_error = 0.0
        total_relative_error = 0.0
        total_samples = 0
        sample_results = []
        errors = []
        
        # Prepare text samples
        texts = [t for t in self.dataset["text"] if t and len(t.strip()) > 0]
        if self.num_samples:
            texts = texts[:self.num_samples]
        
        progress = tqdm(texts, desc="Force validation", disable=not self.verbose)
        
        for sample_idx, text in enumerate(progress):
            try:
                # Tokenize
                encodings = method.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.window_size + self.num_future_queries + 10,
                )
                input_ids = encodings.input_ids[0]
                
                if len(input_ids) < self.window_size + self.num_future_queries:
                    continue
                
                # Extract window and future queries from actual sequence
                window_ids = input_ids[:self.window_size]
                future_ids = input_ids[self.window_size:self.window_size + self.num_future_queries]
                
                # Get KV states from model for window
                with torch.no_grad():
                    window_ids_batch = window_ids.unsqueeze(0).to(method.model.device)
                    outputs = method.model(
                        input_ids=window_ids_batch,
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    
                    # Extract KV from first layer (as proxy)
                    # Shape: (batch, num_heads, seq_len, head_dim)
                    past_kv = outputs.past_key_values[0]
                    dense_keys = past_kv[0][0, 0, :, :]  # First head: (window_size, head_dim)
                    dense_values = past_kv[1][0, 0, :, :]
                    
                    # Get future query vectors from hidden states
                    future_ids_batch = future_ids.unsqueeze(0).to(method.model.device)
                    future_outputs = method.model(
                        input_ids=future_ids_batch,
                        output_hidden_states=True,
                    )
                    # Use last hidden state as queries
                    # Shape: (batch, seq_len, hidden_size)
                    future_hidden = future_outputs.hidden_states[-1][0]  # (num_queries, hidden_size)
                    # Project to head dimension (simple: take first head_dim)
                    queries = future_hidden[:, :head_dim]  # (num_queries, head_dim)
                    
                    # Compress using Sidecar
                    # Add batch dimension
                    k_batch = dense_keys.unsqueeze(0)  # (1, window_size, head_dim)
                    v_batch = dense_values.unsqueeze(0)
                    
                    k_cg, v_cg = method.sidecar.compress_cache(k_batch, v_batch)
                    k_cg = k_cg[0]  # Remove batch: (head_dim,)
                    v_cg = v_cg[0]
                    
                    # Compute Jacobians
                    # Dense: (num_queries, head_dim, head_dim)
                    queries_batch = queries.unsqueeze(0)  # (1, num_queries, head_dim)
                    k_dense_batch = dense_keys.unsqueeze(0)
                    v_dense_batch = dense_values.unsqueeze(0)
                    
                    jacobian_dense = compute_attention_jacobian_batched(
                        queries_batch, k_dense_batch, v_dense_batch, scale
                    )[0]  # Remove batch: (num_queries, head_dim, head_dim)
                    
                    # Compressed: need to expand to sequence dimension
                    k_cg_batch = k_cg.unsqueeze(0).unsqueeze(0)  # (1, 1, head_dim)
                    v_cg_batch = v_cg.unsqueeze(0).unsqueeze(0)
                    
                    jacobian_cg = compute_attention_jacobian_batched(
                        queries_batch, k_cg_batch, v_cg_batch, scale
                    )[0]  # (num_queries, head_dim, head_dim)
                    
                    # Compute errors
                    # Aggregate over queries (as done in training)
                    jac_dense_agg = jacobian_dense.sum(dim=0)  # (head_dim, head_dim)
                    jac_cg_agg = jacobian_cg.sum(dim=0)
                    
                    # Frobenius norm of difference
                    diff = jac_dense_agg - jac_cg_agg
                    frobenius_error = (diff ** 2).sum().sqrt().item()
                    
                    # Relative error
                    dense_norm = (jac_dense_agg ** 2).sum().sqrt().item()
                    relative_error = frobenius_error / (dense_norm + 1e-8)
                    
                    total_frobenius_error += frobenius_error
                    total_relative_error += relative_error
                    total_samples += 1
                    
                    sample_results.append({
                        "sample_idx": sample_idx,
                        "frobenius_error": frobenius_error,
                        "relative_error": relative_error,
                        "dense_jacobian_norm": dense_norm,
                    })
                    
                    # Update progress
                    avg_relative = total_relative_error / total_samples
                    progress.set_postfix({"rel_err": f"{avg_relative:.4f}"})
            
            except Exception as e:
                errors.append(f"Sample {sample_idx}: {str(e)}")
                continue
        
        # Compute final metrics
        if total_samples > 0:
            avg_frobenius = total_frobenius_error / total_samples
            avg_relative = total_relative_error / total_samples
        else:
            avg_frobenius = float("inf")
            avg_relative = float("inf")
        
        self.log(f"Average relative force error: {avg_relative:.6f}")
        
        return BenchmarkResult(
            benchmark_name=self.name,
            method_name=method.name,
            metrics={
                "avg_frobenius_error": avg_frobenius,
                "avg_relative_error": avg_relative,
                "total_samples": total_samples,
            },
            sample_results=sample_results,
            config={
                "dataset": self.dataset_name,
                "window_size": self.window_size,
                "num_future_queries": self.num_future_queries,
            },
            errors=errors,
            num_samples=len(texts),
            num_successful=total_samples,
        )
