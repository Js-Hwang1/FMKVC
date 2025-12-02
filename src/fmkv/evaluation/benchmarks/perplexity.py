"""
Perplexity benchmark for language modeling quality.

Evaluates on standard datasets:
- WikiText-2: Small, commonly used
- WikiText-103: Larger, more comprehensive
- C4: Web text, more diverse

Lower perplexity = better language modeling.
"""

import math
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..methods.base import BaseMethod
from .base import BaseBenchmark, BenchmarkResult


class PerplexityBenchmark(BaseBenchmark):
    """
    Perplexity evaluation on standard language modeling datasets.
    
    This is the primary metric for evaluating compression quality.
    A good compression method should maintain low perplexity.
    """
    
    SUPPORTED_DATASETS = {
        "wikitext-2": ("wikitext", "wikitext-2-raw-v1"),
        "wikitext-103": ("wikitext", "wikitext-103-raw-v1"),
        "c4": ("allenai/c4", "en"),
        "pg19": ("pg19", None),
    }
    
    def __init__(
        self,
        dataset_name: str = "wikitext-2",
        split: str = "test",
        max_length: int = 2048,
        stride: int = 512,
        batch_size: int = 1,
        num_samples: Optional[int] = None,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(num_samples=num_samples, seed=seed, verbose=verbose, **kwargs)
        
        if dataset_name not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Supported: {list(self.SUPPORTED_DATASETS.keys())}"
            )
        
        self.dataset_name = dataset_name
        self.split = split
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        
        self.dataset = None
        self.encodings = None
    
    @property
    def name(self) -> str:
        return f"perplexity_{self.dataset_name}"
    
    @property
    def description(self) -> str:
        return f"Perplexity evaluation on {self.dataset_name}"
    
    def setup(self) -> None:
        """Load and prepare dataset."""
        if self._is_setup:
            return
        
        self.log(f"Loading dataset: {self.dataset_name}")
        
        dataset_path, dataset_config = self.SUPPORTED_DATASETS[self.dataset_name]
        
        if dataset_config:
            self.dataset = load_dataset(
                dataset_path,
                dataset_config,
                split=self.split,
                trust_remote_code=True,
            )
        else:
            self.dataset = load_dataset(
                dataset_path,
                split=self.split,
                trust_remote_code=True,
            )
        
        self.log(f"Loaded {len(self.dataset)} examples")
        self._is_setup = True
    
    def _prepare_data(self, tokenizer) -> list[dict]:
        """Prepare tokenized data for evaluation."""
        # Concatenate all text
        if "text" in self.dataset.column_names:
            texts = self.dataset["text"]
        elif "content" in self.dataset.column_names:
            texts = self.dataset["content"]
        else:
            raise ValueError(f"Dataset has no 'text' or 'content' column")
        
        # Filter empty texts
        texts = [t for t in texts if t and len(t.strip()) > 0]
        
        if self.num_samples:
            texts = texts[:self.num_samples]
        
        # Concatenate texts
        full_text = "\n\n".join(texts)
        
        self.log(f"Tokenizing {len(full_text):,} characters...")
        
        # Tokenize
        encodings = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=False,
        )
        
        # Create sliding window samples
        input_ids = encodings.input_ids[0]
        total_len = len(input_ids)
        
        samples = []
        for start_idx in range(0, total_len - self.max_length, self.stride):
            end_idx = start_idx + self.max_length
            
            sample = {
                "input_ids": input_ids[start_idx:end_idx].unsqueeze(0),
                "start_idx": start_idx,
            }
            samples.append(sample)
            
            # Limit samples if specified
            if self.num_samples and len(samples) >= self.num_samples:
                break
        
        self.log(f"Created {len(samples)} evaluation windows")
        return samples
    
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """Evaluate perplexity using the given method."""
        if not self._is_setup:
            self.setup()
        
        # Ensure method is set up
        method.setup()
        
        self.log(f"Evaluating {method.name} on {self.dataset_name}")
        
        # Prepare data with method's tokenizer
        samples = self._prepare_data(method.tokenizer)
        
        # Compute perplexity
        total_loss = 0.0
        total_tokens = 0
        sample_results = []
        errors = []
        
        progress = tqdm(samples, desc=f"PPL ({method.name})", disable=not self.verbose)
        
        for sample in progress:
            try:
                input_ids = sample["input_ids"]
                
                # Labels are the same as inputs (shifted internally by model)
                labels = input_ids.clone()
                
                # Compute loss
                loss = method.compute_loss(
                    input_ids=input_ids,
                    labels=labels,
                )
                
                # Accumulate
                seq_len = input_ids.shape[1]
                total_loss += loss.item() * seq_len
                total_tokens += seq_len
                
                sample_results.append({
                    "start_idx": sample["start_idx"],
                    "loss": loss.item(),
                    "perplexity": math.exp(loss.item()),
                })
                
                # Update progress
                current_ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
                progress.set_postfix({"ppl": f"{current_ppl:.2f}"})
                
            except Exception as e:
                errors.append(f"Sample {sample['start_idx']}: {str(e)}")
                continue
        
        # Compute final metrics
        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = math.exp(avg_loss)
        else:
            avg_loss = float("inf")
            perplexity = float("inf")
        
        self.log(f"Final perplexity: {perplexity:.4f}")
        
        return BenchmarkResult(
            benchmark_name=self.name,
            method_name=method.name,
            metrics={
                "perplexity": perplexity,
                "loss": avg_loss,
                "total_tokens": total_tokens,
            },
            sample_results=sample_results,
            config={
                "dataset": self.dataset_name,
                "split": self.split,
                "max_length": self.max_length,
                "stride": self.stride,
            },
            errors=errors,
            num_samples=len(samples),
            num_successful=len(sample_results),
        )

