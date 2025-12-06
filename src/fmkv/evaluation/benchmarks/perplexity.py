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
    
    def _prepare_data(self, tokenizer, method=None) -> list[dict]:
        """Prepare tokenized data for evaluation."""
        # Get model's max sequence length
        # Try to get from method's model config first
        max_seq_len = None
        if method is not None and hasattr(method, 'model') and method.model is not None:
            config = getattr(method.model, 'config', None)
            if config is not None:
                max_seq_len = getattr(config, 'max_position_embeddings', None)
        
        # Fall back to tokenizer's model_max_length
        if max_seq_len is None:
            max_seq_len = getattr(tokenizer, "model_max_length", None)
        
        # If still None or unreasonably large, use a safe default
        if max_seq_len is None or max_seq_len > 1e6:
            # Use self.max_length * 2 as a safe upper bound
            max_seq_len = min(self.max_length * 2, 8192)
        
        # Ensure we don't exceed the model's actual limit
        max_seq_len = min(max_seq_len, 8192)  # Cap at reasonable maximum
        
        # Get texts
        if "text" in self.dataset.column_names:
            texts = self.dataset["text"]
        elif "content" in self.dataset.column_names:
            texts = self.dataset["content"]
        else:
            raise ValueError(f"Dataset has no 'text' or 'content' column")
        
        # Filter empty texts
        texts = [t for t in texts if t and len(t.strip()) > 0]

        # Bug #27 Fix: Don't limit texts here - we need enough tokens for sliding windows
        # The num_samples limit is applied to the sliding window samples, not input texts
        # Limit to a reasonable number of texts to avoid OOM during tokenization
        max_texts = min(len(texts), max(1000, (self.num_samples or 1000) * 10))
        texts = texts[:max_texts]

        self.log(f"Tokenizing {len(texts)} text samples (max_seq_len={max_seq_len})...")
        
        # Tokenize each text separately to avoid exceeding model limits
        # Then concatenate token sequences
        all_token_ids = []
        
        for text in texts:
            # Tokenize with truncation to respect model limits
            encodings = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_seq_len,
                add_special_tokens=True,
            )
            token_ids = encodings.input_ids[0].tolist()
            
            # Add separator tokens between texts (use EOS token if available)
            if all_token_ids:  # Not the first text
                eos_token_id = getattr(tokenizer, "eos_token_id", None)
                if eos_token_id is not None:
                    # Add separator tokens between texts
                    all_token_ids.append(eos_token_id)
                    all_token_ids.append(eos_token_id)
            
            all_token_ids.extend(token_ids)
        
        # Convert to tensor
        input_ids = torch.tensor(all_token_ids, dtype=torch.long)
        total_len = len(input_ids)
        
        self.log(f"Total tokenized length: {total_len:,} tokens")
        
        # Create sliding window samples
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
        
        # Prepare data with method's tokenizer (pass method to get max_length)
        samples = self._prepare_data(method.tokenizer, method=method)
        
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
                import traceback
                error_msg = f"Sample {sample['start_idx']}: {str(e)}"
                errors.append(error_msg)
                # Print first error with full traceback for debugging (use print, not logger)
                if len(errors) == 1:
                    print(f"\n{'='*60}")
                    print(f"ERROR in compute_loss: {error_msg}")
                    print(f"Traceback:\n{traceback.format_exc()}")
                    print(f"{'='*60}\n")
                continue

        # Print error summary if any (use print, not logger)
        if errors:
            print(f"\nWARNING: {len(errors)}/{len(samples)} samples failed!")
            print(f"First error: {errors[0]}")

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

