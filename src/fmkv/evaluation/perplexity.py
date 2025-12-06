"""
Perplexity Evaluation
=====================

Evaluates perplexity under different compression ratios to generate
PPL vs Cache Size curves for comparison with baselines.

The hypothesis: Force-Matched compression achieves a better Pareto frontier
(lower perplexity at higher compression ratios) compared to H2O.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PerplexityResult:
    """Result of perplexity evaluation."""
    
    perplexity: float
    loss: float
    num_tokens: int
    compression_ratio: float
    effective_context_length: int
    original_context_length: int


class PerplexityEvaluator:
    """
    Evaluates model perplexity with different compression strategies.
    
    Example:
        >>> evaluator = PerplexityEvaluator(model_wrapper)
        >>> 
        >>> # Evaluate with compression
        >>> results = evaluator.evaluate(
        ...     dataloader=eval_loader,
        ...     compression_ratios=[0.5, 0.25, 0.1],
        ... )
        >>> 
        >>> for ratio, result in results.items():
        ...     print(f"Compression {ratio}: PPL = {result.perplexity:.2f}")
    """
    
    def __init__(
        self,
        model,  # ModelWrapper
        sidecar: Optional["Sidecar"] = None,
        window_size: int = 64,
    ):
        """
        Initialize evaluator.
        
        Args:
            model: ModelWrapper instance
            sidecar: Optional trained Sidecar for compression
            window_size: Compression window size
        """
        self.model = model
        self.sidecar = sidecar
        self.window_size = window_size
    
    @torch.no_grad()
    def compute_perplexity(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        use_compression: bool = False,
        max_cache_length: Optional[int] = None,
    ) -> Tuple[float, int]:
        """
        Compute perplexity for a sequence.
        
        Args:
            input_ids: Input token IDs (batch, seq_len)
            target_ids: Target token IDs for loss computation
            use_compression: Whether to use KV cache compression
            max_cache_length: Maximum cache length (for compression)
        
        Returns:
            Tuple of (loss, num_tokens)
        """
        device = self.model.info.device
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Forward pass
        outputs = self.model.forward(
            input_ids=input_ids,
            use_cache=True,
        )
        
        logits = outputs.logits
        
        # Compute cross-entropy loss
        # Shift logits and targets for next-token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = target_ids[..., 1:].contiguous()
        
        # Flatten for cross-entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="sum",
            ignore_index=-100,
        )
        
        # Count non-padding tokens
        num_tokens = (shift_labels != -100).sum().item()
        
        return loss.item(), num_tokens
    
    def evaluate(
        self,
        dataloader: DataLoader,
        compression_ratios: List[float] = [1.0, 0.5, 0.25, 0.1],
        max_samples: Optional[int] = None,
        show_progress: bool = True,
    ) -> Dict[float, PerplexityResult]:
        """
        Evaluate perplexity at different compression ratios.
        
        Args:
            dataloader: DataLoader with evaluation data
            compression_ratios: List of compression ratios to test
            max_samples: Maximum number of samples
            show_progress: Show progress bar
        
        Returns:
            Dict mapping compression_ratio -> PerplexityResult
        """
        results = {}
        
        for ratio in compression_ratios:
            # Compute max cache length for this ratio
            # ratio = effective / original
            # e.g., ratio=0.25 means 4x compression
            
            total_loss = 0.0
            total_tokens = 0
            total_original_len = 0
            total_effective_len = 0
            
            iterator = tqdm(
                dataloader,
                desc=f"PPL (ratio={ratio})",
                disable=not show_progress,
            )
            
            for batch_idx, batch in enumerate(iterator):
                if max_samples and batch_idx >= max_samples:
                    break
                
                if isinstance(batch, dict):
                    input_ids = batch["input_ids"]
                    target_ids = batch.get("labels", input_ids)
                else:
                    input_ids = batch
                    target_ids = batch
                
                # Compute loss
                loss, num_tokens = self.compute_perplexity(
                    input_ids=input_ids,
                    target_ids=target_ids,
                    use_compression=(ratio < 1.0),
                )
                
                total_loss += loss
                total_tokens += num_tokens
                
                # Track context lengths
                seq_len = input_ids.size(1)
                total_original_len += seq_len
                total_effective_len += int(seq_len * ratio)
                
                # Update progress
                if total_tokens > 0:
                    current_ppl = math.exp(total_loss / total_tokens)
                    iterator.set_postfix({"ppl": f"{current_ppl:.2f}"})
            
            # Compute final perplexity
            avg_loss = total_loss / max(total_tokens, 1)
            perplexity = math.exp(avg_loss)
            
            results[ratio] = PerplexityResult(
                perplexity=perplexity,
                loss=avg_loss,
                num_tokens=total_tokens,
                compression_ratio=ratio,
                effective_context_length=total_effective_len // max(batch_idx + 1, 1),
                original_context_length=total_original_len // max(batch_idx + 1, 1),
            )
        
        return results
    
    def print_results(self, results: Dict[float, PerplexityResult]):
        """Print formatted perplexity results."""
        logger.info("\n" + "=" * 60)
        logger.info("PERPLEXITY VS COMPRESSION RATIO")
        logger.info("=" * 60)

        logger.info(f"\n{'Ratio':<10} {'Effective':<12} {'PPL':<10} {'Loss':<10}")
        logger.info("-" * 42)

        for ratio in sorted(results.keys(), reverse=True):
            result = results[ratio]
            logger.info(
                f"{ratio:<10.2f} "
                f"{result.effective_context_length:<12d} "
                f"{result.perplexity:<10.2f} "
                f"{result.loss:<10.4f}"
            )

        logger.info("=" * 60)

        # Compute relative degradation
        if 1.0 in results:
            baseline_ppl = results[1.0].perplexity
            logger.info("\nRelative PPL increase from dense baseline:")
            for ratio in sorted(results.keys(), reverse=True):
                if ratio < 1.0:
                    result = results[ratio]
                    increase = (result.perplexity - baseline_ppl) / baseline_ppl * 100
                    compression = 1 / ratio
                    logger.info(f"  {compression:.0f}x compression: +{increase:.1f}%")


def create_ppl_curve_data(
    results: Dict[float, PerplexityResult]
) -> Tuple[List[float], List[float]]:
    """
    Extract data for plotting PPL vs compression ratio.
    
    Returns:
        Tuple of (compression_ratios, perplexities)
    """
    ratios = sorted(results.keys(), reverse=True)
    ppls = [results[r].perplexity for r in ratios]
    return ratios, ppls

