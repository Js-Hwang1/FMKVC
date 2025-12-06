"""
Dense (uncompressed) baseline method.

This serves as the upper bound for quality - no compression is applied.
"""

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseMethod, GenerationOutput, MethodConfig

logger = logging.getLogger(__name__)


class DenseMethod(BaseMethod):
    """
    Dense baseline - standard HuggingFace generation without compression.
    
    This is the gold standard for quality comparison.
    """
    
    @property
    def name(self) -> str:
        return "dense"
    
    @property
    def is_compression_method(self) -> bool:
        return False
    
    def setup(self) -> None:
        """Load model and tokenizer."""
        if self._is_setup:
            return
        
        logger.info(f"Loading model: {self.config.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=self.config.torch_dtype_parsed,
            device_map=self.config.device,
            trust_remote_code=True,
        )
        self.model.eval()
        
        self._is_setup = True
        logger.info(f"Model loaded: {self.model.config.num_hidden_layers} layers")
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 100,
        **kwargs,
    ) -> GenerationOutput:
        """Standard HuggingFace generation."""
        if not self._is_setup:
            self.setup()
        
        input_ids = input_ids.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        original_len = input_ids.shape[1]
        
        # Track memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=kwargs.get("do_sample", False),
                temperature=kwargs.get("temperature", 1.0),
                top_p=kwargs.get("top_p", 1.0),
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Get memory stats
        peak_memory = 0.0
        if torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
        
        # Decode output
        sequences = outputs.sequences
        generated_text = self.tokenizer.batch_decode(
            sequences[:, original_len:],
            skip_special_tokens=True,
        )
        
        return GenerationOutput(
            sequences=sequences,
            text=generated_text,
            total_time=end_time - start_time,
            peak_memory_mb=peak_memory,
            cache_size=sequences.shape[1],
            original_size=original_len,
            compression_ratio=1.0,
        )
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute cross-entropy loss for perplexity."""
        if not self._is_setup:
            self.setup()
        
        input_ids = input_ids.to(self.model.device)
        labels = labels.to(self.model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        
        return outputs.loss
    
    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        return {
            "method": self.name,
            "is_compression": False,
            "compression_ratio": 1.0,
            "cache_budget": None,
        }

