"""
Passkey Retrieval Benchmark
===========================

Tests the model's ability to find a random number hidden in long context.

This is a critical benchmark for KV cache compression because:
1. Random numbers have low initial attention scores (bad for H2O)
2. The passkey location is unpredictable
3. Requires preserving the "gradient potential" of rare tokens

Example:
    Context: "There is important information hidden in the text below.
              [... thousands of tokens of filler text ...]
              The passkey is: 847392
              [... more filler text ...]
              What is the passkey?"

    Expected answer: "847392"
"""

import logging
import random
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class PasskeyResult:
    """Result of a single passkey retrieval attempt."""
    
    target_passkey: str
    predicted_passkey: str
    is_correct: bool
    context_length: int
    passkey_position: int  # Position as fraction of context (0.0 to 1.0)
    generation: str  # Full model generation


@dataclass
class PasskeyBenchmarkResult:
    """Aggregated results from passkey benchmark."""
    
    accuracy: float
    num_correct: int
    num_total: int
    results_by_length: Dict[int, float]  # context_length -> accuracy
    results_by_position: Dict[str, float]  # position_bucket -> accuracy
    individual_results: List[PasskeyResult]


class PasskeyRetrievalBenchmark:
    """
    Benchmark for testing passkey retrieval in long contexts.
    
    Generates test cases with random passkeys hidden at various positions
    within filler text, then evaluates retrieval accuracy.
    
    Example:
        >>> from fmkv.models import ModelWrapper
        >>> 
        >>> wrapper = ModelWrapper.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> benchmark = PasskeyRetrievalBenchmark()
        >>> 
        >>> results = benchmark.evaluate(
        ...     model=wrapper,
        ...     context_lengths=[4096, 8192, 16384],
        ...     num_samples_per_length=50,
        ... )
        >>> print(f"Accuracy: {results.accuracy:.2%}")
    """
    
    # Filler text templates (repeated to create long context)
    FILLER_TEMPLATES = [
        "The grass is green. The sky is blue. The sun is yellow. ",
        "Here is a random sentence that does not contain any important information. ",
        "This is filler text that serves no purpose other than to extend the context length. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor. ",
        "The quick brown fox jumps over the lazy dog multiple times throughout the day. ",
    ]
    
    # Prompt templates
    PASSKEY_TEMPLATE = "The passkey is: {passkey}. Remember this number.\n\n"
    
    QUESTION_TEMPLATE = "\n\nBased on the context above, what is the passkey? The passkey is:"
    
    INTRO_TEMPLATE = "There is an important passkey hidden somewhere in the following text. " \
                    "Read carefully and remember the passkey when you find it.\n\n"
    
    def __init__(
        self,
        passkey_length: int = 6,
        seed: int = 42,
    ):
        """
        Initialize benchmark.
        
        Args:
            passkey_length: Number of digits in passkey
            seed: Random seed for reproducibility
        """
        self.passkey_length = passkey_length
        self.rng = random.Random(seed)
    
    def generate_passkey(self) -> str:
        """Generate a random passkey."""
        return "".join(str(self.rng.randint(0, 9)) for _ in range(self.passkey_length))
    
    def generate_filler(self, num_tokens: int, tokenizer) -> str:
        """Generate filler text of approximately num_tokens length."""
        filler = ""
        
        while True:
            template = self.rng.choice(self.FILLER_TEMPLATES)
            filler += template
            
            # Check approximate token count
            tokens = tokenizer.encode(filler, add_special_tokens=False)
            if len(tokens) >= num_tokens:
                break
        
        return filler
    
    def create_test_case(
        self,
        target_length: int,
        passkey_position: float,  # 0.0 to 1.0
        tokenizer,
    ) -> Tuple[str, str, int]:
        """
        Create a single test case.
        
        Args:
            target_length: Target context length in tokens
            passkey_position: Relative position of passkey (0.0 = start, 1.0 = end)
            tokenizer: Tokenizer for length estimation
        
        Returns:
            Tuple of (full_prompt, passkey, actual_position_tokens)
        """
        passkey = self.generate_passkey()
        passkey_text = self.PASSKEY_TEMPLATE.format(passkey=passkey)
        
        # Calculate filler lengths
        intro_tokens = len(tokenizer.encode(self.INTRO_TEMPLATE, add_special_tokens=False))
        passkey_tokens = len(tokenizer.encode(passkey_text, add_special_tokens=False))
        question_tokens = len(tokenizer.encode(self.QUESTION_TEMPLATE, add_special_tokens=False))
        
        available_tokens = target_length - intro_tokens - passkey_tokens - question_tokens
        
        # Split filler before and after passkey
        tokens_before = int(available_tokens * passkey_position)
        tokens_after = available_tokens - tokens_before
        
        filler_before = self.generate_filler(tokens_before, tokenizer) if tokens_before > 0 else ""
        filler_after = self.generate_filler(tokens_after, tokenizer) if tokens_after > 0 else ""
        
        # Construct prompt
        prompt = (
            self.INTRO_TEMPLATE +
            filler_before +
            passkey_text +
            filler_after +
            self.QUESTION_TEMPLATE
        )
        
        # Calculate actual passkey position
        text_before_passkey = self.INTRO_TEMPLATE + filler_before
        actual_position = len(tokenizer.encode(text_before_passkey, add_special_tokens=False))
        
        return prompt, passkey, actual_position
    
    def extract_passkey_from_generation(self, generation: str) -> str:
        """Extract passkey from model generation."""
        # Try to find a sequence of digits
        numbers = re.findall(r"\d+", generation)
        
        if not numbers:
            return ""
        
        # Return the first number that matches passkey length
        for num in numbers:
            if len(num) == self.passkey_length:
                return num
        
        # Fall back to first number
        return numbers[0]
    
    @torch.no_grad()
    def evaluate_single(
        self,
        model,  # ModelWrapper
        prompt: str,
        passkey: str,
        passkey_position: int,
        context_length: int,
        max_new_tokens: int = 20,
    ) -> PasskeyResult:
        """Evaluate a single test case."""
        # Tokenize
        inputs = model.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=context_length,
        )
        input_ids = inputs["input_ids"].to(model.info.device)
        
        # Generate
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy for reproducibility
            pad_token_id=model.tokenizer.pad_token_id,
        )
        
        # Decode generation (only new tokens)
        generation = model.tokenizer.decode(
            output_ids[0, input_ids.size(1):],
            skip_special_tokens=True,
        )
        
        # Extract predicted passkey
        predicted = self.extract_passkey_from_generation(generation)
        
        return PasskeyResult(
            target_passkey=passkey,
            predicted_passkey=predicted,
            is_correct=(predicted == passkey),
            context_length=input_ids.size(1),
            passkey_position=passkey_position / input_ids.size(1),
            generation=generation,
        )
    
    def evaluate(
        self,
        model,  # ModelWrapper
        context_lengths: List[int] = [4096, 8192, 16384, 32768],
        num_samples_per_length: int = 50,
        position_buckets: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9],
        show_progress: bool = True,
    ) -> PasskeyBenchmarkResult:
        """
        Run full benchmark.
        
        Args:
            model: ModelWrapper instance
            context_lengths: List of context lengths to test
            num_samples_per_length: Number of samples per length
            position_buckets: Passkey positions to test (as fraction of context)
            show_progress: Show progress bar
        
        Returns:
            PasskeyBenchmarkResult with aggregated metrics
        """
        all_results = []
        results_by_length = {}
        results_by_position = {f"{p:.2f}": [] for p in position_buckets}
        
        total_samples = len(context_lengths) * num_samples_per_length
        
        iterator = tqdm(
            total=total_samples,
            desc="Passkey Retrieval",
            disable=not show_progress,
        )
        
        for ctx_len in context_lengths:
            length_results = []
            
            samples_per_position = num_samples_per_length // len(position_buckets)
            
            for position in position_buckets:
                for _ in range(samples_per_position):
                    # Create test case
                    prompt, passkey, actual_pos = self.create_test_case(
                        target_length=ctx_len,
                        passkey_position=position,
                        tokenizer=model.tokenizer,
                    )
                    
                    # Evaluate
                    result = self.evaluate_single(
                        model=model,
                        prompt=prompt,
                        passkey=passkey,
                        passkey_position=actual_pos,
                        context_length=ctx_len,
                    )
                    
                    all_results.append(result)
                    length_results.append(result)
                    results_by_position[f"{position:.2f}"].append(result)
                    
                    iterator.update(1)
                    iterator.set_postfix({
                        "len": ctx_len,
                        "acc": f"{sum(r.is_correct for r in all_results) / len(all_results):.2%}",
                    })
            
            # Compute accuracy for this length
            if length_results:
                results_by_length[ctx_len] = (
                    sum(r.is_correct for r in length_results) / len(length_results)
                )
        
        iterator.close()
        
        # Compute position accuracies
        position_accuracies = {}
        for pos_key, results in results_by_position.items():
            if results:
                position_accuracies[pos_key] = (
                    sum(r.is_correct for r in results) / len(results)
                )
        
        # Overall accuracy
        num_correct = sum(r.is_correct for r in all_results)
        accuracy = num_correct / len(all_results) if all_results else 0.0
        
        return PasskeyBenchmarkResult(
            accuracy=accuracy,
            num_correct=num_correct,
            num_total=len(all_results),
            results_by_length=results_by_length,
            results_by_position=position_accuracies,
            individual_results=all_results,
        )
    
    def print_results(self, results: PasskeyBenchmarkResult):
        """Print formatted benchmark results."""
        logger.info("\n" + "=" * 60)
        logger.info("PASSKEY RETRIEVAL BENCHMARK RESULTS")
        logger.info("=" * 60)

        logger.info(f"\nOverall Accuracy: {results.accuracy:.2%}")
        logger.info(f"Correct: {results.num_correct}/{results.num_total}")

        logger.info("\nAccuracy by Context Length:")
        for length, acc in sorted(results.results_by_length.items()):
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            logger.info(f"  {length:6d} tokens: {bar} {acc:.2%}")

        logger.info("\nAccuracy by Passkey Position:")
        for pos, acc in sorted(results.results_by_position.items()):
            bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
            logger.info(f"  Position {pos}: {bar} {acc:.2%}")

        logger.info("=" * 60)

