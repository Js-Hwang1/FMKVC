"""
Passkey Retrieval benchmark.

This is a critical benchmark for KV cache compression methods.
A passkey (random number) is hidden in a long context of irrelevant text.
The model must retrieve the passkey from memory.

Success requires:
1. Encoding the passkey into the KV cache
2. Retaining it through any compression
3. Retrieving it accurately when queried

Reference: "Lost in the Middle" (Liu et al., 2023)
"""

import random
from typing import Optional

import torch
from tqdm import tqdm

from ..methods.base import BaseMethod
from .base import BaseBenchmark, BenchmarkResult


class PasskeyRetrievalBenchmark(BaseBenchmark):
    """
    Passkey retrieval benchmark for long-context evaluation.
    
    Tests the model's ability to retrieve a specific piece of information
    (a random passkey) hidden within a long context.
    """
    
    # Templates for passkey insertion
    PASSKEY_TEMPLATE = "The pass key is {passkey}. Remember it. {passkey} is the pass key."
    
    QUERY_TEMPLATE = "What is the pass key? The pass key is"
    
    # Filler text patterns (irrelevant context)
    FILLER_PATTERNS = [
        "The grass is green. The sky is blue. The sun is yellow. Here we go. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor. ",
        "In a world full of possibilities, every moment brings new opportunities for growth. ",
        "The quick brown fox jumps over the lazy dog. This sentence contains all letters. ",
        "Technology advances rapidly, changing how we live, work, and communicate daily. ",
    ]
    
    def __init__(
        self,
        context_lengths: list[int] = None,
        passkey_positions: list[float] = None,
        num_samples_per_config: int = 10,
        passkey_length: int = 5,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize passkey retrieval benchmark.
        
        Args:
            context_lengths: List of total context lengths to test
            passkey_positions: List of relative positions (0.0 to 1.0) for passkey
            num_samples_per_config: Number of samples per (length, position) config
            passkey_length: Number of digits in passkey
            seed: Random seed for reproducibility
        """
        super().__init__(seed=seed, verbose=verbose, **kwargs)
        
        # Default context lengths (tokens)
        self.context_lengths = context_lengths or [1024, 2048, 4096, 8192]
        
        # Default positions: beginning, middle, end
        self.passkey_positions = passkey_positions or [0.1, 0.3, 0.5, 0.7, 0.9]
        
        self.num_samples_per_config = num_samples_per_config
        self.passkey_length = passkey_length
        
        self.test_cases = []
    
    @property
    def name(self) -> str:
        return "passkey_retrieval"
    
    @property
    def description(self) -> str:
        return "Passkey retrieval accuracy across context lengths and positions"
    
    def _generate_passkey(self, rng: random.Random) -> str:
        """Generate a random passkey."""
        return "".join(str(rng.randint(0, 9)) for _ in range(self.passkey_length))
    
    def _generate_filler(self, target_tokens: int, tokenizer, rng: random.Random) -> str:
        """Generate filler text of approximately target_tokens length."""
        filler = ""
        current_tokens = 0
        
        while current_tokens < target_tokens:
            pattern = rng.choice(self.FILLER_PATTERNS)
            filler += pattern
            # Rough estimate: 4 characters per token
            current_tokens = len(filler) // 4
        
        # Trim to exact token count
        tokens = tokenizer.encode(filler, add_special_tokens=False)
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            filler = tokenizer.decode(tokens)
        
        return filler
    
    def _create_test_case(
        self,
        context_length: int,
        passkey_position: float,
        tokenizer,
        rng: random.Random,
    ) -> dict:
        """Create a single test case."""
        passkey = self._generate_passkey(rng)
        
        # Calculate tokens for each section
        passkey_text = self.PASSKEY_TEMPLATE.format(passkey=passkey)
        passkey_tokens = len(tokenizer.encode(passkey_text, add_special_tokens=False))
        
        query_text = self.QUERY_TEMPLATE
        query_tokens = len(tokenizer.encode(query_text, add_special_tokens=False))
        
        # Available tokens for filler
        available_tokens = context_length - passkey_tokens - query_tokens
        
        # Split filler before and after passkey
        passkey_token_pos = int(available_tokens * passkey_position)
        before_tokens = passkey_token_pos
        after_tokens = available_tokens - before_tokens
        
        # Generate filler text
        before_filler = self._generate_filler(before_tokens, tokenizer, rng)
        after_filler = self._generate_filler(after_tokens, tokenizer, rng)
        
        # Construct full context
        full_text = before_filler + " " + passkey_text + " " + after_filler + " " + query_text
        
        # Tokenize
        tokens = tokenizer.encode(full_text, add_special_tokens=True, return_tensors="pt")
        
        return {
            "input_ids": tokens,
            "passkey": passkey,
            "context_length": context_length,
            "passkey_position": passkey_position,
            "actual_length": tokens.shape[1],
        }
    
    def setup(self) -> None:
        """Prepare test cases (requires tokenizer, so done lazily)."""
        self._is_setup = True
        self.log(f"Passkey benchmark configured for {len(self.context_lengths)} lengths, "
                f"{len(self.passkey_positions)} positions")
    
    def _generate_test_cases(self, tokenizer) -> list[dict]:
        """Generate all test cases with the method's tokenizer."""
        rng = random.Random(self.seed)
        test_cases = []
        
        for context_length in self.context_lengths:
            for position in self.passkey_positions:
                for _ in range(self.num_samples_per_config):
                    try:
                        case = self._create_test_case(
                            context_length, position, tokenizer, rng
                        )
                        test_cases.append(case)
                    except Exception as e:
                        self.log(f"Warning: Failed to create case: {e}")
        
        return test_cases
    
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """Evaluate passkey retrieval accuracy."""
        if not self._is_setup:
            self.setup()
        
        # Ensure method is set up
        method.setup()
        
        self.log(f"Evaluating {method.name} on passkey retrieval")
        
        # Generate test cases
        test_cases = self._generate_test_cases(method.tokenizer)
        self.log(f"Generated {len(test_cases)} test cases")
        
        # Run evaluation
        correct = 0
        total = 0
        sample_results = []
        errors = []
        
        # Track accuracy by position and length
        accuracy_by_position = {pos: {"correct": 0, "total": 0} for pos in self.passkey_positions}
        accuracy_by_length = {length: {"correct": 0, "total": 0} for length in self.context_lengths}
        
        progress = tqdm(test_cases, desc=f"Passkey ({method.name})", disable=not self.verbose)
        
        for case in progress:
            try:
                # Generate response
                output = method.generate(
                    input_ids=case["input_ids"],
                    max_new_tokens=self.passkey_length + 10,
                    do_sample=False,
                )
                
                generated_text = output.text[0].strip()
                
                # Extract predicted passkey (first N digits)
                predicted = "".join(c for c in generated_text if c.isdigit())[:self.passkey_length]
                
                # Check correctness
                is_correct = predicted == case["passkey"]
                
                if is_correct:
                    correct += 1
                    accuracy_by_position[case["passkey_position"]]["correct"] += 1
                    accuracy_by_length[case["context_length"]]["correct"] += 1
                
                total += 1
                accuracy_by_position[case["passkey_position"]]["total"] += 1
                accuracy_by_length[case["context_length"]]["total"] += 1
                
                sample_results.append({
                    "context_length": case["context_length"],
                    "passkey_position": case["passkey_position"],
                    "passkey": case["passkey"],
                    "predicted": predicted,
                    "generated_text": generated_text[:50],
                    "is_correct": is_correct,
                    "cache_stats": method.get_cache_stats(),
                })
                
                # Update progress
                accuracy = correct / total if total > 0 else 0
                progress.set_postfix({"acc": f"{accuracy:.2%}"})
                
                method.reset_cache()
                
            except Exception as e:
                errors.append(f"Case {total}: {str(e)}")
                total += 1
        
        # Compute final metrics
        overall_accuracy = correct / total if total > 0 else 0
        
        metrics = {
            "accuracy": overall_accuracy,
            "correct": correct,
            "total": total,
        }
        
        # Add per-position accuracy
        for pos, counts in accuracy_by_position.items():
            if counts["total"] > 0:
                metrics[f"accuracy_pos_{pos}"] = counts["correct"] / counts["total"]
        
        # Add per-length accuracy
        for length, counts in accuracy_by_length.items():
            if counts["total"] > 0:
                metrics[f"accuracy_len_{length}"] = counts["correct"] / counts["total"]
        
        self.log(f"Final accuracy: {overall_accuracy:.2%} ({correct}/{total})")
        
        return BenchmarkResult(
            benchmark_name=self.name,
            method_name=method.name,
            metrics=metrics,
            sample_results=sample_results,
            config={
                "context_lengths": self.context_lengths,
                "passkey_positions": self.passkey_positions,
                "num_samples_per_config": self.num_samples_per_config,
                "passkey_length": self.passkey_length,
            },
            errors=errors,
            num_samples=len(test_cases),
            num_successful=total - len(errors),
        )

