"""
Needle-in-a-Haystack benchmark.

Similar to passkey retrieval but uses more naturalistic "needles" (facts)
embedded in essay-style "haystack" text.

This tests:
1. Single fact retrieval across varying context depths
2. Multi-needle retrieval (optional)
3. Generates visual heatmaps of retrieval accuracy

Reference: Greg Kamradt's Needle in a Haystack test
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""

import random
from typing import Optional

import torch
from tqdm import tqdm

from ..methods.base import BaseMethod
from .base import BaseBenchmark, BenchmarkResult


class NeedleInHaystackBenchmark(BaseBenchmark):
    """
    Needle-in-a-Haystack benchmark for long-context evaluation.
    
    Tests the model's ability to find specific facts embedded at
    various depths within long contexts.
    """
    
    # Example needles (facts to retrieve)
    NEEDLES = [
        {
            "needle": "The best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.",
            "question": "What is the best thing to do in San Francisco?",
            "answer": "eat a sandwich and sit in Dolores Park",
        },
        {
            "needle": "The secret ingredient to the best pizza in the world is truffle oil drizzled generously.",
            "question": "What is the secret ingredient to the best pizza?",
            "answer": "truffle oil",
        },
        {
            "needle": "The magic word to open the treasure chest is 'alakazam'.",
            "question": "What is the magic word to open the treasure chest?",
            "answer": "alakazam",
        },
        {
            "needle": "The winning lottery numbers for this week are 7, 14, 21, 35, 42.",
            "question": "What are the winning lottery numbers?",
            "answer": "7, 14, 21, 35, 42",
        },
        {
            "needle": "The password to access the secret lab is 'quantum_leap_2024'.",
            "question": "What is the password to access the secret lab?",
            "answer": "quantum_leap_2024",
        },
    ]
    
    # Haystack text (Paul Graham essays style filler)
    HAYSTACK_TEMPLATE = """
The concept of innovation is often misunderstood. People think it requires a flash of 
genius, but really it's about persistence and iteration. The most successful companies 
weren't built on a single brilliant idea but on thousands of small improvements.

When we look at history, we see that major breakthroughs often came from unexpected 
directions. The printing press, the steam engine, the computer - none of these were 
predicted by the experts of their time. This suggests that the future will similarly 
surprise us.

What makes a good startup? It's not just about the idea. Execution matters far more 
than most people realize. A mediocre idea brilliantly executed will almost always 
beat a brilliant idea poorly executed.

The importance of teams cannot be overstated. Even the most talented individual 
cannot match the output of a well-coordinated team. This is why company culture 
matters so much - it's the operating system that enables team coordination.

Technology changes rapidly, but human nature changes slowly. This is why the best 
companies focus on solving fundamental human needs rather than chasing the latest 
trends. The tools may change, but the problems remain remarkably consistent.
"""
    
    def __init__(
        self,
        context_lengths: list[int] = None,
        depth_percents: list[float] = None,
        num_samples_per_config: int = 3,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize needle-in-a-haystack benchmark.
        
        Args:
            context_lengths: List of total context lengths to test (in tokens)
            depth_percents: List of depth percentages (0-100) for needle placement
            num_samples_per_config: Samples per (length, depth) configuration
            seed: Random seed
        """
        super().__init__(seed=seed, verbose=verbose, **kwargs)
        
        # Default context lengths
        self.context_lengths = context_lengths or [1024, 2048, 4096, 8192, 16384]
        
        # Default depths: 0% = beginning, 100% = end
        self.depth_percents = depth_percents or [0, 25, 50, 75, 100]
        
        self.num_samples_per_config = num_samples_per_config
    
    @property
    def name(self) -> str:
        return "needle_in_haystack"
    
    @property
    def description(self) -> str:
        return "Single-needle retrieval accuracy across depths and lengths"
    
    def _generate_haystack(self, target_tokens: int, tokenizer, rng: random.Random) -> str:
        """Generate haystack text of approximately target_tokens length."""
        # Shuffle paragraphs for variety
        paragraphs = self.HAYSTACK_TEMPLATE.strip().split("\n\n")
        
        haystack = ""
        current_tokens = 0
        
        while current_tokens < target_tokens:
            rng.shuffle(paragraphs)
            for para in paragraphs:
                haystack += para.strip() + "\n\n"
                current_tokens = len(tokenizer.encode(haystack, add_special_tokens=False))
                if current_tokens >= target_tokens:
                    break
        
        # Trim to approximate target
        tokens = tokenizer.encode(haystack, add_special_tokens=False)
        if len(tokens) > target_tokens:
            tokens = tokens[:target_tokens]
            haystack = tokenizer.decode(tokens)
        
        return haystack
    
    def _create_test_case(
        self,
        context_length: int,
        depth_percent: float,
        tokenizer,
        rng: random.Random,
    ) -> dict:
        """Create a single test case."""
        # Select random needle
        needle_data = rng.choice(self.NEEDLES)
        needle = needle_data["needle"]
        question = needle_data["question"]
        answer = needle_data["answer"]
        
        # Calculate needle position
        needle_tokens = len(tokenizer.encode(needle, add_special_tokens=False))
        question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        
        # Available space for haystack
        haystack_space = context_length - needle_tokens - question_tokens - 50  # buffer
        
        # Generate haystack
        full_haystack = self._generate_haystack(haystack_space, tokenizer, rng)
        
        # Split haystack at depth position
        haystack_tokens = tokenizer.encode(full_haystack, add_special_tokens=False)
        depth_position = int(len(haystack_tokens) * (depth_percent / 100))
        
        before_tokens = haystack_tokens[:depth_position]
        after_tokens = haystack_tokens[depth_position:]
        
        before_text = tokenizer.decode(before_tokens)
        after_text = tokenizer.decode(after_tokens)
        
        # Construct full context
        full_context = before_text + "\n\n" + needle + "\n\n" + after_text
        
        # Add question at the end
        prompt = full_context + f"\n\nQuestion: {question}\nAnswer:"
        
        # Tokenize
        input_ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        
        return {
            "input_ids": input_ids,
            "needle": needle,
            "question": question,
            "expected_answer": answer,
            "context_length": context_length,
            "depth_percent": depth_percent,
            "actual_length": input_ids.shape[1],
        }
    
    def setup(self) -> None:
        """Prepare benchmark configuration."""
        self._is_setup = True
        total_configs = len(self.context_lengths) * len(self.depth_percents) * self.num_samples_per_config
        self.log(f"Needle-in-Haystack configured for {total_configs} test cases")
    
    def _check_answer(self, generated: str, expected: str) -> tuple[bool, float]:
        """
        Check if the generated answer contains the expected answer.
        
        Returns:
            (is_correct, score) where score is a partial match score
        """
        generated_lower = generated.lower().strip()
        expected_lower = expected.lower().strip()
        
        # Exact match
        if expected_lower in generated_lower:
            return True, 1.0
        
        # Partial match: check if key words are present
        expected_words = set(expected_lower.split())
        generated_words = set(generated_lower.split())
        
        if not expected_words:
            return False, 0.0
        
        overlap = expected_words & generated_words
        score = len(overlap) / len(expected_words)
        
        # Consider correct if >70% word overlap
        is_correct = score > 0.7
        
        return is_correct, score
    
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """Evaluate needle-in-haystack retrieval."""
        if not self._is_setup:
            self.setup()
        
        # Ensure method is set up
        method.setup()
        
        self.log(f"Evaluating {method.name} on needle-in-haystack")
        
        rng = random.Random(self.seed)
        
        # Results storage
        results_matrix = {}  # (length, depth) -> list of scores
        sample_results = []
        errors = []
        
        total_correct = 0
        total_cases = 0
        
        # Generate and evaluate test cases
        for context_length in self.context_lengths:
            for depth_percent in self.depth_percents:
                key = (context_length, depth_percent)
                results_matrix[key] = []
                
                for _ in range(self.num_samples_per_config):
                    try:
                        # Create test case
                        case = self._create_test_case(
                            context_length, depth_percent, method.tokenizer, rng
                        )
                        
                        # Check if context fits in model's max length
                        max_model_length = getattr(
                            method.model.config, "max_position_embeddings", 32768
                        )
                        
                        if case["actual_length"] > max_model_length:
                            self.log(f"Skipping case: length {case['actual_length']} > max {max_model_length}")
                            continue
                        
                        # Generate response
                        output = method.generate(
                            input_ids=case["input_ids"],
                            max_new_tokens=100,
                            do_sample=False,
                        )
                        
                        generated_text = output.text[0].strip()
                        
                        # Check answer
                        is_correct, score = self._check_answer(
                            generated_text, case["expected_answer"]
                        )
                        
                        results_matrix[key].append(score)
                        
                        if is_correct:
                            total_correct += 1
                        
                        total_cases += 1
                        
                        sample_results.append({
                            "context_length": context_length,
                            "depth_percent": depth_percent,
                            "needle": case["needle"],
                            "question": case["question"],
                            "expected": case["expected_answer"],
                            "generated": generated_text[:200],
                            "is_correct": is_correct,
                            "score": score,
                        })
                        
                        method.reset_cache()
                        
                    except Exception as e:
                        errors.append(f"Case ({context_length}, {depth_percent}): {str(e)}")
                
                # Log progress
                if results_matrix[key]:
                    avg_score = sum(results_matrix[key]) / len(results_matrix[key])
                    self.log(f"  Length {context_length}, Depth {depth_percent}%: {avg_score:.2%}")
        
        # Compute metrics
        overall_accuracy = total_correct / total_cases if total_cases > 0 else 0
        
        metrics = {
            "accuracy": overall_accuracy,
            "total_correct": total_correct,
            "total_cases": total_cases,
        }
        
        # Add per-length accuracy
        for length in self.context_lengths:
            scores = []
            for depth in self.depth_percents:
                scores.extend(results_matrix.get((length, depth), []))
            if scores:
                metrics[f"accuracy_len_{length}"] = sum(scores) / len(scores)
        
        # Add per-depth accuracy
        for depth in self.depth_percents:
            scores = []
            for length in self.context_lengths:
                scores.extend(results_matrix.get((length, depth), []))
            if scores:
                metrics[f"accuracy_depth_{depth}"] = sum(scores) / len(scores)
        
        self.log(f"Final accuracy: {overall_accuracy:.2%} ({total_correct}/{total_cases})")
        
        return BenchmarkResult(
            benchmark_name=self.name,
            method_name=method.name,
            metrics=metrics,
            sample_results=sample_results,
            config={
                "context_lengths": self.context_lengths,
                "depth_percents": self.depth_percents,
                "num_samples_per_config": self.num_samples_per_config,
            },
            errors=errors,
            num_samples=total_cases + len(errors),
            num_successful=total_cases,
        )

