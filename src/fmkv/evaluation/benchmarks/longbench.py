"""
LongBench benchmark integration.

LongBench is a comprehensive benchmark for long-context understanding
covering multiple tasks:
- Single-doc QA: NarrativeQA, Qasper, MultiFieldQA
- Multi-doc QA: HotpotQA, 2WikiMQA, MuSiQue
- Summarization: GovReport, QMSum, MultiNews
- Few-shot Learning: TREC, TriviaQA, SAMSum
- Synthetic: PassageCount, PassageRetrieval
- Code: RepoBench-P, LCC

Reference: "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding"
https://github.com/THUDM/LongBench
"""

import json
from typing import Optional

import torch
from datasets import load_dataset
from tqdm import tqdm

from ..methods.base import BaseMethod
from .base import BaseBenchmark, BenchmarkResult


class LongBenchBenchmark(BaseBenchmark):
    """
    LongBench benchmark for comprehensive long-context evaluation.
    
    Evaluates on multiple task types to provide a holistic view
    of long-context capabilities.
    """
    
    # Available datasets in LongBench
    DATASETS = {
        # Single-document QA
        "narrativeqa": {
            "category": "single_doc_qa",
            "metric": "f1",
            "max_length": 16384,
        },
        "qasper": {
            "category": "single_doc_qa",
            "metric": "f1",
            "max_length": 16384,
        },
        "multifieldqa_en": {
            "category": "single_doc_qa",
            "metric": "f1",
            "max_length": 8192,
        },
        # Multi-document QA
        "hotpotqa": {
            "category": "multi_doc_qa",
            "metric": "f1",
            "max_length": 16384,
        },
        "2wikimqa": {
            "category": "multi_doc_qa",
            "metric": "f1",
            "max_length": 16384,
        },
        "musique": {
            "category": "multi_doc_qa",
            "metric": "f1",
            "max_length": 16384,
        },
        # Summarization
        "gov_report": {
            "category": "summarization",
            "metric": "rouge",
            "max_length": 16384,
        },
        "qmsum": {
            "category": "summarization",
            "metric": "rouge",
            "max_length": 16384,
        },
        "multi_news": {
            "category": "summarization",
            "metric": "rouge",
            "max_length": 16384,
        },
        # Few-shot
        "trec": {
            "category": "few_shot",
            "metric": "accuracy",
            "max_length": 8192,
        },
        "triviaqa": {
            "category": "few_shot",
            "metric": "f1",
            "max_length": 16384,
        },
        "samsum": {
            "category": "few_shot",
            "metric": "rouge",
            "max_length": 8192,
        },
        # Synthetic
        "passage_count": {
            "category": "synthetic",
            "metric": "accuracy",
            "max_length": 16384,
        },
        "passage_retrieval_en": {
            "category": "synthetic",
            "metric": "accuracy",
            "max_length": 16384,
        },
        # Code
        "lcc": {
            "category": "code",
            "metric": "edit_similarity",
            "max_length": 16384,
        },
        "repobench-p": {
            "category": "code",
            "metric": "edit_similarity",
            "max_length": 16384,
        },
    }
    
    # Default subset for quick evaluation
    DEFAULT_SUBSETS = ["narrativeqa", "hotpotqa", "gov_report", "trec", "passage_count"]
    
    def __init__(
        self,
        datasets: list[str] = None,
        num_samples_per_dataset: int = 100,
        max_length: int = 8192,
        seed: int = 42,
        verbose: bool = True,
        **kwargs,
    ):
        """
        Initialize LongBench benchmark.
        
        Args:
            datasets: List of LongBench datasets to evaluate on
            num_samples_per_dataset: Maximum samples per dataset
            max_length: Maximum context length in tokens
            seed: Random seed
        """
        super().__init__(seed=seed, verbose=verbose, **kwargs)
        
        self.selected_datasets = datasets or self.DEFAULT_SUBSETS
        self.num_samples_per_dataset = num_samples_per_dataset
        self.max_length = max_length
        
        # Validate datasets
        for ds in self.selected_datasets:
            if ds not in self.DATASETS:
                raise ValueError(f"Unknown dataset: {ds}. Available: {list(self.DATASETS.keys())}")
        
        self.loaded_datasets = {}
    
    @property
    def name(self) -> str:
        return "longbench"
    
    @property
    def description(self) -> str:
        return f"LongBench evaluation on {len(self.selected_datasets)} datasets"
    
    def setup(self) -> None:
        """Load LongBench datasets."""
        if self._is_setup:
            return
        
        self.log("Loading LongBench datasets...")
        
        for ds_name in self.selected_datasets:
            try:
                # LongBench datasets are available on HuggingFace
                dataset = load_dataset(
                    "THUDM/LongBench",
                    ds_name,
                    split="test",
                    trust_remote_code=True,
                )
                
                # Limit samples
                if len(dataset) > self.num_samples_per_dataset:
                    dataset = dataset.select(range(self.num_samples_per_dataset))
                
                self.loaded_datasets[ds_name] = dataset
                self.log(f"  Loaded {ds_name}: {len(dataset)} samples")
                
            except Exception as e:
                self.log(f"  Warning: Could not load {ds_name}: {e}")
        
        self._is_setup = True
    
    def _build_prompt(self, sample: dict, ds_name: str) -> str:
        """Build prompt for a LongBench sample."""
        context = sample.get("context", "")
        input_text = sample.get("input", "")
        
        # Different prompt formats based on task type
        category = self.DATASETS[ds_name]["category"]
        
        if category in ["single_doc_qa", "multi_doc_qa"]:
            prompt = f"Context:\n{context}\n\nQuestion: {input_text}\n\nAnswer:"
        elif category == "summarization":
            prompt = f"Document:\n{context}\n\nSummarize the above document:\n"
        elif category == "few_shot":
            prompt = f"{context}\n\n{input_text}"
        elif category == "synthetic":
            prompt = f"{context}\n\n{input_text}"
        elif category == "code":
            prompt = f"{context}\n\n{input_text}"
        else:
            prompt = f"{context}\n\n{input_text}"
        
        return prompt
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute F1 score between prediction and ground truth."""
        pred_tokens = prediction.lower().split()
        gt_tokens = ground_truth.lower().split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = set(pred_tokens) & set(gt_tokens)
        
        if not common:
            return 0.0
        
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    def _compute_rouge(self, prediction: str, ground_truth: str) -> float:
        """Compute ROUGE-L score (simplified version)."""
        # Use F1 as a simple proxy for ROUGE-L
        return self._compute_f1(prediction, ground_truth)
    
    def _compute_accuracy(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match accuracy."""
        pred_clean = prediction.lower().strip()
        gt_clean = ground_truth.lower().strip()
        
        # Check if ground truth is contained in prediction
        if gt_clean in pred_clean:
            return 1.0
        
        # Check first word match for classification tasks
        pred_first = pred_clean.split()[0] if pred_clean else ""
        gt_first = gt_clean.split()[0] if gt_clean else ""
        
        return 1.0 if pred_first == gt_first else 0.0
    
    def _compute_metric(self, prediction: str, ground_truths: list[str], metric: str) -> float:
        """Compute the appropriate metric for a sample."""
        scores = []
        
        for gt in ground_truths:
            if metric == "f1":
                scores.append(self._compute_f1(prediction, gt))
            elif metric == "rouge":
                scores.append(self._compute_rouge(prediction, gt))
            elif metric == "accuracy":
                scores.append(self._compute_accuracy(prediction, gt))
            elif metric == "edit_similarity":
                # For code, use a simple token overlap
                scores.append(self._compute_f1(prediction, gt))
            else:
                scores.append(self._compute_f1(prediction, gt))
        
        return max(scores) if scores else 0.0
    
    def evaluate(self, method: BaseMethod) -> BenchmarkResult:
        """Evaluate on LongBench datasets."""
        if not self._is_setup:
            self.setup()
        
        # Ensure method is set up
        method.setup()
        
        self.log(f"Evaluating {method.name} on LongBench")
        
        all_results = []
        errors = []
        dataset_scores = {}
        category_scores = {}
        
        for ds_name, dataset in self.loaded_datasets.items():
            ds_info = self.DATASETS[ds_name]
            metric = ds_info["metric"]
            category = ds_info["category"]
            
            self.log(f"  Evaluating {ds_name} ({len(dataset)} samples)...")
            
            scores = []
            
            progress = tqdm(dataset, desc=ds_name, disable=not self.verbose, leave=False)
            
            for sample in progress:
                try:
                    # Build prompt
                    prompt = self._build_prompt(sample, ds_name)
                    
                    # Tokenize
                    inputs = method.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.max_length,
                    )
                    
                    # Generate
                    output = method.generate(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask"),
                        max_new_tokens=128,
                        do_sample=False,
                    )
                    
                    prediction = output.text[0].strip()
                    
                    # Get ground truth(s)
                    answers = sample.get("answers", [sample.get("answer", "")])
                    if isinstance(answers, str):
                        answers = [answers]
                    
                    # Compute score
                    score = self._compute_metric(prediction, answers, metric)
                    scores.append(score)
                    
                    all_results.append({
                        "dataset": ds_name,
                        "category": category,
                        "prediction": prediction[:200],
                        "ground_truth": answers[0][:100] if answers else "",
                        "score": score,
                    })
                    
                    method.reset_cache()
                    
                except Exception as e:
                    errors.append(f"{ds_name}: {str(e)}")
            
            # Compute dataset average
            if scores:
                avg_score = sum(scores) / len(scores)
                dataset_scores[ds_name] = avg_score
                
                # Accumulate category scores
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].extend(scores)
                
                self.log(f"    {ds_name}: {avg_score:.4f} ({metric})")
        
        # Compute overall metrics
        metrics = {}
        
        # Per-dataset scores
        for ds_name, score in dataset_scores.items():
            metrics[f"score_{ds_name}"] = score
        
        # Per-category averages
        for category, scores in category_scores.items():
            if scores:
                metrics[f"score_{category}"] = sum(scores) / len(scores)
        
        # Overall average
        all_scores = [r["score"] for r in all_results]
        if all_scores:
            metrics["score_overall"] = sum(all_scores) / len(all_scores)
        
        self.log(f"Overall score: {metrics.get('score_overall', 0):.4f}")
        
        return BenchmarkResult(
            benchmark_name=self.name,
            method_name=method.name,
            metrics=metrics,
            sample_results=all_results,
            config={
                "datasets": self.selected_datasets,
                "num_samples_per_dataset": self.num_samples_per_dataset,
                "max_length": self.max_length,
            },
            errors=errors,
            num_samples=len(all_results) + len(errors),
            num_successful=len(all_results),
        )

