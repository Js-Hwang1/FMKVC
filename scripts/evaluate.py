#!/usr/bin/env python3
"""
Main evaluation script for KV cache compression methods.

Usage:
    # Quick evaluation (Dense baseline on perplexity)
    python scripts/evaluate.py \
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --methods dense \
        --benchmarks perplexity

    # Full evaluation
    python scripts/evaluate.py \
        --model_name meta-llama/Llama-2-7b-hf \
        --methods dense fmkv \
        --benchmarks perplexity passkey needle \
        --output_dir ./results \
        --sidecar_checkpoint ./checkpoints/sidecar.pt

    # Compare compression ratios
    python scripts/evaluate.py \
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --methods dense fmkv \
        --compression_ratios 0.5 0.25 0.1 \
        --benchmarks perplexity passkey \
        --output_dir ./results/compression_sweep
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fmkv.evaluation import (
    get_benchmark,
    get_method,
    list_benchmarks,
    list_methods,
    MethodConfig,
    BenchmarkResult,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate KV cache compression methods",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Quick test
    python scripts/evaluate.py --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 --methods dense --benchmarks perplexity --num_samples 10
    
    # Full evaluation
    python scripts/evaluate.py --model_name meta-llama/Llama-2-7b-hf --methods dense fmkv --benchmarks perplexity passkey needle longbench
        """,
    )
    
    # Model settings
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    
    # Methods
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["dense"],
        choices=list_methods(),
        help=f"Methods to evaluate. Available: {list_methods()}",
    )
    
    # FMKV-specific
    parser.add_argument(
        "--sidecar_checkpoint",
        type=str,
        default=None,
        help="Path to trained Sidecar checkpoint (for FMKV method)",
    )
    parser.add_argument(
        "--compression_ratios",
        type=float,
        nargs="+",
        default=[0.5],
        help="Compression ratios to evaluate (for FMKV)",
    )
    parser.add_argument(
        "--cache_budget",
        type=int,
        default=None,
        help="Fixed cache budget in tokens (alternative to compression ratio)",
    )
    
    # Benchmarks
    parser.add_argument(
        "--benchmarks",
        type=str,
        nargs="+",
        default=["perplexity"],
        choices=list_benchmarks(),
        help=f"Benchmarks to run. Available: {list_benchmarks()}",
    )
    
    # Benchmark settings
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per benchmark (None = full dataset)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum context length",
    )
    parser.add_argument(
        "--perplexity_dataset",
        type=str,
        default="wikitext-2",
        choices=["wikitext-2", "wikitext-103", "c4", "pg19"],
        help="Dataset for perplexity evaluation",
    )
    parser.add_argument(
        "--passkey_lengths",
        type=int,
        nargs="+",
        default=[1024, 2048, 4096],
        help="Context lengths for passkey benchmark",
    )
    parser.add_argument(
        "--longbench_datasets",
        type=str,
        nargs="+",
        default=None,
        help="Specific LongBench datasets to evaluate",
    )
    
    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./results",
        help="Directory to save results",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help="Experiment name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--save_samples",
        action="store_true",
        help="Save per-sample results (can be large)",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )
    
    return parser.parse_args()


def create_experiment_name(args) -> str:
    """Generate experiment name from settings."""
    if args.experiment_name:
        return args.experiment_name
    
    model_short = args.model_name.split("/")[-1][:20]
    methods_str = "_".join(args.methods)
    benchmarks_str = "_".join(args.benchmarks[:3])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    return f"{model_short}_{methods_str}_{benchmarks_str}_{timestamp}"


def setup_output_dir(args) -> Path:
    """Create output directory structure."""
    exp_name = create_experiment_name(args)
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config = vars(args).copy()
    config["timestamp"] = datetime.now().isoformat()
    config["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        config["gpu_name"] = torch.cuda.get_device_name(0)
        config["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    return output_dir


def create_method(
    method_name: str,
    args,
    compression_ratio: Optional[float] = None,
) -> "BaseMethod":
    """Create a method instance with configuration."""
    config = MethodConfig(
        model_name=args.model_name,
        torch_dtype=args.torch_dtype,
        device=args.device,
        compression_ratio=compression_ratio,
        cache_budget=args.cache_budget,
        method_kwargs={
            "sidecar_checkpoint": args.sidecar_checkpoint,
        },
    )
    
    return get_method(method_name, config=config)


def create_benchmark(benchmark_name: str, args) -> "BaseBenchmark":
    """Create a benchmark instance with configuration."""
    common_kwargs = {
        "num_samples": args.num_samples,
        "seed": args.seed,
        "verbose": not args.quiet,
    }
    
    if benchmark_name == "perplexity":
        return get_benchmark(
            benchmark_name,
            dataset_name=args.perplexity_dataset,
            max_length=args.max_length,
            **common_kwargs,
        )
    elif benchmark_name == "passkey":
        return get_benchmark(
            benchmark_name,
            context_lengths=args.passkey_lengths,
            num_samples_per_config=args.num_samples or 10,
            **{k: v for k, v in common_kwargs.items() if k != "num_samples"},
        )
    elif benchmark_name == "needle":
        return get_benchmark(
            benchmark_name,
            context_lengths=args.passkey_lengths,
            num_samples_per_config=args.num_samples or 3,
            **{k: v for k, v in common_kwargs.items() if k != "num_samples"},
        )
    elif benchmark_name == "longbench":
        return get_benchmark(
            benchmark_name,
            datasets=args.longbench_datasets,
            num_samples_per_dataset=args.num_samples or 100,
            max_length=args.max_length,
            **{k: v for k, v in common_kwargs.items() if k != "num_samples"},
        )
    else:
        return get_benchmark(benchmark_name, **common_kwargs)


def save_results(
    results: list[BenchmarkResult],
    output_dir: Path,
    save_samples: bool = False,
) -> None:
    """Save evaluation results."""
    # Summary JSON
    summary = []
    for result in results:
        entry = result.to_dict()
        if not save_samples:
            entry.pop("sample_results", None)
        summary.append(entry)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Markdown report
    report = generate_markdown_report(results)
    with open(output_dir / "report.md", "w") as f:
        f.write(report)
    
    # CSV for easy plotting
    csv_lines = ["method,benchmark,metric,value"]
    for result in results:
        for metric, value in result.metrics.items():
            csv_lines.append(f"{result.method_name},{result.benchmark_name},{metric},{value}")
    
    with open(output_dir / "metrics.csv", "w") as f:
        f.write("\n".join(csv_lines))
    
    print(f"\nResults saved to: {output_dir}")


def generate_markdown_report(results: list[BenchmarkResult]) -> str:
    """Generate a markdown report from results."""
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {datetime.now().isoformat()}",
        "",
        "## Summary",
        "",
    ]
    
    # Group by benchmark
    by_benchmark = {}
    for r in results:
        if r.benchmark_name not in by_benchmark:
            by_benchmark[r.benchmark_name] = []
        by_benchmark[r.benchmark_name].append(r)
    
    for benchmark_name, benchmark_results in by_benchmark.items():
        lines.append(f"### {benchmark_name}")
        lines.append("")
        lines.append("| Method | " + " | ".join(
            k for k in benchmark_results[0].metrics.keys() if not k.startswith("accuracy_")
        ) + " |")
        lines.append("|" + "---|" * (1 + len([
            k for k in benchmark_results[0].metrics.keys() if not k.startswith("accuracy_")
        ])))
        
        for r in benchmark_results:
            values = [
                f"{v:.4f}" if isinstance(v, float) else str(v)
                for k, v in r.metrics.items()
                if not k.startswith("accuracy_")
            ]
            lines.append(f"| {r.method_name} | " + " | ".join(values) + " |")
        
        lines.append("")
    
    # Errors
    all_errors = []
    for r in results:
        all_errors.extend(r.errors)
    
    if all_errors:
        lines.append("## Errors")
        lines.append("")
        for error in all_errors[:10]:
            lines.append(f"- {error}")
        if len(all_errors) > 10:
            lines.append(f"- ... and {len(all_errors) - 10} more")
        lines.append("")
    
    return "\n".join(lines)


def main():
    args = parse_args()
    
    if not args.quiet:
        print("=" * 60)
        print("FMKV Evaluation")
        print("=" * 60)
        print(f"Model: {args.model_name}")
        print(f"Methods: {args.methods}")
        print(f"Benchmarks: {args.benchmarks}")
        print(f"Device: {args.device}")
        print("=" * 60)
    
    # Setup output
    output_dir = setup_output_dir(args)
    
    # Create benchmarks
    benchmarks = {}
    for benchmark_name in args.benchmarks:
        try:
            benchmark = create_benchmark(benchmark_name, args)
            benchmark.setup()
            benchmarks[benchmark_name] = benchmark
        except Exception as e:
            print(f"Warning: Could not create benchmark {benchmark_name}: {e}")
    
    # Run evaluation
    all_results = []
    
    for method_name in args.methods:
        print(f"\n{'='*60}")
        print(f"Method: {method_name}")
        print("=" * 60)
        
        # For compression methods, evaluate at different ratios
        if method_name == "fmkv":
            compression_ratios = args.compression_ratios
        else:
            compression_ratios = [None]  # Dense doesn't use compression ratio
        
        for ratio in compression_ratios:
            # Create method
            method = create_method(method_name, args, compression_ratio=ratio)
            
            method_label = method_name
            if ratio is not None:
                method_label = f"{method_name}_r{ratio}"
                print(f"\nCompression ratio: {ratio}")
            
            # Setup method
            method.setup()
            
            # Run benchmarks
            for benchmark_name, benchmark in benchmarks.items():
                print(f"\nRunning {benchmark_name}...")
                start_time = time.perf_counter()
                
                try:
                    result = benchmark.evaluate(method)
                    result.method_name = method_label  # Update label if needed
                    all_results.append(result)
                    
                    elapsed = time.perf_counter() - start_time
                    print(f"  Completed in {elapsed:.1f}s")
                    
                    # Print key metrics
                    for key, value in result.metrics.items():
                        if not key.startswith("accuracy_") and not key.startswith("score_"):
                            if isinstance(value, float):
                                print(f"  {key}: {value:.4f}")
                            else:
                                print(f"  {key}: {value}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Clear CUDA cache between methods
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Save results
    save_results(all_results, output_dir, save_samples=args.save_samples)
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    
    # Print summary table
    print("\nSummary:")
    for result in all_results:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in list(result.metrics.items())[:3]
        )
        print(f"  {result.method_name}/{result.benchmark_name}: {metrics_str}")


if __name__ == "__main__":
    main()

