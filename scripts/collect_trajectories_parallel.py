#!/usr/bin/env python3
"""
Parallel Trajectory Collection for HPC
======================================

Launches multiple workers to collect trajectories in parallel.
Designed for HPC systems with many CPU cores and large memory.

Each worker processes a shard of the dataset independently,
writing to its own output directory. Results can be merged after.

Usage (single node, 96 cores):
    python scripts/collect_trajectories_parallel.py \
        --model_name TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --output_dir ./data/trajectories_large \
        --num_samples 1000000 \
        --num_workers 16 \
        --collect_forces

Usage with SLURM (multi-node):
    # See slurm_collect.sh for job script
"""

import argparse
import subprocess
import os
import sys
import json
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import shutil


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel trajectory collection for HPC"
    )

    # Model
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name",
    )

    # Output
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/trajectories_parallel",
        help="Base output directory",
    )

    # Parallelization
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: cpu_count // 6 for ~16 on 96 cores)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="Total number of samples to collect",
    )

    # Collection parameters
    parser.add_argument(
        "--window_size",
        type=int,
        default=64,
        help="Window size for KV extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size per worker",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--collect_forces",
        action="store_true",
        help="Collect force vectors (enables TRUE force matching)",
    )

    # Dataset
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="wikitext",
        help="Dataset name",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-103-raw-v1",
        help="Dataset config",
    )

    # Misc
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge results from previous parallel run",
    )

    return parser.parse_args()


def run_worker(args: Tuple[int, dict]) -> int:
    """Run a single collection worker."""
    worker_id, config = args

    output_dir = Path(config["output_dir"]) / f"shard_{worker_id:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "scripts/collect_trajectories.py",
        "--model_name", config["model_name"],
        "--output_dir", str(output_dir),
        "--num_samples", str(config["samples_per_worker"]),
        "--window_size", str(config["window_size"]),
        "--batch_size", str(config["batch_size"]),
        "--max_seq_length", str(config["max_seq_length"]),
        "--dataset_name", config["dataset_name"],
        "--dataset_config", config["dataset_config"],
        "--seed", str(42 + worker_id),  # Different seed per worker
        "--num_workers", "2",  # DataLoader workers (reduced to avoid oversubscription)
    ]

    if config["collect_forces"]:
        cmd.append("--collect_forces")

    print(f"[Worker {worker_id}] Starting: {' '.join(cmd[:5])}...")

    if config["dry_run"]:
        print(f"[Worker {worker_id}] DRY RUN: {' '.join(cmd)}")
        return 0

    # Run subprocess
    log_file = output_dir / "worker.log"
    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)

    print(f"[Worker {worker_id}] Finished with return code {result.returncode}")
    return result.returncode


def merge_shards(output_dir: Path, num_workers: int) -> Path:
    """Merge trajectory shards into single directory."""
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    file_idx = 0
    total_windows = 0

    # Copy metadata from first shard
    first_shard = output_dir / "shard_000"
    if (first_shard / "metadata.json").exists():
        shutil.copy(first_shard / "metadata.json", merged_dir / "metadata.json")

    # Merge trajectory files
    for worker_id in range(num_workers):
        shard_dir = output_dir / f"shard_{worker_id:03d}"
        if not shard_dir.exists():
            print(f"Warning: Shard {worker_id} not found")
            continue

        for traj_file in sorted(shard_dir.glob("trajectories_*.pt")):
            # Copy with new index
            dest = merged_dir / f"trajectories_{file_idx:05d}.pt"
            shutil.copy(traj_file, dest)
            file_idx += 1

            # Count windows
            import torch
            data = torch.load(traj_file, weights_only=False)
            total_windows += len(data.get("windows", []))

    print(f"Merged {file_idx} files with {total_windows} total windows")

    # Save merged stats
    stats = {
        "num_shards": num_workers,
        "num_files": file_idx,
        "total_windows": total_windows,
    }
    with open(merged_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return merged_dir


def estimate_disk_usage(num_samples: int, window_size: int, collect_forces: bool) -> float:
    """Estimate disk usage in GB."""
    # Per window: keys + values + queries + metadata
    # Keys/values: window_size * d_head * 4 bytes (float32) * 2 (K+V)
    d_head = 64  # TinyLlama
    kv_size = window_size * d_head * 4 * 2
    queries_size = 16 * d_head * 4  # 16 queries

    # Forces if collected: window_size * hidden_size * 4
    forces_size = window_size * 2048 * 4 if collect_forces else 0

    # Windows per sample (rough estimate)
    windows_per_sample = 10  # Depends on seq_length and stride

    bytes_per_sample = (kv_size + queries_size + forces_size) * windows_per_sample
    total_bytes = bytes_per_sample * num_samples

    return total_bytes / (1024 ** 3)


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine number of workers
    if args.num_workers is None:
        # Default: ~1/6 of CPUs to leave headroom for PyTorch threads
        args.num_workers = max(1, cpu_count() // 6)

    print("=" * 60)
    print("PARALLEL TRAJECTORY COLLECTION")
    print("=" * 60)
    print(f"Model: {args.model_name}")
    print(f"Output: {output_dir}")
    print(f"Total samples: {args.num_samples}")
    print(f"Workers: {args.num_workers}")
    print(f"Samples per worker: {args.num_samples // args.num_workers}")
    print(f"Collect forces: {args.collect_forces}")

    # Estimate resources
    estimated_gb = estimate_disk_usage(args.num_samples, args.window_size, args.collect_forces)
    print(f"\nEstimated disk usage: {estimated_gb:.1f} GB")
    print(f"Memory per worker: ~4-6 GB (TinyLlama + activations)")
    print(f"Total memory estimate: ~{args.num_workers * 5:.0f} GB")

    if args.merge:
        print("\n[Merge Mode] Merging existing shards...")
        merged_dir = merge_shards(output_dir, args.num_workers)
        print(f"Merged to: {merged_dir}")
        return

    # Prepare worker configs
    samples_per_worker = args.num_samples // args.num_workers

    config = {
        "model_name": args.model_name,
        "output_dir": str(output_dir),
        "samples_per_worker": samples_per_worker,
        "window_size": args.window_size,
        "batch_size": args.batch_size,
        "max_seq_length": args.max_seq_length,
        "collect_forces": args.collect_forces,
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "dry_run": args.dry_run,
    }

    # Save config
    with open(output_dir / "parallel_config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Launch workers
    print(f"\nLaunching {args.num_workers} workers...")

    worker_args = [(i, config) for i in range(args.num_workers)]

    with Pool(processes=args.num_workers) as pool:
        results = pool.map(run_worker, worker_args)

    # Check results
    failed = sum(1 for r in results if r != 0)
    if failed > 0:
        print(f"\nWarning: {failed}/{args.num_workers} workers failed")
    else:
        print(f"\nAll {args.num_workers} workers completed successfully!")

    # Auto-merge if all succeeded
    if failed == 0 and not args.dry_run:
        print("\nMerging shards...")
        merged_dir = merge_shards(output_dir, args.num_workers)
        print(f"\nFinal output: {merged_dir}")
        print(f"\nTo train Sidecar with this data:")
        print(f"  python scripts/train_sidecar.py \\")
        print(f"    --model_name {args.model_name} \\")
        print(f"    --trajectories_path {merged_dir}")


if __name__ == "__main__":
    main()
