#!/usr/bin/env python3
"""
FMKV Demo Script
================

Demonstrates the Force-Matched KV Cache Compression framework.

This script:
1. Creates a Sidecar network
2. Generates synthetic KV data
3. Runs compression and computes force matching loss
4. Shows the compression works correctly

Usage:
    python scripts/demo.py
"""

import torch
import torch.nn.functional as F

from fmkv.sidecar import Sidecar, SidecarConfig
from fmkv.losses import ForceMatchingLoss
from fmkv.losses.jacobian import compute_attention_jacobian_batched


def print_separator():
    print("=" * 60)


def demo_sidecar():
    """Demonstrate Sidecar network compression."""
    print_separator()
    print("1. SIDECAR NETWORK DEMO")
    print_separator()
    
    # Configuration
    config = SidecarConfig(
        d_head=128,          # Head dimension (like Llama-2)
        n_heads=32,          # Number of heads
        window_size=64,      # Compress 64 tokens -> 1 super-token
        encoder_type="transformer",
        encoder_hidden_dim=256,
        encoder_num_layers=3,
    )
    
    print(f"\nSidecar Configuration:")
    print(f"  - Head dimension: {config.d_head}")
    print(f"  - Window size: {config.window_size}")
    print(f"  - Encoder: {config.encoder_type}")
    print(f"  - Hidden dim: {config.encoder_hidden_dim}")
    print(f"  - Layers: {config.encoder_num_layers}")
    
    # Create Sidecar
    sidecar = Sidecar(config)
    print(f"\nSidecar created!")
    print(f"  - Parameters: {sidecar.num_parameters:,}")
    print(f"  - Compression ratio: {config.window_size}:1")
    
    # Test compression
    batch_size = 4
    keys = torch.randn(batch_size, config.window_size, config.d_head)
    values = torch.randn(batch_size, config.window_size, config.d_head)
    
    print(f"\nInput shapes:")
    print(f"  - Keys: {keys.shape}")
    print(f"  - Values: {values.shape}")
    
    with torch.no_grad():
        k_cg, v_cg = sidecar.compress_cache(keys, values)
    
    print(f"\nOutput shapes (compressed):")
    print(f"  - K_CG: {k_cg.shape}")
    print(f"  - V_CG: {v_cg.shape}")
    
    print(f"\n✓ Successfully compressed {config.window_size} tokens -> 1 super-token!")
    
    return sidecar, config


def demo_force_matching():
    """Demonstrate force matching loss computation."""
    print_separator()
    print("\n2. FORCE MATCHING LOSS DEMO")
    print_separator()
    
    d_head = 128
    batch_size = 4
    window_size = 64
    num_queries = 16
    
    # Create loss function
    loss_fn = ForceMatchingLoss(d_head=d_head)
    
    # Generate test data
    queries = torch.randn(batch_size, num_queries, d_head)  # Future queries
    keys = torch.randn(batch_size, window_size, d_head)      # Original KV window
    values = torch.randn(batch_size, window_size, d_head)
    
    # Simulate compressed representations (would come from Sidecar)
    k_cg = keys.mean(dim=1)  # Simple mean for demo
    v_cg = values.mean(dim=1)
    
    print(f"\nComputing Force Matching Loss...")
    print(f"  - Queries: {queries.shape} (future queries)")
    print(f"  - Keys/Values: {keys.shape} (original window)")
    print(f"  - K_CG/V_CG: {k_cg.shape} (compressed)")
    
    loss, metrics = loss_fn(queries, keys, values, k_cg, v_cg)
    
    print(f"\nLoss breakdown:")
    print(f"  - Total loss: {metrics['loss/total'].item():.4f}")
    print(f"  - Force matching: {metrics['loss/force_matching'].item():.4f}")
    print(f"  - Consistency: {metrics['loss/consistency'].item():.4f}")
    print(f"  - Magnitude: {metrics['loss/magnitude'].item():.4f}")
    
    print(f"\nJacobian norms (force magnitudes):")
    print(f"  - Dense: {metrics['jacobian/dense_norm'].item():.4f}")
    print(f"  - Compressed: {metrics['jacobian/cg_norm'].item():.4f}")
    
    print(f"\n✓ Force matching loss computed successfully!")
    
    return loss_fn


def demo_jacobian_computation():
    """Demonstrate attention Jacobian computation."""
    print_separator()
    print("\n3. ATTENTION JACOBIAN DEMO")
    print_separator()
    
    d_head = 128
    batch_size = 2
    seq_len = 16
    num_queries = 4
    
    queries = torch.randn(batch_size, num_queries, d_head)
    keys = torch.randn(batch_size, seq_len, d_head)
    values = torch.randn(batch_size, seq_len, d_head)
    
    print(f"\nComputing attention Jacobian ∂Attn/∂q...")
    print(f"  - This represents the 'force' that KV cache exerts on queries")
    
    jacobians = compute_attention_jacobian_batched(queries, keys, values)
    
    print(f"\nJacobian shape: {jacobians.shape}")
    print(f"  - (batch, num_queries, d_out, d_in)")
    print(f"  - Each matrix is the gradient of attention output w.r.t. query")
    
    # Analyze Jacobian properties
    jacobian_norms = jacobians.norm(dim=(-2, -1))
    print(f"\nJacobian statistics:")
    print(f"  - Mean norm: {jacobian_norms.mean().item():.4f}")
    print(f"  - Std: {jacobian_norms.std().item():.4f}")
    print(f"  - Max: {jacobian_norms.max().item():.4f}")
    print(f"  - Min: {jacobian_norms.min().item():.4f}")
    
    print(f"\n✓ Jacobian computation successful!")


def demo_compression_quality():
    """Demonstrate how compression affects attention."""
    print_separator()
    print("\n4. COMPRESSION QUALITY ANALYSIS")
    print_separator()
    
    d_head = 128
    batch_size = 1
    window_size = 32
    num_queries = 8
    
    # Create data
    queries = torch.randn(batch_size, num_queries, d_head)
    keys = torch.randn(batch_size, window_size, d_head)
    values = torch.randn(batch_size, window_size, d_head)
    
    # Dense attention
    scale = d_head ** -0.5
    dense_scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
    dense_weights = F.softmax(dense_scores, dim=-1)
    dense_output = torch.matmul(dense_weights, values)
    
    # Mean pooling compression (baseline)
    k_mean = keys.mean(dim=1, keepdim=True)
    v_mean = values.mean(dim=1, keepdim=True)
    
    mean_scores = torch.matmul(queries, k_mean.transpose(-2, -1)) * scale
    mean_weights = F.softmax(mean_scores, dim=-1)
    mean_output = torch.matmul(mean_weights, v_mean)
    
    # Create a simple Sidecar for learned compression
    config = SidecarConfig(d_head=d_head, window_size=window_size)
    sidecar = Sidecar(config)
    
    with torch.no_grad():
        k_cg, v_cg = sidecar.compress_cache(keys, values)
    
    k_cg = k_cg.unsqueeze(1)
    v_cg = v_cg.unsqueeze(1)
    
    cg_scores = torch.matmul(queries, k_cg.transpose(-2, -1)) * scale
    cg_weights = F.softmax(cg_scores, dim=-1)
    cg_output = torch.matmul(cg_weights, v_cg)
    
    # Compare outputs
    mean_error = (dense_output - mean_output).norm() / dense_output.norm()
    cg_error = (dense_output - cg_output).norm() / dense_output.norm()
    
    mean_cosine = F.cosine_similarity(
        dense_output.flatten(), mean_output.flatten(), dim=0
    )
    cg_cosine = F.cosine_similarity(
        dense_output.flatten(), cg_output.flatten(), dim=0
    )
    
    print(f"\nCompression Quality Comparison:")
    print(f"\n  Dense Attention (Ground Truth):")
    print(f"    - Output norm: {dense_output.norm().item():.4f}")
    
    print(f"\n  Mean Pooling (Baseline):")
    print(f"    - Relative error: {mean_error.item():.4f}")
    print(f"    - Cosine similarity: {mean_cosine.item():.4f}")
    
    print(f"\n  Sidecar (Untrained):")
    print(f"    - Relative error: {cg_error.item():.4f}")
    print(f"    - Cosine similarity: {cg_cosine.item():.4f}")
    
    print(f"\n  Note: After training, Sidecar should significantly")
    print(f"  outperform mean pooling by preserving gradient dynamics!")
    
    print(f"\n✓ Compression quality analysis complete!")


def demo_training_step():
    """Demonstrate a single training step."""
    print_separator()
    print("\n5. TRAINING STEP DEMO")
    print_separator()
    
    # Setup
    d_head = 64
    window_size = 32
    num_queries = 8
    batch_size = 4
    
    config = SidecarConfig(
        d_head=d_head,
        window_size=window_size,
        encoder_hidden_dim=128,
        encoder_num_layers=2,
    )
    
    sidecar = Sidecar(config)
    loss_fn = ForceMatchingLoss(d_head=d_head)
    optimizer = torch.optim.AdamW(sidecar.parameters(), lr=1e-4)
    
    print(f"\nRunning training step...")
    
    # Generate batch
    queries = torch.randn(batch_size, num_queries, d_head)
    keys = torch.randn(batch_size, window_size, d_head)
    values = torch.randn(batch_size, window_size, d_head)
    
    # Forward pass
    k_cg, v_cg = sidecar.compress_cache(keys, values)
    
    # Compute loss
    loss, metrics = loss_fn(queries, keys, values, k_cg, v_cg)
    
    print(f"  - Forward pass complete")
    print(f"  - Loss: {loss.item():.4f}")
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    
    # Check gradients
    total_grad_norm = 0
    for p in sidecar.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
    total_grad_norm = total_grad_norm ** 0.5
    
    print(f"  - Backward pass complete")
    print(f"  - Gradient norm: {total_grad_norm:.4f}")
    
    # Optimizer step
    optimizer.step()
    print(f"  - Optimizer step complete")
    
    print(f"\n✓ Training step successful!")


def main():
    print("\n" + "=" * 60)
    print("  FORCE-MATCHED KV CACHE COMPRESSION DEMO")
    print("  A Physics-Inspired Framework for LLM Inference")
    print("=" * 60)
    
    # Run demos
    demo_sidecar()
    demo_force_matching()
    demo_jacobian_computation()
    demo_compression_quality()
    demo_training_step()
    
    print("\n" + "=" * 60)
    print("  DEMO COMPLETE!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Sidecar compresses N tokens -> 1 super-token")
    print("  2. Force Matching loss preserves gradient dynamics")
    print("  3. Jacobians capture the 'force' KV exerts on queries")
    print("  4. Training minimizes force mismatch between dense/compressed")
    print("\nNext steps:")
    print("  - Collect trajectories: python scripts/collect_trajectories.py")
    print("  - Train Sidecar: python scripts/train_sidecar.py")
    print("  - Evaluate: python scripts/evaluate.py")
    print()


if __name__ == "__main__":
    main()

