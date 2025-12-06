"""
Sanity Test: Force Matching Pipeline
=====================================

Verifies that:
1. Trajectory data loads correctly with forces
2. Forces flow through the dataset pipeline
3. Loss function correctly uses force weights
4. Basic training step works

Run: python tests/test_force_matching_sanity.py
"""

import sys
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, "src")

from fmkv.data.trajectory import load_trajectories
from fmkv.data.multi_window_dataset import MultiWindowDataset
from fmkv.losses.force_matching import ForceMatchingLoss, ForceMatchingLossConfig
from fmkv.sidecar import Sidecar, SidecarConfig


def test_trajectory_loading():
    """Test 1: Load trajectory and verify forces are present."""
    logger.info("=" * 60)
    logger.info("Test 1: Trajectory Loading")
    logger.info("=" * 60)

    # Load first trajectory file
    data = torch.load("data/tinyllama_trajectories/trajectory_00001.pt", weights_only=False)

    windows = data["windows"]
    logger.info(f"Loaded {len(windows)} windows")

    # Check first window
    w = windows[0]
    logger.info(f"Window keys: {list(w.keys())}")

    # Verify forces exist
    assert "forces" in w, "Missing 'forces' in trajectory!"
    assert "force_queries" in w, "Missing 'force_queries' in trajectory!"

    # Check shapes
    logger.info(f"  keys shape: {w['keys'].shape}")
    logger.info(f"  values shape: {w['values'].shape}")
    logger.info(f"  queries shape: {w['queries'].shape}")
    logger.info(f"  forces shape: {w['forces'].shape}")
    logger.info(f"  force_queries shape: {w['force_queries'].shape}")

    # Verify dimensions
    window_size, head_dim = w['keys'].shape
    num_queries, hidden_dim = w['queries'].shape
    force_window_size, force_hidden = w['forces'].shape
    force_query_size, force_query_hidden = w['force_queries'].shape

    assert window_size == force_window_size, f"Window size mismatch: {window_size} vs {force_window_size}"
    assert num_queries == force_query_size, f"Query count mismatch: {num_queries} vs {force_query_size}"

    logger.info("✓ Trajectory loading PASSED")
    return True


def test_dataset_pipeline():
    """Test 2: Verify forces flow through dataset."""
    logger.info("=" * 60)
    logger.info("Test 2: Dataset Pipeline")
    logger.info("=" * 60)

    # Load trajectories
    trajectories = load_trajectories("data/tinyllama_trajectories")
    logger.info(f"Loaded {len(trajectories)} trajectory windows")

    # Check if forces are loaded
    has_forces = sum(1 for t in trajectories if t.forces is not None)
    has_force_queries = sum(1 for t in trajectories if t.force_queries is not None)
    logger.info(f"  Windows with forces: {has_forces}/{len(trajectories)}")
    logger.info(f"  Windows with force_queries: {has_force_queries}/{len(trajectories)}")

    # Create dataset
    dataset = MultiWindowDataset(
        trajectories=trajectories,
        num_windows_per_sample=4,
        window_size=64,
        d_head=64,
    )
    logger.info(f"Created dataset with {len(dataset)} samples")

    # Get a sample
    sample = dataset[0]
    logger.info(f"Sample keys: {list(sample.keys())}")

    # Check sample shapes
    logger.info(f"  keys shape: {sample['keys'].shape}")
    logger.info(f"  values shape: {sample['values'].shape}")
    logger.info(f"  queries shape: {sample['queries'].shape}")

    if "forces" in sample:
        logger.info(f"  forces shape: {sample['forces'].shape}")
    else:
        logger.warning("  forces NOT in sample!")

    if "force_queries" in sample:
        logger.info(f"  force_queries shape: {sample['force_queries'].shape}")
    else:
        logger.warning("  force_queries NOT in sample!")

    # Test collate function
    collate_fn = dataset.get_collate_fn()
    batch = collate_fn([dataset[0], dataset[1]])
    logger.info(f"Batch keys: {list(batch.keys())}")

    if "force_queries" in batch:
        logger.info(f"  Batched force_queries shape: {batch['force_queries'].shape}")

    logger.info("✓ Dataset pipeline PASSED")
    return True


def test_loss_function():
    """Test 3: Verify loss function uses force weights."""
    logger.info("=" * 60)
    logger.info("Test 3: Loss Function")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Create loss function
    config = ForceMatchingLossConfig(
        consistency_weight=50.0,
        force_matching_weight=0.0,  # Disabled for n=1
    )
    loss_fn = ForceMatchingLoss(d_head=64, config=config)

    # Create synthetic data
    batch_size = 4
    num_windows = 4
    window_size = 64
    d_head = 64
    num_queries = 16
    hidden_dim = 2048

    # Dense KV
    keys = torch.randn(batch_size, num_windows * window_size, d_head, device=device)
    values = torch.randn(batch_size, num_windows * window_size, d_head, device=device)

    # Queries
    queries = torch.randn(batch_size, num_queries, d_head, device=device)

    # Compressed KV (4 super-tokens)
    k_cg = torch.randn(batch_size, num_windows, d_head, device=device)
    v_cg = torch.randn(batch_size, num_windows, d_head, device=device)

    # Force vectors
    forces = torch.randn(batch_size, num_windows * window_size, hidden_dim, device=device)
    force_queries = torch.randn(batch_size, num_queries, hidden_dim, device=device)

    # Test without force_queries
    logger.info("Testing loss WITHOUT force_queries...")
    loss1, metrics1 = loss_fn(queries, keys, values, k_cg, v_cg, forces=forces)
    logger.info(f"  Loss: {loss1.item():.4f}")
    logger.info(f"  Using force weights: {metrics1.get('force/using_force_weights', 'N/A')}")

    # Test with force_queries
    logger.info("Testing loss WITH force_queries...")
    loss2, metrics2 = loss_fn(queries, keys, values, k_cg, v_cg, forces=forces, force_queries=force_queries)
    logger.info(f"  Loss: {loss2.item():.4f}")
    logger.info(f"  Using force weights: {metrics2.get('force/using_force_weights', 'N/A')}")

    # The losses should be different when using force_queries
    logger.info(f"Loss difference: {abs(loss1.item() - loss2.item()):.6f}")

    logger.info("✓ Loss function PASSED")
    return True


def test_training_step():
    """Test 4: Full training step with real data."""
    logger.info("=" * 60)
    logger.info("Test 4: Training Step")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load just one trajectory file for faster testing
    data = torch.load("data/tinyllama_trajectories/trajectory_00001.pt", weights_only=False)
    from fmkv.data.trajectory import _parse_trajectory_window
    trajectories = [_parse_trajectory_window(w) for w in data["windows"][:100]]
    trajectories = [t for t in trajectories if t is not None]
    logger.info(f"Loaded {len(trajectories)} windows for quick test")
    dataset = MultiWindowDataset(
        trajectories=trajectories,
        num_windows_per_sample=4,
        window_size=64,
        d_head=64,
    )

    collate_fn = dataset.get_collate_fn()
    batch = collate_fn([dataset[i] for i in range(min(4, len(dataset)))])

    # Move to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    # Create sidecar
    sidecar_config = SidecarConfig(
        d_head=64,
        encoder_hidden_dim=256,
        encoder_num_layers=2,
        dtype="float32",
    )
    sidecar = Sidecar(sidecar_config).to(device)

    # Create loss
    loss_fn = ForceMatchingLoss(d_head=64)

    # Get data
    keys = batch["keys"]  # (batch, num_windows, window_size, d_head)
    values = batch["values"]
    queries = batch["queries"]
    forces = batch.get("forces")
    force_queries = batch.get("force_queries")

    logger.info(f"Batch shapes:")
    logger.info(f"  keys: {keys.shape}")
    logger.info(f"  values: {values.shape}")
    logger.info(f"  queries: {queries.shape}")
    if forces is not None:
        logger.info(f"  forces: {forces.shape}")
    if force_queries is not None:
        logger.info(f"  force_queries: {force_queries.shape}")

    # Flatten windows for dense attention
    batch_size, num_windows, window_size, d_head = keys.shape
    keys_flat = keys.reshape(batch_size, num_windows * window_size, d_head)
    values_flat = values.reshape(batch_size, num_windows * window_size, d_head)

    # Flatten forces if present
    if forces is not None:
        forces_flat = forces.reshape(batch_size, num_windows * window_size, -1)
    else:
        forces_flat = None

    # Compress each window
    keys_for_sidecar = keys.reshape(batch_size * num_windows, window_size, d_head)
    values_for_sidecar = values.reshape(batch_size * num_windows, window_size, d_head)
    k_cg, v_cg = sidecar.compress_cache(keys_for_sidecar, values_for_sidecar)
    k_cg = k_cg.reshape(batch_size, num_windows, d_head)
    v_cg = v_cg.reshape(batch_size, num_windows, d_head)

    # Handle query dimension mismatch
    q_dim = queries.size(-1)
    if q_dim != d_head:
        queries = queries[..., :d_head] if q_dim > d_head else torch.nn.functional.pad(queries, (0, d_head - q_dim))

    # Compute loss
    loss, metrics = loss_fn(
        queries=queries,
        keys=keys_flat,
        values=values_flat,
        k_cg=k_cg,
        v_cg=v_cg,
        forces=forces_flat,
        force_queries=force_queries,
    )

    logger.info(f"Loss: {loss.item():.4f}")
    logger.info(f"Metrics:")
    for k, v in metrics.items():
        if torch.is_tensor(v):
            logger.info(f"  {k}: {v.item():.4f}")

    # Test backward
    loss.backward()

    # Check gradients exist
    total_params = sum(p.numel() for p in sidecar.parameters())
    params_with_grad = sum(p.numel() for p in sidecar.parameters() if p.grad is not None)
    logger.info(f"Parameters with gradients: {params_with_grad}/{total_params}")

    # Check gradient norms
    grad_norm = sum(p.grad.norm().item() ** 2 for p in sidecar.parameters() if p.grad is not None) ** 0.5
    logger.info(f"Gradient norm: {grad_norm:.4f}")

    assert grad_norm > 0, "Gradients are zero - training won't work!"

    logger.info("✓ Training step PASSED")
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("FORCE MATCHING SANITY TESTS")
    logger.info("=" * 60)

    results = []

    try:
        results.append(("Trajectory Loading", test_trajectory_loading()))
    except Exception as e:
        logger.error(f"Test 1 FAILED: {e}")
        results.append(("Trajectory Loading", False))

    try:
        results.append(("Dataset Pipeline", test_dataset_pipeline()))
    except Exception as e:
        logger.error(f"Test 2 FAILED: {e}")
        results.append(("Dataset Pipeline", False))

    try:
        results.append(("Loss Function", test_loss_function()))
    except Exception as e:
        logger.error(f"Test 3 FAILED: {e}")
        results.append(("Loss Function", False))

    try:
        results.append(("Training Step", test_training_step()))
    except Exception as e:
        logger.error(f"Test 4 FAILED: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Training Step", False))

    # Summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        logger.info(f"  {name}: {status}")

    logger.info(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        logger.info("\n🎉 All tests passed! Pipeline is ready for training.")
    else:
        logger.error("\n❌ Some tests failed. Fix issues before training.")
        sys.exit(1)
