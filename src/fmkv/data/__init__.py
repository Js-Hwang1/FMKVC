"""
Data Module
===========

Components for collecting training data and creating datasets:
- TrajectoryCollector: Collect KV states from model forward passes
- GradientSnapshotter: Compute and cache "true forces" for training
- ForceMatchingDataset: PyTorch Dataset for training Sidecar
"""

from fmkv.data.trajectory import TrajectoryCollector
from fmkv.data.dataset import ForceMatchingDataset, create_dataloader
from fmkv.data.gradient_cache import GradientCache

__all__ = [
    "TrajectoryCollector",
    "ForceMatchingDataset",
    "create_dataloader",
    "GradientCache",
]

