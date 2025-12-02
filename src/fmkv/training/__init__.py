"""
Training Module
===============

Components for training the Sidecar network:
- Trainer: Main training loop with logging and checkpointing
- Callbacks: Training callbacks for evaluation, early stopping, etc.
"""

from fmkv.training.trainer import SidecarTrainer, TrainingConfig

__all__ = [
    "SidecarTrainer",
    "TrainingConfig",
]

