"""
Training Dataset
================

PyTorch Dataset for training the Sidecar network on collected
trajectories and precomputed gradients.
"""

from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from fmkv.data.trajectory import TrajectoryWindow, load_trajectories
from fmkv.data.gradient_cache import GradientCache, CachedGradient


class ForceMatchingDataset(Dataset):
    """
    Dataset for Force Matching training.
    
    Each sample contains:
    - KV window (keys, values for window_size tokens)
    - Future queries (queries that will attend to this window)
    - Target Jacobian (precomputed "true force" to match)
    - Target attention outputs (for consistency loss)
    
    Example:
        >>> dataset = ForceMatchingDataset.from_trajectories(
        ...     trajectories_path="./trajectories",
        ...     gradients_path="./gradients",
        ... )
        >>> 
        >>> sample = dataset[0]
        >>> print(sample["keys"].shape)      # (window_size, d_head)
        >>> print(sample["queries"].shape)   # (num_queries, d_head)
        >>> print(sample["target_jacobian"].shape)  # (d_head, d_head)
    """
    
    def __init__(
        self,
        trajectories: List[TrajectoryWindow],
        gradients: Optional[List[CachedGradient]] = None,
        window_size: int = 64,
        num_queries: int = 16,
        d_head: int = 128,
        precompute_gradients: bool = True,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize dataset.
        
        Args:
            trajectories: List of trajectory windows
            gradients: Optional precomputed gradients (computed if not provided)
            window_size: Expected window size
            num_queries: Number of future queries per sample
            d_head: Head dimension
            precompute_gradients: Whether to compute gradients if not provided
            device: Device for gradient computation
        """
        self.trajectories = trajectories
        self.window_size = window_size
        self.num_queries = num_queries
        self.d_head = d_head
        
        # Match trajectories with gradients
        if gradients is not None:
            assert len(gradients) == len(trajectories), (
                f"Mismatch: {len(gradients)} gradients vs {len(trajectories)} trajectories"
            )
            self.gradients = gradients
        elif precompute_gradients:
            # Compute gradients on-the-fly (slower startup, but complete)
            from fmkv.data.gradient_cache import GradientCache
            cache = GradientCache(d_head=d_head)
            self.gradients = cache.compute_from_trajectories(
                trajectories,
                device=device or torch.device("cpu"),
            )
        else:
            self.gradients = None
    
    @classmethod
    def from_trajectories(
        cls,
        trajectories_path: Union[str, Path],
        gradients_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> "ForceMatchingDataset":
        """
        Create dataset from saved trajectories.
        
        Args:
            trajectories_path: Path to saved trajectories
            gradients_path: Path to precomputed gradients (optional)
            **kwargs: Additional arguments for __init__
        
        Returns:
            ForceMatchingDataset instance
        """
        trajectories = load_trajectories(trajectories_path)
        
        if gradients_path is not None:
            grad_cache = GradientCache.load(gradients_path)
            gradients = list(grad_cache)
        else:
            gradients = None
        
        return cls(trajectories=trajectories, gradients=gradients, **kwargs)
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns dict with:
            - keys: (window_size, d_head)
            - values: (window_size, d_head)
            - queries: (num_queries, d_query)
            - target_jacobian: (d_head, d_head) if gradients available
            - target_outputs: (num_queries, d_head) if gradients available
            - layer_idx: int
            - window_idx: int
        """
        traj = self.trajectories[idx]
        
        sample = {
            "keys": traj.keys,
            "values": traj.values,
            "queries": traj.future_queries,
            "layer_idx": torch.tensor(traj.layer_idx),
            "window_idx": torch.tensor(traj.window_idx),
        }
        
        if self.gradients is not None:
            grad = self.gradients[idx]
            sample["target_jacobian"] = grad.aggregate_jacobian
            sample["target_outputs"] = grad.attention_outputs
        
        return sample
    
    def get_collate_fn(self):
        """Get a collate function for batching."""
        def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            """Stack batch samples."""
            keys = ["keys", "values", "queries", "layer_idx", "window_idx"]
            
            result = {}
            for key in keys:
                if key in batch[0]:
                    result[key] = torch.stack([b[key] for b in batch])
            
            # Optional gradient targets
            if "target_jacobian" in batch[0]:
                result["target_jacobian"] = torch.stack([b["target_jacobian"] for b in batch])
                result["target_outputs"] = torch.stack([b["target_outputs"] for b in batch])
            
            return result
        
        return collate_fn


class OnlineDataset(Dataset):
    """
    Dataset that generates training data on-the-fly.
    
    Instead of precomputing all trajectories, this dataset samples
    from a text corpus and extracts windows during training.
    More memory efficient for large datasets.
    """
    
    def __init__(
        self,
        model_wrapper: "ModelWrapper",
        text_dataset,
        window_size: int = 64,
        num_queries: int = 16,
        max_seq_length: int = 2048,
    ):
        self.model = model_wrapper
        self.text_dataset = text_dataset
        self.window_size = window_size
        self.num_queries = num_queries
        self.max_seq_length = max_seq_length
        
        self.tokenizer = model_wrapper.tokenizer
    
    def __len__(self) -> int:
        return len(self.text_dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Generate a training sample on-the-fly.
        
        This is slower but more memory efficient than precomputing all data.
        """
        text = self.text_dataset[idx]
        if isinstance(text, dict):
            text = text.get("text", text.get("content", str(text)))
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            max_length=self.max_seq_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"]
        
        # Forward pass
        with torch.no_grad():
            device = self.model.info.device
            input_ids = input_ids.to(device)
            
            with self.model.capture_kv():
                outputs = self.model.forward(input_ids, output_hidden_states=True)
                kv_states = self.model.get_kv_states()
                hidden_states = outputs.hidden_states
        
        # Sample a random window
        seq_len = input_ids.size(1)
        if seq_len < self.window_size + self.num_queries + 10:
            # Sequence too short, return dummy
            return self._get_dummy_sample()
        
        # Random start position
        max_start = seq_len - self.window_size - self.num_queries
        start = torch.randint(0, max_start, (1,)).item()
        end = start + self.window_size
        
        # Extract KV
        layer_idx = 0  # Use first layer for simplicity
        state = kv_states.get(layer_idx)
        if state is None:
            return self._get_dummy_sample()
        
        keys = state.keys[0, 0, start:end, :].cpu()  # First batch, first head
        values = state.values[0, 0, start:end, :].cpu()
        
        # Sample queries
        query_start = end
        query_end = min(end + self.num_queries * 2, seq_len)
        query_positions = torch.randperm(query_end - query_start)[:self.num_queries]
        query_positions = query_positions + query_start
        
        queries = hidden_states[layer_idx][0, query_positions, :].cpu()
        
        return {
            "keys": keys,
            "values": values,
            "queries": queries,
            "layer_idx": torch.tensor(layer_idx),
            "window_idx": torch.tensor(0),
        }
    
    def _get_dummy_sample(self) -> Dict[str, torch.Tensor]:
        """Return a dummy sample for too-short sequences."""
        d_head = self.model.info.head_dim
        hidden_size = self.model.info.hidden_size
        
        return {
            "keys": torch.zeros(self.window_size, d_head),
            "values": torch.zeros(self.window_size, d_head),
            "queries": torch.zeros(self.num_queries, hidden_size),
            "layer_idx": torch.tensor(0),
            "window_idx": torch.tensor(0),
        }


def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    **kwargs,
) -> DataLoader:
    """
    Create a DataLoader for training.
    
    Args:
        dataset: Training dataset
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Number of worker processes
        pin_memory: Pin memory for faster GPU transfer
        **kwargs: Additional DataLoader arguments
    
    Returns:
        DataLoader instance
    """
    collate_fn = None
    if hasattr(dataset, "get_collate_fn"):
        collate_fn = dataset.get_collate_fn()
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        **kwargs,
    )

