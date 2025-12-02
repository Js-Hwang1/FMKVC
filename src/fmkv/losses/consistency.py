"""
Consistency Loss
================

Regularization losses to ensure the compressed cache maintains
consistent behavior with the original dense cache.

These act as auxiliary losses alongside the primary Force Matching loss.
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConsistencyLoss(nn.Module):
    """
    Consistency loss ensuring attention outputs are preserved.
    
    L_consistency = || Attn(q, K, V) - Attn(q, K_CG, V_CG) ||^2
    
    This is the regularization term λ·L_consistency from the paper,
    ensuring output magnitude preservation.
    """
    
    def __init__(self, d_head: int, normalize: bool = True):
        super().__init__()
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.normalize = normalize
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute consistency loss between dense and compressed attention.
        
        Args:
            queries: Shape (batch, num_queries, d_head)
            keys: Shape (batch, window_size, d_head)
            values: Shape (batch, window_size, d_head)
            k_cg: Shape (batch, d_head)
            v_cg: Shape (batch, d_head)
        
        Returns:
            Tuple of (loss, metrics)
        """
        # Dense attention output
        scores_dense = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights_dense = F.softmax(scores_dense, dim=-1)
        output_dense = torch.matmul(attn_weights_dense, values)  # (batch, num_q, d)
        
        # Compressed attention output
        k_cg = k_cg.unsqueeze(1)  # (batch, 1, d)
        v_cg = v_cg.unsqueeze(1)
        
        scores_cg = torch.matmul(queries, k_cg.transpose(-2, -1)) * self.scale
        attn_weights_cg = F.softmax(scores_cg, dim=-1)  # Will be all 1s
        output_cg = torch.matmul(attn_weights_cg, v_cg)  # (batch, num_q, d)
        
        # MSE loss
        diff = output_dense - output_cg
        loss = (diff ** 2).mean()
        
        # Cosine similarity for monitoring
        cos_sim = F.cosine_similarity(
            output_dense.reshape(-1, self.d_head),
            output_cg.reshape(-1, self.d_head),
            dim=-1,
        ).mean()
        
        metrics = {
            "loss/consistency": loss.detach(),
            "consistency/cosine_sim": cos_sim.detach(),
            "consistency/output_dense_norm": output_dense.norm(dim=-1).mean().detach(),
            "consistency/output_cg_norm": output_cg.norm(dim=-1).mean().detach(),
        }
        
        return loss, metrics


class AttentionEntropyLoss(nn.Module):
    """
    Entropy regularization for attention patterns.
    
    Encourages the Sidecar to produce K_CG that leads to
    attention distributions with similar entropy to dense attention.
    """
    
    def __init__(self, d_head: int, target_entropy: Optional[float] = None):
        super().__init__()
        self.d_head = d_head
        self.scale = d_head ** -0.5
        self.target_entropy = target_entropy
    
    def _compute_entropy(self, attn_weights: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Compute entropy of attention distribution."""
        # H = -Σ p log p
        log_weights = torch.log(attn_weights + eps)
        entropy = -(attn_weights * log_weights).sum(dim=-1)
        return entropy
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute attention entropy matching loss.
        """
        # Dense attention entropy
        scores_dense = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn_weights_dense = F.softmax(scores_dense, dim=-1)
        entropy_dense = self._compute_entropy(attn_weights_dense)  # (batch, num_q)
        
        # Note: With single CG token, entropy is always 0 (log(1) = 0)
        # So we can't directly match entropy
        # Instead, we use the dense entropy as a target for a proxy metric
        
        # Proxy: match the "concentration" via temperature scaling
        # High entropy -> spread attention, Low entropy -> focused
        mean_entropy = entropy_dense.mean()
        
        if self.target_entropy is not None:
            loss = (mean_entropy - self.target_entropy) ** 2
        else:
            # Just report, don't optimize
            loss = torch.tensor(0.0, device=queries.device, dtype=queries.dtype)
        
        metrics = {
            "attention/dense_entropy": mean_entropy.detach(),
            "attention/entropy_std": entropy_dense.std().detach(),
        }
        
        return loss, metrics


class GeometricPreservationLoss(nn.Module):
    """
    Loss ensuring geometric properties are preserved.
    
    Encourages K_CG, V_CG to lie in the span of original K, V,
    preserving the geometric structure of the KV space.
    """
    
    def __init__(self, d_head: int):
        super().__init__()
        self.d_head = d_head
    
    def forward(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute geometric preservation loss.
        
        Measures how well K_CG, V_CG can be reconstructed from
        the original K, V via linear combination.
        """
        batch_size = keys.size(0)
        
        # Compute centroid and deviation for reference
        k_centroid = keys.mean(dim=1)
        v_centroid = values.mean(dim=1)
        
        # Distance from centroid (should be small for good compression)
        k_dist = (k_cg - k_centroid).norm(dim=-1).mean()
        v_dist = (v_cg - v_centroid).norm(dim=-1).mean()
        
        # Projection onto span of original vectors
        # Using least squares: find α such that K @ α ≈ k_cg
        # This is expensive, so we use a proxy
        
        # Proxy: cosine similarity with nearest neighbor
        k_sims = F.cosine_similarity(
            k_cg.unsqueeze(1),  # (batch, 1, d)
            keys,  # (batch, seq, d)
            dim=-1,
        )  # (batch, seq)
        v_sims = F.cosine_similarity(
            v_cg.unsqueeze(1),
            values,
            dim=-1,
        )
        
        k_max_sim = k_sims.max(dim=-1).values.mean()
        v_max_sim = v_sims.max(dim=-1).values.mean()
        
        # Loss: encourage high similarity with at least one original vector
        # (1 - max_sim) penalizes being far from all originals
        loss = (1 - k_max_sim) + (1 - v_max_sim)
        
        metrics = {
            "geometric/k_centroid_dist": k_dist.detach(),
            "geometric/v_centroid_dist": v_dist.detach(),
            "geometric/k_max_similarity": k_max_sim.detach(),
            "geometric/v_max_similarity": v_max_sim.detach(),
        }
        
        return loss, metrics


class CombinedLoss(nn.Module):
    """
    Combined loss aggregating all loss components.
    
    Provides a convenient interface for training with multiple losses.
    """
    
    def __init__(
        self,
        d_head: int,
        force_matching_weight: float = 1.0,
        consistency_weight: float = 0.1,
        geometric_weight: float = 0.01,
    ):
        super().__init__()
        self.d_head = d_head
        
        # Import here to avoid circular import
        from fmkv.losses.force_matching import ForceMatchingLoss, ForceMatchingLossConfig
        
        config = ForceMatchingLossConfig(
            force_matching_weight=force_matching_weight,
            consistency_weight=consistency_weight,
        )
        
        self.force_matching_loss = ForceMatchingLoss(d_head, config)
        self.geometric_loss = GeometricPreservationLoss(d_head)
        self.geometric_weight = geometric_weight
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        k_cg: torch.Tensor,
        v_cg: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        """
        # Force matching (includes consistency)
        fm_loss, fm_metrics = self.force_matching_loss(
            queries, keys, values, k_cg, v_cg
        )
        
        # Geometric preservation
        geo_loss, geo_metrics = self.geometric_loss(keys, values, k_cg, v_cg)
        
        # Total
        total_loss = fm_loss + self.geometric_weight * geo_loss
        
        metrics = {**fm_metrics, **geo_metrics}
        metrics["loss/total_combined"] = total_loss.detach()
        metrics["loss/geometric"] = geo_loss.detach()
        
        return total_loss, metrics

