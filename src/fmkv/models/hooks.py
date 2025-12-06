"""
Model Hooks
===========

Hook classes for capturing KV cache states, attention patterns,
and gradients from HuggingFace transformer models.

These hooks enable:
1. Extracting KV states without modifying model code
2. Capturing gradients for Force Matching training
3. Injecting compressed KV states during inference
"""

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import weakref

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class HookOutput:
    """Container for captured hook outputs."""
    
    layer_idx: int
    keys: Optional[torch.Tensor] = None
    values: Optional[torch.Tensor] = None
    queries: Optional[torch.Tensor] = None
    attention_weights: Optional[torch.Tensor] = None
    attention_output: Optional[torch.Tensor] = None
    
    # Gradients (if captured)
    key_grad: Optional[torch.Tensor] = None
    value_grad: Optional[torch.Tensor] = None
    query_grad: Optional[torch.Tensor] = None


class KVCacheHook:
    """
    Hook for capturing Key-Value cache states from attention layers.
    
    Registers forward hooks on attention layers to capture K, V projections
    after they're computed but before attention is applied.
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> hook = KVCacheHook(model, capture_layers=[0, 1, 2])
        >>> 
        >>> # Forward pass
        >>> outputs = model(**inputs)
        >>> 
        >>> # Access captured KV states
        >>> kv_states = hook.get_captured_states()
        >>> print(kv_states[0].keys.shape)  # Layer 0 keys
        >>> 
        >>> # Clean up
        >>> hook.remove()
    """
    
    def __init__(
        self,
        model: nn.Module,
        capture_layers: Optional[List[int]] = None,
        model_config: Optional["ModelConfig"] = None,
    ):
        """
        Initialize KV cache hook.
        
        Args:
            model: HuggingFace model to hook
            capture_layers: List of layer indices to capture (None = all layers)
            model_config: Model configuration (auto-detected if not provided)
        """
        self.model = model
        self.model_config = model_config
        self.capture_layers = capture_layers
        
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured: Dict[int, HookOutput] = {}
        self._enabled = True
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on attention layers."""
        from fmkv.models.registry import get_model_config, get_model_dimensions
        
        # Auto-detect model config if not provided
        if self.model_config is None:
            model_name = getattr(self.model.config, "_name_or_path", "")
            self.model_config = get_model_config(model_name)
            
            if self.model_config is None:
                raise ValueError(
                    f"Could not auto-detect model config for {model_name}. "
                    "Please provide model_config explicitly."
                )
        
        # Get model dimensions
        dims = get_model_dimensions(self.model_config, self.model.config)
        self.num_layers = dims["num_layers"]
        self.num_heads = dims["num_heads"]
        self.head_dim = dims["head_dim"]
        
        # Determine which layers to capture
        if self.capture_layers is None:
            self.capture_layers = list(range(self.num_layers))
        
        # Register hooks on each attention layer
        for layer_idx in self.capture_layers:
            layer_name = self.model_config.get_attention_layer_name(layer_idx)
            
            # Navigate to the attention module
            try:
                attn_module = self._get_module_by_name(self.model, layer_name)
            except AttributeError:
                logger.warning(f"Could not find attention module at {layer_name}")
                continue
            
            # Register forward hook
            handle = attn_module.register_forward_hook(
                self._create_forward_hook(layer_idx)
            )
            self._handles.append(handle)
    
    def _get_module_by_name(self, model: nn.Module, name: str) -> nn.Module:
        """Get a submodule by its dot-separated name."""
        parts = name.split(".")
        module = model
        for part in parts:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module
    
    def _create_forward_hook(self, layer_idx: int) -> Callable:
        """Create a forward hook function for a specific layer."""
        def hook(module: nn.Module, inputs: Tuple, outputs: Any):
            if not self._enabled:
                return
            
            # Extract KV states from the output
            # The exact format depends on the model architecture
            kv_state = self._extract_kv_from_output(module, inputs, outputs, layer_idx)
            
            if kv_state is not None:
                self._captured[layer_idx] = kv_state
        
        return hook
    
    def _extract_kv_from_output(
        self,
        module: nn.Module,
        inputs: Tuple,
        outputs: Any,
        layer_idx: int,
    ) -> Optional[HookOutput]:
        """
        Extract K, V tensors from attention module output.
        
        This handles different output formats across model families.
        """
        # Most modern models return: (attn_output, attn_weights, past_key_value)
        # or just (attn_output, past_key_value) if output_attentions=False
        
        hook_output = HookOutput(layer_idx=layer_idx)
        
        if isinstance(outputs, tuple):
            # Try to find KV cache in outputs
            for out in outputs:
                if isinstance(out, tuple) and len(out) == 2:
                    # This is likely (key_states, value_states)
                    if isinstance(out[0], torch.Tensor) and isinstance(out[1], torch.Tensor):
                        k, v = out
                        if k.dim() == 4:  # (batch, heads, seq, dim)
                            hook_output.keys = k
                            hook_output.values = v
                            break
                elif hasattr(out, 'key_cache') and hasattr(out, 'value_cache'):
                    # DynamicCache format
                    hook_output.keys = out.key_cache[layer_idx] if layer_idx < len(out.key_cache) else None
                    hook_output.values = out.value_cache[layer_idx] if layer_idx < len(out.value_cache) else None
                    break
        
        # If we couldn't extract from outputs, try from module's cached states
        if hook_output.keys is None:
            if hasattr(module, 'key_states'):
                hook_output.keys = module.key_states
            if hasattr(module, 'value_states'):
                hook_output.values = module.value_states
        
        # Check if we got valid tensors
        if hook_output.keys is not None and hook_output.values is not None:
            return hook_output
        
        return None
    
    def get_captured_states(self) -> Dict[int, HookOutput]:
        """Get all captured KV states."""
        return self._captured.copy()
    
    def get_layer_states(self, layer_idx: int) -> Optional[HookOutput]:
        """Get captured states for a specific layer."""
        return self._captured.get(layer_idx)
    
    def clear(self):
        """Clear captured states."""
        self._captured.clear()
    
    def enable(self):
        """Enable capturing."""
        self._enabled = True
    
    def disable(self):
        """Disable capturing (but keep hooks registered)."""
        self._enabled = False
    
    def remove(self):
        """Remove all hooks and clean up."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._captured.clear()
    
    def __enter__(self):
        self.enable()
        return self
    
    def __exit__(self, *args):
        self.disable()


class GradientCaptureHook:
    """
    Hook for capturing gradients during backpropagation.
    
    Used during training to compute the "true force" (gradients w.r.t. keys)
    that the Sidecar should learn to match.
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained(...)
        >>> grad_hook = GradientCaptureHook(model, capture_layers=[0, 1, 2])
        >>> 
        >>> # Forward pass with gradient tracking
        >>> outputs = model(**inputs)
        >>> loss = outputs.loss
        >>> 
        >>> # Backward pass
        >>> loss.backward()
        >>> 
        >>> # Access captured gradients
        >>> gradients = grad_hook.get_captured_gradients()
    """
    
    def __init__(
        self,
        model: nn.Module,
        capture_layers: Optional[List[int]] = None,
        capture_keys: bool = True,
        capture_values: bool = True,
        capture_queries: bool = False,
    ):
        self.model = model
        self.capture_layers = capture_layers
        self.capture_keys = capture_keys
        self.capture_values = capture_values
        self.capture_queries = capture_queries
        
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_grads: Dict[int, Dict[str, torch.Tensor]] = {}
        self._enabled = True
        
        # We'll register hooks lazily when we have the actual tensors
        self._tensor_refs: Dict[int, Dict[str, torch.Tensor]] = {}
    
    def register_tensor(
        self,
        layer_idx: int,
        tensor_name: str,
        tensor: torch.Tensor,
    ):
        """
        Register a tensor for gradient capture.
        
        Call this after the forward pass with the tensors you want to track.
        """
        if not tensor.requires_grad:
            return
        
        if layer_idx not in self._tensor_refs:
            self._tensor_refs[layer_idx] = {}
        
        self._tensor_refs[layer_idx][tensor_name] = tensor
        
        # Register backward hook
        def grad_hook(grad: torch.Tensor):
            if not self._enabled:
                return
            
            if layer_idx not in self._captured_grads:
                self._captured_grads[layer_idx] = {}
            
            self._captured_grads[layer_idx][tensor_name] = grad.detach().clone()
        
        handle = tensor.register_hook(grad_hook)
        self._handles.append(handle)
    
    def get_captured_gradients(self) -> Dict[int, Dict[str, torch.Tensor]]:
        """Get all captured gradients."""
        return self._captured_grads.copy()
    
    def clear(self):
        """Clear captured gradients."""
        self._captured_grads.clear()
        self._tensor_refs.clear()
    
    def enable(self):
        self._enabled = True
    
    def disable(self):
        self._enabled = False
    
    def remove(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()


class AttentionHook:
    """
    Hook for capturing and modifying attention computations.
    
    Can be used for:
    1. Capturing attention weights for analysis
    2. Injecting compressed KV states during inference
    3. Modifying attention patterns for ablation studies
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.target_layers = target_layers or []
        
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self._captured_attention: Dict[int, torch.Tensor] = {}
        self._injection_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._enabled = True
    
    def set_injection_cache(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """
        Set compressed KV cache to inject at a specific layer.
        
        During the next forward pass, the attention layer will use
        these KV states instead of computing them from the input.
        """
        self._injection_cache[layer_idx] = (keys, values)
    
    def clear_injection_cache(self):
        """Clear all injection caches."""
        self._injection_cache.clear()
    
    def get_captured_attention(self) -> Dict[int, torch.Tensor]:
        """Get captured attention weights."""
        return self._captured_attention.copy()
    
    def clear(self):
        """Clear all captured data."""
        self._captured_attention.clear()
        self._injection_cache.clear()
    
    def remove(self):
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.clear()


class CacheManager:
    """
    High-level manager for KV cache manipulation.
    
    Provides a clean interface for:
    1. Capturing KV states during forward passes
    2. Compressing windows using the Sidecar
    3. Maintaining hybrid dense/compressed caches
    """
    
    def __init__(
        self,
        model: nn.Module,
        sidecar: Optional[nn.Module] = None,
        window_size: int = 64,
        compression_threshold: int = 128,
    ):
        """
        Initialize cache manager.
        
        Args:
            model: The LLM to manage caches for
            sidecar: Optional Sidecar network for compression
            window_size: Window size for compression
            compression_threshold: Start compressing when cache exceeds this
        """
        self.model = model
        self.sidecar = sidecar
        self.window_size = window_size
        self.compression_threshold = compression_threshold
        
        self._kv_hook = KVCacheHook(model)
        
        # Hybrid cache: compressed + recent dense
        self._compressed_cache: Dict[int, Tuple[List[torch.Tensor], List[torch.Tensor]]] = {}
        self._dense_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    
    def update_cache(self, new_tokens: int = 1):
        """
        Update cache after generating new tokens.
        
        Checks if compression should be triggered and applies it.
        """
        captured = self._kv_hook.get_captured_states()
        
        for layer_idx, state in captured.items():
            if state.keys is None or state.values is None:
                continue
            
            keys, values = state.keys, state.values
            cache_len = keys.size(2)  # sequence length
            
            # Check if we should compress
            if cache_len >= self.compression_threshold and self.sidecar is not None:
                self._compress_oldest_window(layer_idx, keys, values)
    
    def _compress_oldest_window(
        self,
        layer_idx: int,
        keys: torch.Tensor,
        values: torch.Tensor,
    ):
        """Compress the oldest window in the cache."""
        # Extract window to compress
        window_keys = keys[:, :, :self.window_size, :]
        window_values = values[:, :, :self.window_size, :]
        
        # Compress using Sidecar
        # Note: Sidecar expects (batch, seq, 2*d_head) format
        # Need to handle multi-head format
        batch_size, num_heads, seq_len, head_dim = window_keys.shape
        
        # Process each head (or use MultiHeadSidecar)
        compressed_keys = []
        compressed_values = []
        
        for h in range(num_heads):
            k_h = window_keys[:, h, :, :]  # (batch, seq, d)
            v_h = window_values[:, h, :, :]
            
            k_cg, v_cg = self.sidecar.compress_cache(k_h, v_h)
            compressed_keys.append(k_cg)
            compressed_values.append(v_cg)
        
        # Stack compressed keys/values
        k_compressed = torch.stack(compressed_keys, dim=1)  # (batch, heads, d)
        v_compressed = torch.stack(compressed_values, dim=1)
        
        # Store in compressed cache
        if layer_idx not in self._compressed_cache:
            self._compressed_cache[layer_idx] = ([], [])
        
        self._compressed_cache[layer_idx][0].append(k_compressed)
        self._compressed_cache[layer_idx][1].append(v_compressed)
    
    def get_full_cache(
        self,
        layer_idx: int,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the full (compressed + dense) cache for a layer.
        
        Returns concatenated compressed tokens + recent dense tokens.
        """
        if layer_idx not in self._compressed_cache:
            return self._dense_cache.get(layer_idx)
        
        comp_keys, comp_values = self._compressed_cache[layer_idx]
        
        if not comp_keys:
            return self._dense_cache.get(layer_idx)
        
        # Concatenate compressed tokens
        k_compressed = torch.cat(comp_keys, dim=2)  # Along seq dim
        v_compressed = torch.cat(comp_values, dim=2)
        
        # Add dense cache if present
        if layer_idx in self._dense_cache:
            k_dense, v_dense = self._dense_cache[layer_idx]
            k_full = torch.cat([k_compressed.unsqueeze(2), k_dense], dim=2)
            v_full = torch.cat([v_compressed.unsqueeze(2), v_dense], dim=2)
        else:
            k_full = k_compressed.unsqueeze(2)
            v_full = v_compressed.unsqueeze(2)
        
        return k_full, v_full
    
    def clear(self):
        """Clear all caches."""
        self._compressed_cache.clear()
        self._dense_cache.clear()
        self._kv_hook.clear()
    
    def remove(self):
        """Clean up hooks."""
        self._kv_hook.remove()
        self.clear()
