"""
Tests for Sidecar Network
=========================

Unit tests for the Sidecar architecture and training components.
Run with: pytest tests/test_sidecar.py -v
"""

import pytest
import torch
import torch.nn as nn

from fmkv.sidecar import Sidecar, SidecarConfig
from fmkv.sidecar.encoder import TransformerEncoder, GINEncoder, create_encoder
from fmkv.sidecar.aggregator import SetTransformerAggregator, create_aggregator


class TestSidecarConfig:
    """Tests for SidecarConfig."""
    
    def test_default_config(self):
        config = SidecarConfig()
        assert config.d_head == 128
        assert config.window_size == 64
        assert config.encoder_num_layers == 3
    
    def test_input_output_dims(self):
        config = SidecarConfig(d_head=64)
        assert config.input_dim == 128  # 2 * d_head
        assert config.output_dim == 128
    
    def test_parameter_estimation(self):
        config = SidecarConfig(d_head=128, encoder_hidden_dim=256)
        estimated = config.estimate_parameters()
        assert estimated > 0
        assert estimated < 10_000_000  # Should be < 10M params


class TestEncoders:
    """Tests for encoder architectures."""
    
    @pytest.fixture
    def config(self):
        return SidecarConfig(
            d_head=64,
            window_size=16,
            encoder_hidden_dim=128,
            encoder_num_layers=2,
        )
    
    def test_transformer_encoder_forward(self, config):
        encoder = TransformerEncoder(config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.window_size, config.input_dim)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, config.window_size, config.encoder_hidden_dim)
    
    def test_gin_encoder_forward(self, config):
        config.encoder_type = "gin"
        encoder = GINEncoder(config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.window_size, config.input_dim)
        
        output = encoder(x)
        
        assert output.shape == (batch_size, config.window_size, config.encoder_hidden_dim)
    
    def test_encoder_factory(self, config):
        for enc_type in ["transformer", "gin", "mlp"]:
            config.encoder_type = enc_type
            encoder = create_encoder(config)
            assert encoder is not None


class TestAggregators:
    """Tests for aggregation mechanisms."""
    
    @pytest.fixture
    def config(self):
        return SidecarConfig(
            d_head=64,
            window_size=16,
            encoder_hidden_dim=128,
        )
    
    def test_set_transformer_aggregator(self, config):
        aggregator = SetTransformerAggregator(config)
        
        batch_size = 4
        x = torch.randn(batch_size, config.window_size, config.encoder_hidden_dim)
        
        output = aggregator(x)
        
        # Should compress N -> 1
        assert output.shape == (batch_size, config.encoder_hidden_dim)
    
    def test_aggregator_factory(self, config):
        for agg_type in ["set_transformer", "attention_pool", "mean_pool"]:
            config.aggregator_type = agg_type
            aggregator = create_aggregator(config)
            assert aggregator is not None


class TestSidecar:
    """Tests for the full Sidecar network."""
    
    @pytest.fixture
    def config(self):
        return SidecarConfig(
            d_head=64,
            window_size=16,
            encoder_hidden_dim=128,
            encoder_num_layers=2,
        )
    
    def test_sidecar_forward(self, config):
        sidecar = Sidecar(config)
        
        batch_size = 4
        kv_window = torch.randn(batch_size, config.window_size, config.input_dim)
        
        k_cg, v_cg = sidecar(kv_window)
        
        assert k_cg.shape == (batch_size, config.d_head)
        assert v_cg.shape == (batch_size, config.d_head)
    
    def test_sidecar_compress_cache(self, config):
        sidecar = Sidecar(config)
        
        batch_size = 4
        keys = torch.randn(batch_size, config.window_size, config.d_head)
        values = torch.randn(batch_size, config.window_size, config.d_head)
        
        k_cg, v_cg = sidecar.compress_cache(keys, values)
        
        assert k_cg.shape == (batch_size, config.d_head)
        assert v_cg.shape == (batch_size, config.d_head)
    
    def test_sidecar_batched_compression(self, config):
        sidecar = Sidecar(config)
        
        batch_size = 2
        seq_len = 64  # 4 windows
        
        keys = torch.randn(batch_size, seq_len, config.d_head)
        values = torch.randn(batch_size, seq_len, config.d_head)
        
        k_compressed, v_compressed = sidecar.compress_cache_batched(
            keys, values, window_size=config.window_size
        )
        
        num_windows = seq_len // config.window_size
        assert k_compressed.shape == (batch_size, num_windows, config.d_head)
        assert v_compressed.shape == (batch_size, num_windows, config.d_head)
    
    def test_sidecar_parameter_count(self, config):
        sidecar = Sidecar(config)
        
        assert sidecar.num_parameters > 0
        assert sidecar.num_parameters < 10_000_000  # Should be small
    
    def test_sidecar_gradient_flow(self, config):
        sidecar = Sidecar(config)
        
        batch_size = 2
        kv_window = torch.randn(
            batch_size, config.window_size, config.input_dim,
            requires_grad=True,
        )
        
        k_cg, v_cg = sidecar(kv_window)
        loss = k_cg.sum() + v_cg.sum()
        loss.backward()
        
        # Check gradients flow through
        assert kv_window.grad is not None
        assert kv_window.grad.shape == kv_window.shape


class TestSidecarDtypes:
    """Test Sidecar with different data types."""
    
    @pytest.fixture
    def config(self):
        return SidecarConfig(d_head=64, window_size=16)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_float16(self, config):
        sidecar = Sidecar(config).cuda().half()
        x = torch.randn(2, config.window_size, config.input_dim, 
                       dtype=torch.float16, device="cuda")
        k, v = sidecar(x)
        assert k.dtype == torch.float16
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_bfloat16(self, config):
        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported")
        
        sidecar = Sidecar(config).cuda().bfloat16()
        x = torch.randn(2, config.window_size, config.input_dim,
                       dtype=torch.bfloat16, device="cuda")
        k, v = sidecar(x)
        assert k.dtype == torch.bfloat16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

