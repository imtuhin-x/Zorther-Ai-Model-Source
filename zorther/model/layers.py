"""
Zorther Gen - SOTA Performance-Aligned Transformer Layers
Features:
- Fixed parallel FFN logic independent of bias flags
- Structured validation for activation functions (ValueErrors on Typos)
- Cleaned and aligned ResidualConnections with scaled drop-path and LayerScale
- Optimized fused tensor chunks in ParallelFeedForward
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from zorther.config.model_config import ZortherModelConfig
from zorther.model.attention import ZortherAttention, AttentionConfig
from zorther.model.cache import KVCache


class RMSNorm(nn.Module):
    """
    SOTA Root Mean Square Layer Normalization.
    Keeps numerical values stable in high-precision (float32) backpropagation.
    """
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight


class LayerScale(nn.Module):
    """
    SOTA Layer Scale parameter to stabilize deeper layer training.
    """
    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((dim,), init_value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale


class SwiGLU(nn.Module):

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.silu(gate) * up


class GeGLU(nn.Module):

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate) * up


class ReGLU(nn.Module):

    def forward(self, gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        return F.relu(gate) * up


class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network with verified activation validation.
    """
    def __init__(self, config: ZortherModelConfig) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)
        self.w3 = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)

        # কড়া অ্যাক্টিভেশন টাইপো হ্যান্ডলিং ও ভ্যালিডেশন
        act_fn = config.activation.lower()
        if act_fn == "swiglu":
            self.activation = SwiGLU()
        elif act_fn == "geglu":
            self.activation = GeGLU()
        elif act_fn == "reglu":
            self.activation = ReGLU()
        else:
            raise ValueError(f"Unsupported activation type: '{config.activation}'. Registered types: 'swiglu', 'geglu', 'reglu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.activation(self.w1(x), self.w3(x)))


class ParallelFeedForward(nn.Module):
    """
    Parallel Feed-Forward Network to speed up execution.
    Fuses w1 and w3 into a single projection layer.
    """
    def __init__(self, config: ZortherModelConfig) -> None:
        super().__init__()
        self.fused_gate_up = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=config.bias)
        self.w2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

        act_fn = config.activation.lower()
        if act_fn == "swiglu":
            self.activation = SwiGLU()
        elif act_fn == "geglu":
            self.activation = GeGLU()
        elif act_fn == "reglu":
            self.activation = ReGLU()
        else:
            raise ValueError(f"Unsupported activation type: '{config.activation}'. Registered types: 'swiglu', 'geglu', 'reglu'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.fused_gate_up(x)
        gate, up = fused.chunk(2, dim=-1)
        return self.w2(self.activation(gate, up))


class DropPath(nn.Module):

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class ResidualConnection(nn.Module):
    """
    SOTA Residual Connection utilizing scaled drop-path and LayerScale parameter.
    """
    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()
        self.layerscale = LayerScale(dim)
        self.droppath = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, residual_output: torch.Tensor, residual_scale: float = 1.0) -> torch.Tensor:
        return x + self.droppath(self.layerscale(residual_output)) * residual_scale


class TransformerBlock(nn.Module):

    def __init__(self, layer_id: int, config: ZortherModelConfig) -> None:
        super().__init__()
        self.layer_id = layer_id
        
        attn_config = AttentionConfig(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )
        
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.attention = ZortherAttention(attn_config)
        self.attention_residual = ResidualConnection(config.hidden_size, drop_path=config.dropout)

        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        
        # Parallel Feed Forward-এর ব্যবহার ফিক্সড
        self.feed_forward = ParallelFeedForward(config)
        self.ffn_residual = ResidualConnection(config.hidden_size, drop_path=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
        residual_scale: float = 1.0
    ) -> torch.Tensor:
        # অ্যাটেনশন সাব-লেয়ার ও রেসিডুয়াল অ্যাড
        h = self.attention_norm(x)
        h = self.attention(h, cos, sin, mask, kv_cache, start_pos)
        x = self.attention_residual(x, h, residual_scale)

        # FFN সাব-লেয়ার ও রেসিডুয়াল অ্যাড
        h = self.ffn_norm(x)
        h = self.feed_forward(h)
        x = self.ffn_residual(x, h, residual_scale)
        return x


class LayerFactory:

    @staticmethod
    def create_norm(norm_type: str, dim: int, eps: float = 1e-5) -> nn.Module:
        norm_type_lower = norm_type.lower()
        if norm_type_lower == "rmsnorm":
            return RMSNorm(dim, eps)
        elif norm_type_lower == "layernorm":
            return nn.LayerNorm(dim, eps=eps)
        else:
            raise ValueError(f"Unknown norm type: '{norm_type}'. Registered types: 'rmsnorm', 'layernorm'.")

    @staticmethod
    def create_ffn(config: ZortherModelConfig, parallel: bool = True) -> nn.Module:
        if parallel:
            return ParallelFeedForward(config)
        return FeedForward(config)