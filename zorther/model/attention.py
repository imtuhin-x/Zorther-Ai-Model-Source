"""
Zorther Gen - SOTA Attention Layers
Features:
- Strict mathematical Rotary Embedding (RoPE) application PRIOR to cache storage
- Correct sequence length and coordinate tracking inside KVCacheManager
- Optimized Multi-Head, Grouped-Query (GQA), and Multi-Query (MQA) support
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from zorther.model.embeddings import apply_rotary_emb
from zorther.model.cache import KVCache


@dataclass
class AttentionConfig:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_seq_len: int
    dropout: float = 0.0


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    bs, sl, n_kv_heads, head_dim = x.shape
    return x[:, :, :, None, :].expand(bs, sl, n_kv_heads, n_rep, head_dim).reshape(bs, sl, n_kv_heads * n_rep, head_dim)


def apply_attention_mask(scores: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is not None:
        scores = scores + mask
    return scores


class ScaledDotProductAttention(nn.Module):

    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout_layer = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        head_dim = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        scores = apply_attention_mask(scores, mask)
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout_layer(attention_probs)
        return torch.matmul(attention_probs, v)


class KVCacheManager:

    @staticmethod
    def update(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        start_pos: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if kv_cache is None:
            return key_states, value_states
        # গাণিতিক সিঙ্ক নিশ্চিত করে ক্যাশ আপডেট করা হচ্ছে
        return kv_cache.update(key_states, value_states, start_pos)


class MultiHeadAttention(nn.Module):

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.attention_engine = ScaledDotProductAttention(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        bs, sl, _ = x.shape
        q = self.q_proj(x).view(bs, sl, self.config.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(bs, sl, self.config.num_attention_heads, self.head_dim)
        v = self.v_proj(x).view(bs, sl, self.config.num_attention_heads, self.head_dim)

        # ১. অত্যন্ত গুরুত্বপূর্ণ: প্রথমে নতুন টোকেনের কি এবং কুয়েরির ওপর RoPE প্রয়োগ করা
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # ২. RoPE প্রয়োগ করার পর কন্ডিশনাল ক্যাশ আপডেট (0, 1, 2... N ইনডেক্সিং সহ)
        k, v = KVCacheManager.update(k, v, kv_cache, start_pos)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attention_engine(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, sl, -1)
        return self.o_proj(output)


class GroupedQueryAttention(nn.Module):

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.num_queries_per_kv = config.num_attention_heads // config.num_key_value_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.attention_engine = ScaledDotProductAttention(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        bs, sl, _ = x.shape
        q = self.q_proj(x).view(bs, sl, self.config.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(bs, sl, self.config.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(bs, sl, self.config.num_key_value_heads, self.head_dim)

        # ১. অত্যন্ত গুরুত্বপূর্ণ: প্রথমে নতুন টোকেনের কি এবং কুয়েরির ওপর RoPE প্রয়োগ করা
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # ২. RoPE প্রয়োগ করার পর কন্ডিশনাল ক্যাশ আপডেট
        k, v = KVCacheManager.update(k, v, kv_cache, start_pos)

        k = repeat_kv(k, self.num_queries_per_kv)
        v = repeat_kv(v, self.num_queries_per_kv)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attention_engine(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, sl, -1)
        return self.o_proj(output)


class MultiQueryAttention(nn.Module):

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=False)
        
        self.attention_engine = ScaledDotProductAttention(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        bs, sl, _ = x.shape
        q = self.q_proj(x).view(bs, sl, self.config.num_attention_heads, self.head_dim)
        k = self.k_proj(x).view(bs, sl, 1, self.head_dim)
        v = self.v_proj(x).view(bs, sl, 1, self.head_dim)

        # ১. অত্যন্ত গুরুত্বপূর্ণ: প্রথমে নতুন টোকেনের কি এবং কুয়েরির ওপর RoPE প্রয়োগ করা
        q, k = apply_rotary_emb(q, k, cos, sin)
        
        # ২. RoPE প্রয়োগ করার পর কন্ডিশনাল ক্যাশ আপডেট
        k, v = KVCacheManager.update(k, v, kv_cache, start_pos)

        k = repeat_kv(k, self.config.num_attention_heads)
        v = repeat_kv(v, self.config.num_attention_heads)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = self.attention_engine(q, k, v, mask)
        output = output.transpose(1, 2).contiguous().view(bs, sl, -1)
        return self.o_proj(output)


class ZortherAttention(nn.Module):

    def __init__(self, config: AttentionConfig) -> None:
        super().__init__()
        self.config = config
        
        if config.num_key_value_heads == config.num_attention_heads:
            self.core_attention = MultiHeadAttention(config)
        elif config.num_key_value_heads == 1:
            self.core_attention = MultiQueryAttention(config)
        else:
            self.core_attention = GroupedQueryAttention(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0
    ) -> torch.Tensor:
        return self.core_attention(x, cos, sin, mask, kv_cache, start_pos)