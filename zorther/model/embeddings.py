"""
Zorther Gen - SOTA Embeddings & Rotary Positional Embeddings (RoPE)
Optimized for:
- Interleaved Rotation (LLaMA, Qwen, DeepSeek Compatible)
- Safe Boundary Range & Sequence Overflow Checks
- Structural Even-Dimension Assertions
- No-redundant Device Transfers
"""

import torch
import torch.nn as nn
from typing import Tuple


class ZortherEmbeddings(nn.Module):

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # standard scaling to prevent gradient vanishing
        return self.token_embeddings(input_ids) * (self.hidden_size ** 0.5)


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0) -> None:
        super().__init__()
        # কড়া গাণিতিক চেক: RoPE-এর হেড ডাইমেনশন অবশ্যই জোড় সংখ্যা হতে হবে
        assert dim % 2 == 0, f"RoPE dimension (dim) must be even, but got: {dim}"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqss = torch.outer(t, self.inv_freq)
        
        # Interleaved sinusoids তৈরির জন্য ইন্টারলিভড ক্যাশিং মেকানিজম
        emb = torch.stack((freqss, freqss), dim=-1).flatten(-2)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, start_pos: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        # প্রোডাকশন গ্রেড বাউন্ডারি রেঞ্জ চেক
        if start_pos + seq_len > self.max_seq_len:
            raise RuntimeError(
                f"RoPE position sequence overflow! Maximum sequence length configured is {self.max_seq_len}, "
                f"but requested position slice goes up to: {start_pos + seq_len} (start_pos: {start_pos}, seq_len: {seq_len})."
            )
            
        # অপ্রয়োজনীয় ডিভাইস কপি রিমুভ করে ডাইনামিক এফিওরিটি নিশ্চিত করা
        cos = self.cos_cached[start_pos : start_pos + seq_len].to(device=device, non_blocking=True)
        sin = self.sin_cached[start_pos : start_pos + seq_len].to(device=device, non_blocking=True)
        return cos, sin


def _rotate_interleaved(x: torch.Tensor) -> torch.Tensor:
    """
    SOTA Interleaved Rotation (Qwen, LLaMA, DeepSeek Compatible)
    0 1 -> -1 0 | 2 3 -> -3 2
    """
    # শেষ ডাইমেনশনটিকে ২ ভাগে বিভক্ত করে রিম্যাপ করা হচ্ছে
    x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x_reshaped[..., 0]
    x2 = x_reshaped[..., 1]
    
    # ইন্টারলিভড রোটেশন স্ট্যাক ও ফ্ল্যাট করা হচ্ছে
    rotated = torch.stack((-x2, x1), dim=-1)
    return rotated.flatten(-2)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # (batch_size, seq_len, heads, head_dim) ডাইমেনশন এলাইনমেন্ট
    cos_q = cos.unsqueeze(0).unsqueeze(2)
    sin_q = sin.unsqueeze(0).unsqueeze(2)

    cos_k = cos.unsqueeze(0).unsqueeze(2)
    sin_k = sin.unsqueeze(0).unsqueeze(2)

    # ইন্টারলিভড রোটেশনের প্রয়োগ
    xq_out = (xq * cos_q) + (_rotate_interleaved(xq) * sin_q)
    xk_out = (xk * cos_k) + (_rotate_interleaved(xk) * sin_k)

    return xq_out, xk_out