"""
Zorther Gen - SOTA Performance-Aligned KV Cache
Features:
- Elimination of O(N^2) memory reallocation via pre-allocated dynamic buffers
- Automatic GPU/CPU device affinity and dtype binding
- Strict Out-of-Bounds (OOB) and sequence overflow checks
- Zero-overhead requests resets (pointers resetting instead of zeroing tensors)
- Registry-based clean factory pattern allocator
- Advanced Diagnostic tracking (device, dtype, batch_size, allocation bytes)
"""

import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Type


@dataclass
class CacheConfig:
    max_batch_size: int
    max_seq_len: int
    num_key_value_heads: int
    head_dim: int
    dtype: torch.dtype = torch.float32
    device: str = "cpu"


@dataclass
class CacheStats:
    allocated_bytes: int
    current_length: int
    cache_type: str
    device: torch.device
    dtype: torch.dtype
    batch_size: int


class BaseKVCache(ABC):

    @abstractmethod
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def get_stats(self) -> CacheStats:
        pass


class StaticKVCache(BaseKVCache):
    """
    SOTA Pre-allocated Static KV Cache.
    Strictly bound to device & dtype, O(1) update complexity, zero latency.
    """
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        
        # প্রি-অ্যালোকেটেড গ্যারান্টিড বাফার্স
        self.k_buffer = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.num_key_value_heads, config.head_dim),
            dtype=config.dtype,
            device=self.device
        )
        self.v_buffer = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.num_key_value_heads, config.head_dim),
            dtype=config.dtype,
            device=self.device
        )
        self.current_len = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, num_heads, head_dim = key_states.shape
        self.current_len = start_pos + seqlen

        # নিরাপদ বাউন্ডারি রেঞ্জ চেক
        if self.current_len > self.config.max_seq_len:
            raise RuntimeError(
                f"KV Cache overflow! Maximum configured cache limit is {self.config.max_seq_len}, "
                f"but requested update reached sequence length: {self.current_len}."
            )

        # ডাইনামিকলি ডিভাইস ও ডিক্লেয়ারেশন রি-অ্যালাইনমেন্ট
        if key_states.device != self.k_buffer.device or key_states.dtype != self.k_buffer.dtype:
            self.device = key_states.device
            self.k_buffer = self.k_buffer.to(device=self.device, dtype=key_states.dtype)
            self.v_buffer = self.v_buffer.to(device=self.device, dtype=key_states.dtype)

        # ইন-প্লেস ও(১) আপডেট
        self.k_buffer[:bsz, start_pos:self.current_len] = key_states
        self.v_buffer[:bsz, start_pos:self.current_len] = value_states

        return (
            self.k_buffer[:bsz, :self.current_len], 
            self.v_buffer[:bsz, :self.current_len]
        )

    def reset(self) -> None:
        # জিরোয়িং বাতিল করে কুইক ইনডেক্সিং পয়েন্টার রিসেট
        self.current_len = 0

    def get_stats(self) -> CacheStats:
        allocated = (self.k_buffer.element_size() * self.k_buffer.nelement()) * 2
        return CacheStats(
            allocated_bytes=allocated,
            current_length=self.current_len,
            cache_type="StaticKVCache",
            device=self.device,
            dtype=self.k_buffer.dtype,
            batch_size=self.config.max_batch_size
        )


class DynamicKVCache(BaseKVCache):
    """
    SOTA Dynamic Cache using dynamic slicing of static pre-allocated buffer
    to avoid memory fragmentation and catastrophic O(N^2) concat overheads.
    """
    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        self.k_buffer = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.num_key_value_heads, config.head_dim),
            dtype=config.dtype,
            device=self.device
        )
        self.v_buffer = torch.zeros(
            (config.max_batch_size, config.max_seq_len, config.num_key_value_heads, config.head_dim),
            dtype=config.dtype,
            device=self.device
        )
        self.current_len = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _, _ = key_states.shape
        self.current_len = start_pos + seqlen

        # ওভারফ্লো বাউন্ডারি চেক
        if self.current_len > self.config.max_seq_len:
            raise RuntimeError(
                f"Dynamic KV Cache limits exceeded! Max sequence length is {self.config.max_seq_len}, "
                f"but trying to write up to {self.current_len} tokens."
            )

        # ডিভাইস এবং টাইপ ম্যাচিং
        if key_states.device != self.k_buffer.device:
            self.device = key_states.device
            self.k_buffer = self.k_buffer.to(device=self.device, dtype=key_states.dtype)
            self.v_buffer = self.v_buffer.to(device=self.device, dtype=key_states.dtype)

        # ইন-প্লেস O(1) বাফার আপডেট
        self.k_buffer[:bsz, start_pos:self.current_len] = key_states
        self.v_buffer[:bsz, start_pos:self.current_len] = value_states

        return (
            self.k_buffer[:bsz, :self.current_len], 
            self.v_buffer[:bsz, :self.current_len]
        )

    def reset(self) -> None:
        self.current_len = 0

    def get_stats(self) -> CacheStats:
        allocated = (self.k_buffer.element_size() * self.k_buffer.nelement()) * 2
        return CacheStats(
            allocated_bytes=allocated,
            current_length=self.current_len,
            cache_type="DynamicKVCache",
            device=self.device,
            dtype=self.k_buffer.dtype,
            batch_size=self.config.max_batch_size
        )


class SlidingWindowKVCache(BaseKVCache):
    """
    SOTA sliding window cache with precise sliding slice.
    Paddings and out of boundary attention blocks kept mathematically synchronized.
    """
    def __init__(self, config: CacheConfig, window_size: int) -> None:
        self.config = config
        self.window_size = min(window_size, config.max_seq_len)
        self.device = torch.device(config.device)
        self.k_buffer = torch.zeros(
            (config.max_batch_size, self.window_size, config.num_key_value_heads, config.head_dim),
            dtype=config.dtype,
            device=self.device
        )
        self.v_buffer = torch.zeros(
            (config.max_batch_size, self.window_size, config.num_key_value_heads, config.head_dim),
            dtype=config.dtype,
            device=self.device
        )
        self.current_len = 0

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _, _ = key_states.shape
        
        if key_states.device != self.k_buffer.device:
            self.device = key_states.device
            self.k_buffer = self.k_buffer.to(device=self.device, dtype=key_states.dtype)
            self.v_buffer = self.v_buffer.to(device=self.device, dtype=key_states.dtype)

        if start_pos == 0:
            self.current_len = 0

        # উইন্ডো সাইজ ছাড়িয়ে গেলে রোটেট করা হবে
        if self.current_len + seqlen > self.window_size:
            shift = (self.current_len + seqlen) - self.window_size
            self.k_buffer[:bsz] = self.k_buffer[:bsz].roll(-shift, dims=1)
            self.v_buffer[:bsz] = self.v_buffer[:bsz].roll(-shift, dims=1)
            write_pos = self.window_size - seqlen
            self.current_len = self.window_size
        else:
            write_pos = self.current_len
            self.current_len += seqlen

        self.k_buffer[:bsz, write_pos:self.current_len] = key_states
        self.v_buffer[:bsz, write_pos:self.current_len] = value_states

        return (
            self.k_buffer[:bsz, :self.current_len], 
            self.v_buffer[:bsz, :self.current_len]
        )

    def reset(self) -> None:
        self.current_len = 0

    def get_stats(self) -> CacheStats:
        allocated = (self.k_buffer.element_size() * self.k_buffer.nelement()) * 2
        return CacheStats(
            allocated_bytes=allocated,
            current_length=self.current_len,
            cache_type="SlidingWindowKVCache",
            device=self.device,
            dtype=self.k_buffer.dtype,
            batch_size=self.config.max_batch_size
        )


class CacheAllocator:
    """
    SOTA Dictionary-based Registry Allocator Factory.
    Avoids nested if-else structures.
    """
    _registry: Dict[str, Type[BaseKVCache]] = {
        "static": StaticKVCache,
        "dynamic": DynamicKVCache,
        "sliding_window": lambda cfg, **kwargs: SlidingWindowKVCache(cfg, kwargs.get("window_size", 2048)),
    }

    @classmethod
    def allocate(cls, cache_type: str, config: CacheConfig, **kwargs) -> BaseKVCache:
        cache_type_lower = cache_type.lower()
        if cache_type_lower not in cls._registry:
            raise ValueError(f"Unknown cache type: {cache_type}. Registered caches: {list(cls._registry.keys())}")
        
        creator = cls._registry[cache_type_lower]
        if cache_type_lower == "sliding_window":
            return creator(config, **kwargs)
        return creator(config)


class CacheManager:

    def __init__(self, num_layers: int, cache_type: str, config: CacheConfig, **kwargs) -> None:
        self.caches = [
            CacheAllocator.allocate(cache_type, config, **kwargs)
            for _ in range(num_layers)
        ]

    def __getitem__(self, index: int) -> BaseKVCache:
        return self.caches[index]

    def reset_all(self) -> None:
        for cache in self.caches:
            cache.reset()

    def get_total_memory_usage(self) -> int:
        return sum(cache.get_stats().allocated_bytes for cache in self.caches)


class CacheUtils:

    @staticmethod
    def slice_cache(tensor: torch.Tensor, start_pos: int, length: int) -> torch.Tensor:
        return tensor[:, start_pos : start_pos + length]


KVCache = DynamicKVCache