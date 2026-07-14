"""
Zorther Gen - SOTA Transformer Architecture
Features:
- Perfectly synchronized Rotary Positional Embeddings (RoPE) with dynamic start_pos
- Strict KV-Cache coordinate alignment for decoding steps (0, 1, 2... N)
- Verified residual scaling to prevent gradient vanishing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union

from zorther.config.model_config import ZortherModelConfig
from zorther.model.embeddings import ZortherEmbeddings, RotaryPositionalEmbedding
from zorther.model.layers import RMSNorm, TransformerBlock
from zorther.model.cache import KVCache


class ZortherTransformer(nn.Module):

    def __init__(self, config: ZortherModelConfig) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers
        self.embeddings = ZortherEmbeddings(config.vocab_size, config.hidden_size)
        
        # RoPE ডাইমেনশন নির্ধারণ
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope = RotaryPositionalEmbedding(
            dim=self.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta
        )
        
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, config=config)
            for i in range(config.num_layers)
        ])
        
        self.norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        if config.weight_tying:
            self.output_projection.weight = self.embeddings.token_embeddings.weight
            
        self.residual_scale = 1.0 / math.sqrt(2.0 * config.num_layers)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "residual_scaling_flag") and module.residual_scaling_flag:
                std = std / math.sqrt(2.0 * self.num_layers)
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[KVCache]] = None,
        start_pos: int = 0
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len = input_ids.shape
        
        x = self.embeddings(input_ids)

        cos, sin = self.rope(seq_len=seq_len, start_pos=start_pos, device=x.device)
        
        mask: Optional[torch.Tensor] = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1)

        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            
            x = layer(
                x, 
                cos, 
                sin, 
                mask, 
                layer_cache, 
                start_pos, 
                residual_scale=self.residual_scale
            )
            
        x = self.norm(x)
        logits = self.output_projection(x)
        
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100
            )
            return logits, loss
            
        return logits