"""Model architecture configuration."""
import json
import os
import dataclasses
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Union

@dataclass
class ZortherModelConfig:
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_seq_len: int = 4096
    rope_theta: float = 10000.0
    ffn_dim_multiplier: float = 1.0
    multiple_of: int = 256
    bias: bool = False
    dropout: float = 0.0
    norm_eps: float = 1e-5
    activation: str = "swiglu"
    weight_tying: bool = False
    intermediate_size: int = field(init=False)

    def __post_init__(self) -> None:
        if self.activation not in {"swiglu", "geglu"}:
            raise ValueError(f"Activation must be either 'swiglu' or 'geglu', got: {self.activation}")

        hidden_dim = int(2 * (4 * self.hidden_size) / 3)
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        self.intermediate_size = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ZortherModelConfig":
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ZortherModelConfig":
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike]) -> "ZortherModelConfig":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)