"""CPU optimization configuration."""
import dataclasses
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Union

@dataclass
class ZortherCPUConfig:
    intra_op_num_threads: int = 8
    inter_op_num_threads: int = 2
    enable_mkl: bool = True
    enable_openmp: bool = True
    enable_avx2: bool = True
    pin_cores: bool = True
    cache_friendly_batching: bool = True
    torch_compile_enabled: bool = True
    torch_compile_backend: str = "inductor"
    torch_compile_mode: str = "max-autotune"
    memory_alignment_bytes: int = 64

    def __post_init__(self) -> None:
        if self.intra_op_num_threads <= 0 or self.intra_op_num_threads > 16:
            raise ValueError(f"intra_op_num_threads must be between 1 and 16, got: {self.intra_op_num_threads}")

        if self.inter_op_num_threads <= 0:
            raise ValueError(f"inter_op_num_threads must be greater than 0, got: {self.inter_op_num_threads}")

        if self.torch_compile_mode not in {"default", "reduce-overhead", "max-autotune"}:
            raise ValueError(f"Unsupported torch_compile_mode: {self.torch_compile_mode}")

        if self.memory_alignment_bytes not in {32, 64, 128}:
            raise ValueError(f"Memory alignment bytes must be 32, 64, or 128, got: {self.memory_alignment_bytes}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ZortherCPUConfig":
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ZortherCPUConfig":
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike]) -> "ZortherCPUConfig":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)