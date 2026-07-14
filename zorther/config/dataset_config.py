"""Dataset configuration."""
import dataclasses
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Union

@dataclass
class ZortherDatasetConfig:
    dataset_root: str = r"Z:\File\Zorther Model\dataset"
    packing_length: int = 4096
    min_character_length: int = 10
    deduplication_threshold: float = 0.85
    sampling_ratios: Dict[str, float] = field(default_factory=lambda: {
        "conversation": 1.0,
    })
    shuffle_buffer_size: int = 10000
    num_workers: int = 8
    seed: int = 42
    streaming: bool = True
    quality_filter_enabled: bool = True

    def __post_init__(self) -> None:
        if self.packing_length <= 0:
            raise ValueError(f"Packing length must be greater than 0, got: {self.packing_length}")

        if self.deduplication_threshold < 0.0 or self.deduplication_threshold > 1.0:
            raise ValueError(f"Deduplication threshold must be between 0.0 and 1.0, got: {self.deduplication_threshold}")

        if not self.sampling_ratios:
            raise ValueError("Sampling ratios dictionary cannot be empty.")

        total_ratio = sum(self.sampling_ratios.values())
        if not (0.99 <= total_ratio <= 1.01):
            raise ValueError(f"Sum of sampling ratios must be approximately 1.0, got: {total_ratio}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ZortherDatasetConfig":
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ZortherDatasetConfig":
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike]) -> "ZortherDatasetConfig":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)