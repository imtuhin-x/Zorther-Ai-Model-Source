"""Training configuration."""
import dataclasses
import json
import os
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Union

@dataclass
class ZortherTrainingConfig:
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    total_steps: int = 100000
    warmup_steps: int = 2000
    lr_scheduler_type: str = "cosine"
    label_smoothing: float = 0.1
    mixed_precision: str = "fp32"
    seed: int = 42
    eval_interval: int = 500
    eval_steps: int = 100
    save_interval: int = 1000
    checkpoint_dir: str = "checkpoints"
    logging_dir: str = "logs"
    use_tensorboard: bool = True
    use_csv: bool = True
    resume_from_checkpoint: Optional[str] = None
    early_stopping_patience: int = 5

    def __post_init__(self) -> None:
        if self.mixed_precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError(f"Mixed precision must be 'fp32', 'fp16', or 'bf16', got: {self.mixed_precision}")

        if self.lr_scheduler_type not in {"cosine", "linear", "constant"}:
            raise ValueError(f"Scheduler must be 'cosine', 'linear', or 'constant', got: {self.lr_scheduler_type}")

        if self.batch_size <= 0:
            raise ValueError(f"Batch size must be greater than 0, got: {self.batch_size}")

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(f"Gradient accumulation steps must be greater than 0, got: {self.gradient_accumulation_steps}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ZortherTrainingConfig":
        valid_keys = {f.name for f in dataclasses.fields(cls) if f.init}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

    @classmethod
    def from_json(cls, json_str: str) -> "ZortherTrainingConfig":
        return cls.from_dict(json.loads(json_str))

    def save(self, filepath: Union[str, os.PathLike]) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: Union[str, os.PathLike]) -> "ZortherTrainingConfig":
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found at: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)