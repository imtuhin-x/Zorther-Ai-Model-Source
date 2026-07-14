"""Loss functions."""
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LossConfig:
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    ignore_index: int = -100
    temperature: float = 2.0
    huber_delta: float = 1.0


class BaseLoss(ABC):

    @abstractmethod
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pass


class CrossEntropyLoss(BaseLoss):

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=self.config.ignore_index
        )


class LabelSmoothingLoss(BaseLoss):

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            targets.view(-1),
            ignore_index=self.config.ignore_index,
            label_smoothing=self.config.label_smoothing
        )


class FocalLoss(BaseLoss):

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits_flat = logits.view(-1, logits.shape[-1])
        targets_flat = targets.view(-1)

        mask = targets_flat != self.config.ignore_index
        logits_flat = logits_flat[mask]
        targets_flat = targets_flat[mask]

        if targets_flat.numel() == 0:
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        log_pt = F.log_softmax(logits_flat, dim=-1)
        pt = torch.exp(log_pt)
        
        log_pt_target = log_pt.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)
        pt_target = pt.gather(dim=-1, index=targets_flat.unsqueeze(-1)).squeeze(-1)

        focal_weight = (1.0 - pt_target) ** self.config.focal_gamma
        loss = -focal_weight * log_pt_target
        return loss.mean()


class KLDivLoss(BaseLoss):

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        p = F.log_softmax(logits / self.config.temperature, dim=-1)
        q = F.softmax(teacher_logits / self.config.temperature, dim=-1)
        
        loss = F.kl_div(p, q, reduction="batchmean") * (self.config.temperature ** 2)
        return loss


class HuberLoss(BaseLoss):

    def __init__(self, config: LossConfig) -> None:
        self.config = config

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        diff = torch.abs(logits - targets)
        delta = self.config.huber_delta
        
        mask = diff < delta
        loss = torch.where(mask, 0.5 * (diff ** 2), delta * (diff - 0.5 * delta))
        return loss.mean()


class LossFactory:

    @staticmethod
    def create(loss_type: str, config: LossConfig) -> BaseLoss:
        lt = loss_type.lower()
        if lt == "cross_entropy":
            return CrossEntropyLoss(config)
        elif lt == "label_smoothing":
            return LabelSmoothingLoss(config)
        elif lt == "focal":
            return FocalLoss(config)
        elif lt == "huber":
            return HuberLoss(config)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")


class TokenMask:

    @staticmethod
    def generate(targets: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        return (targets != ignore_index).to(torch.float32)


class IgnoreIndexHandler:

    @staticmethod
    def filter_active(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> Tuple[torch.Tensor, torch.Tensor]:
        flat_logits = logits.view(-1, logits.shape[-1])
        flat_targets = targets.view(-1)
        active_mask = flat_targets != ignore_index
        return flat_logits[active_mask], flat_targets[active_mask]


class LossScaler:

    def __init__(self, scale_factor: float = 1.0) -> None:
        self.scale_factor = scale_factor

    def scale(self, loss: torch.Tensor) -> torch.Tensor:
        return loss * self.scale_factor


class Perplexity:

    @staticmethod
    def calculate(loss_value: float) -> float:
        try:
            return math.exp(min(loss_value, 20.0))
        except OverflowError:
            return float("inf")


class LossMetrics:

    def __init__(self) -> None:
        self.reset()

    def update(self, loss_val: float) -> None:
        self.total_loss += loss_val
        self.step_count += 1

    def get_avg_loss(self) -> float:
        return self.total_loss / self.step_count if self.step_count > 0 else 0.0

    def get_ppl(self) -> float:
        return Perplexity.calculate(self.get_avg_loss())

    def reset(self) -> None:
        self.total_loss = 0.0
        self.step_count = 0


class LossManager:

    def __init__(self, config: LossConfig, default_loss_type: str = "cross_entropy") -> None:
        self.config = config
        self.loss_fn = LossFactory.create(default_loss_type, config)
        self.metrics = LossMetrics()

    def compute(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.loss_fn(logits, targets)
        self.metrics.update(loss.item())
        return loss

    def get_stats(self) -> Dict[str, float]:
        return {
            "average_loss": self.metrics.get_avg_loss(),
            "perplexity": self.metrics.get_ppl()
        }

    def reset_stats(self) -> None:
        self.metrics.reset()