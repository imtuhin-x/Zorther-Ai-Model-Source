"""Training pipeline."""
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
from torch.utils.data import DataLoader

from zorther.config.model_config import ZortherModelConfig
from zorther.config.training_config import ZortherTrainingConfig
from zorther.optimization.optimizer import OptimizerManager, OptimizerConfig
from zorther.optimization.loss import LossManager, LossConfig
import math

@dataclass
class TrainerConfig:
    epochs: int = 3
    gradient_accumulation_steps: int = 8
    mixed_precision: str = "fp32"
    max_grad_norm: float = 1.0
    eval_interval: int = 500
    save_interval: int = 1000
    early_stopping_patience: int = 5
    checkpoint_dir: str = "checkpoints"
    device: str = "cpu"


@dataclass
class TrainState:
    epoch: int = 0
    step: int = 0
    global_step: int = 0
    best_val_loss: float = float("inf")
    early_stopping_counter: int = 0
    is_terminated: bool = False


class BatchProcessor:

    @staticmethod
    def to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
        return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


class MixedPrecisionManager:

    def __init__(self, precision_mode: str) -> None:
        self.precision_mode = precision_mode.lower()
        self.enabled = self.precision_mode in {"fp16", "bf16"}
        self.dtype = torch.bfloat16 if self.precision_mode == "bf16" else (torch.float16 if self.precision_mode == "fp16" else torch.float32)

    def get_autocast_context(self, device_type: str = "cpu") -> torch.amp.autocast:
        return torch.amp.autocast(device_type=device_type, dtype=self.dtype, enabled=self.enabled)


class MetricsTracker:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.running_loss = 0.0
        self.total_tokens = 0
        self.start_time = time.perf_counter()
        self.steps = 0

    def update(self, loss_val: float, num_tokens: int) -> None:
        self.running_loss += loss_val
        self.total_tokens += num_tokens
        self.steps += 1

    def get_metrics(self) -> Dict[str, float]:
        elapsed = time.perf_counter() - self.start_time
        avg_loss = self.running_loss / self.steps if self.steps > 0 else 0.0
        tps = self.total_tokens / elapsed if elapsed > 0 else 0.0
        return {
            "average_loss": avg_loss,
            "tokens_per_second": tps,
            "perplexity": math.exp(min(avg_loss, 20.0)) if avg_loss > 0 else float("inf")
        }


class EarlyStopping:

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience

    def check(self, state: TrainState, current_val_loss: float) -> bool:
        if current_val_loss < state.best_val_loss:
            state.best_val_loss = current_val_loss
            state.early_stopping_counter = 0
            return False
        else:
            state.early_stopping_counter += 1
            if state.early_stopping_counter >= self.patience:
                state.is_terminated = True
                return True
        return False


class Callback:

    def on_train_begin(self, state: TrainState) -> None:
        pass

    def on_train_end(self, state: TrainState) -> None:
        pass

    def on_epoch_begin(self, state: TrainState) -> None:
        pass

    def on_epoch_end(self, state: TrainState, metrics: Dict[str, float]) -> None:
        pass

    def on_step_begin(self, state: TrainState) -> None:
        pass

    def on_step_end(self, state: TrainState, loss: float) -> None:
        pass


class CallbackManager:

    def __init__(self, callbacks: Optional[List[Callback]] = None) -> None:
        self.callbacks = callbacks if callbacks is not None else []

    def add_callback(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_begin(self, state: TrainState) -> None:
        for c in self.callbacks:
            c.on_train_begin(state)

    def on_train_end(self, state: TrainState) -> None:
        for c in self.callbacks:
            c.on_train_end(state)

    def on_epoch_begin(self, state: TrainState) -> None:
        for c in self.callbacks:
            c.on_epoch_begin(state)

    def on_epoch_end(self, state: TrainState, metrics: Dict[str, float]) -> None:
        for c in self.callbacks:
            c.on_epoch_end(state, metrics)

    def on_step_begin(self, state: TrainState) -> None:
        for c in self.callbacks:
            c.on_step_begin(state)

    def on_step_end(self, state: TrainState, loss: float) -> None:
        for c in self.callbacks:
            c.on_step_end(state, loss)


class Trainer:

    def __init__(
        self,
        model: torch.nn.Module,
        train_config: TrainerConfig,
        opt_config: OptimizerConfig,
        loss_config: LossConfig,
        callbacks: Optional[List[Callback]] = None
    ) -> None:
        self.model = model
        self.config = train_config
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        self.amp_manager = MixedPrecisionManager(train_config.mixed_precision)
        self.optimizer_manager = OptimizerManager(model, opt_config, train_config.gradient_accumulation_steps)
        self.loss_manager = LossManager(loss_config)
        self.early_stopping = EarlyStopping(train_config.early_stopping_patience)
        self.callback_manager = CallbackManager(callbacks)
        self.state = TrainState()
        self.tracker = MetricsTracker()

    def _train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        batch = BatchProcessor.to_device(batch, self.device)
        input_ids = batch["input_ids"]
        targets = batch["targets"]

        with self.amp_manager.get_autocast_context(self.device.type):
            logits = self.model(input_ids)
            loss = self.loss_manager.compute(logits, targets)

        opt_state = self.optimizer_manager.step(loss)
        num_tokens = input_ids.numel()
        self.tracker.update(loss.item(), num_tokens)

        return loss.item()

    @torch.no_grad()
    def _validation_step(self, val_loader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        steps = 0

        for batch in val_loader:
            batch = BatchProcessor.to_device(batch, self.device)
            input_ids = batch["input_ids"]
            targets = batch["targets"]

            with self.amp_manager.get_autocast_context(self.device.type):
                logits = self.model(input_ids)
                loss = self.loss_manager.compute(logits, targets)
                total_loss += loss.item()
                steps += 1

        self.model.train()
        return total_loss / steps if steps > 0 else float("inf")

    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None) -> TrainState:
        self.callback_manager.on_train_begin(self.state)
        self.model.train()

        for epoch in range(self.config.epochs):
            self.state.epoch = epoch
            self.callback_manager.on_epoch_begin(self.state)
            self.tracker.reset()

            for step, batch in enumerate(train_loader):
                self.state.step = step
                self.callback_manager.on_step_begin(self.state)

                loss_val = self._train_step(batch)
                self.state.global_step += 1

                self.callback_manager.on_step_end(self.state, loss_val)

                if self.state.global_step % self.config.eval_interval == 0 and val_loader is not None:
                    val_loss = self._validation_step(val_loader)
                    stop_triggered = self.early_stopping.check(self.state, val_loss)
                    
                    if stop_triggered:
                        break

                if self.state.global_step % self.config.save_interval == 0:
                    self._save_checkpoint()

            if self.state.is_terminated:
                break

            epoch_metrics = self.tracker.get_metrics()
            self.callback_manager.on_epoch_end(self.state, epoch_metrics)

        self.callback_manager.on_train_end(self.state)
        return self.state

    def _save_checkpoint(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.state.global_step}.pt")
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer_manager.optimizer.state_dict(),
            "scheduler_state_dict": self.optimizer_manager.scheduler.optimizer.state_dict(),
            "state": self.state
        }
        torch.save(checkpoint, filepath)