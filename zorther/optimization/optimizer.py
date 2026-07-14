"""Optimizer."""
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import torch
from torch.optim import Optimizer


@dataclass
class OptimizerConfig:
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    optimizer_type: str = "adamw"
    scheduler_type: str = "cosine"
    warmup_steps: int = 2000
    total_steps: int = 100000
    max_grad_norm: float = 1.0
    ema_decay: float = 0.999
    use_ema: bool = True
    sophia_rho: float = 0.04


class Lion(Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 1e-4, betas: Tuple[float, float] = (0.9, 0.99), weight_decay: float = 0.0) -> None:
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                
                if group["weight_decay"] != 0:
                    p.mul_(1.0 - group["lr"] * group["weight_decay"])

                update = exp_avg * beta1 + grad * (1.0 - beta1)
                p.add_(torch.sign(update), alpha=-group["lr"])

                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss


class SophiaG(Optimizer):

    def __init__(self, params: Iterable[torch.nn.Parameter], lr: float = 3e-4, betas: Tuple[float, float] = (0.96, 0.99), rho: float = 0.04, eps: float = 1e-12, weight_decay: float = 0.1) -> None:
        defaults = dict(lr=lr, betas=betas, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def update_hessian(self, bs: int = 1) -> None:
        for group in self.param_groups:
            beta2 = group["betas"][1]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "hessian" not in state:
                    state["hessian"] = torch.zeros_like(p)
                
                hessian = state["hessian"]
                hat_h = p.grad.abs() * bs
                hessian.mul_(beta2).add_(hat_h, alpha=1.0 - beta2)

    @torch.no_grad()
    def step(self, closure: Optional[Any] = None) -> Optional[float]:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, _ = group["betas"]
            rho = group["rho"]
            eps = group["eps"]
            lr = group["lr"]
            wd = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["hessian"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                hessian = state["hessian"]

                if wd != 0:
                    p.mul_(1.0 - lr * wd)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)

                denom = torch.max(hessian, torch.full_like(hessian, eps))
                ratio = exp_avg / denom
                clipped_ratio = torch.clamp(ratio, -rho, rho)

                p.add_(clipped_ratio, alpha=-lr)

        return loss


class OptimizerFactory:

    @staticmethod
    def create(config: OptimizerConfig, model_params: Iterable[torch.nn.Parameter]) -> Optimizer:
        opt_type = config.optimizer_type.lower()
        if opt_type == "adamw":
            return torch.optim.AdamW(
                model_params,
                lr=config.learning_rate,
                betas=(config.adam_beta1, config.adam_beta2),
                eps=config.adam_epsilon,
                weight_decay=config.weight_decay
            )
        elif opt_type == "sgd":
            return torch.optim.SGD(model_params, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=0.9)
        elif opt_type == "lion":
            return Lion(model_params, lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), weight_decay=config.weight_decay)
        elif opt_type == "sophiag":
            return SophiaG(model_params, lr=config.learning_rate, betas=(config.adam_beta1, config.adam_beta2), rho=config.sophia_rho, weight_decay=config.weight_decay)
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")


class BaseScheduler:

    def __init__(self, optimizer: Optimizer, config: OptimizerConfig) -> None:
        self.optimizer = optimizer
        self.config = config
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr

    def get_lr(self) -> float:
        raise NotImplementedError


class WarmupScheduler(BaseScheduler):

    def get_lr(self) -> float:
        if self.current_step < self.config.warmup_steps:
            return self.config.learning_rate * float(self.current_step) / float(max(1, self.config.warmup_steps))
        return self.config.learning_rate


class CosineScheduler(BaseScheduler):

    def get_lr(self) -> float:
        if self.current_step < self.config.warmup_steps:
            return self.config.learning_rate * float(self.current_step) / float(max(1, self.config.warmup_steps))
        
        progress = float(self.current_step - self.config.warmup_steps) / float(max(1, self.config.total_steps - self.config.warmup_steps))
        if progress >= 1.0:
            return 0.0
            
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.learning_rate * cosine_decay


class CosineRestartScheduler(BaseScheduler):

    def __init__(self, optimizer: Optimizer, config: OptimizerConfig, restart_steps: int = 20000) -> None:
        super().__init__(optimizer, config)
        self.restart_steps = restart_steps

    def get_lr(self) -> float:
        if self.current_step < self.config.warmup_steps:
            return self.config.learning_rate * float(self.current_step) / float(max(1, self.config.warmup_steps))
            
        step_in_cycle = (self.current_step - self.config.warmup_steps) % self.restart_steps
        progress = float(step_in_cycle) / float(self.restart_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.learning_rate * cosine_decay


class LinearScheduler(BaseScheduler):

    def get_lr(self) -> float:
        if self.current_step < self.config.warmup_steps:
            return self.config.learning_rate * float(self.current_step) / float(max(1, self.config.warmup_steps))
            
        total_decay_steps = max(1, self.config.total_steps - self.config.warmup_steps)
        step_decay = float(self.current_step - self.config.warmup_steps)
        progress = step_decay / total_decay_steps
        return max(0.0, self.config.learning_rate * (1.0 - progress))


class ConstantScheduler(BaseScheduler):

    def get_lr(self) -> float:
        return self.config.learning_rate


class PolynomialScheduler(BaseScheduler):

    def get_lr(self) -> float:
        if self.current_step < self.config.warmup_steps:
            return self.config.learning_rate * float(self.current_step) / float(max(1, self.config.warmup_steps))
            
        progress = float(self.current_step - self.config.warmup_steps) / float(max(1, self.config.total_steps - self.config.warmup_steps))
        return self.config.learning_rate * ((1.0 - progress) ** 2)


class SchedulerFactory:

    @staticmethod
    def create(optimizer: Optimizer, config: OptimizerConfig) -> BaseScheduler:
        sch_type = config.scheduler_type.lower()
        if sch_type == "warmup":
            return WarmupScheduler(optimizer, config)
        elif sch_type == "cosine":
            return CosineScheduler(optimizer, config)
        elif sch_type == "cosine_restart":
            return CosineRestartScheduler(optimizer, config)
        elif sch_type == "linear":
            return LinearScheduler(optimizer, config)
        elif sch_type == "constant":
            return ConstantScheduler(optimizer, config)
        elif sch_type == "polynomial":
            return PolynomialScheduler(optimizer, config)
        else:
            raise ValueError(f"Unknown scheduler type: {sch_type}")


class GradientClipper:

    @staticmethod
    def clip(parameters: Iterable[torch.nn.Parameter], max_norm: float) -> float:
        if max_norm <= 0.0:
            return 0.0
        return torch.nn.utils.clip_grad_norm_(parameters, max_norm).item()


class EMA:

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.model = model
        self.decay = decay
        self.shadow_params: Dict[str, torch.Tensor] = {}
        self.backup_params: Dict[str, torch.Tensor] = {}
        self.register()

    def register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                shadow = self.shadow_params[name]
                shadow.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])

    def restore_backup(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_params[name])
        self.backup_params.clear()


class GradientAccumulator:

    def __init__(self, steps: int = 1) -> None:
        self.steps = steps
        self.current_step = 0

    def should_step(self) -> bool:
        self.current_step += 1
        if self.current_step % self.steps == 0:
            return True
        return False

    def reset(self) -> None:
        self.current_step = 0


@dataclass
class OptimizerState:
    step_count: int = 0
    current_loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0


class OptimizerManager:

    def __init__(self, model: torch.nn.Module, config: OptimizerConfig, accumulation_steps: int = 1) -> None:
        self.model = model
        self.config = config
        self.optimizer = OptimizerFactory.create(config, model.parameters())
        self.scheduler = SchedulerFactory.create(self.optimizer, config)
        self.accumulator = GradientAccumulator(accumulation_steps)
        self.ema = EMA(model, config.ema_decay) if config.use_ema else None
        self.state = OptimizerState()

    def step(self, loss: torch.Tensor) -> OptimizerState:
        scaled_loss = loss / self.accumulator.steps
        scaled_loss.backward()
        
        self.state.current_loss = loss.item()

        if self.accumulator.should_step():
            grad_norm = GradientClipper.clip(self.model.parameters(), self.config.max_grad_norm)
            
            if isinstance(self.optimizer, SophiaG):
                self.optimizer.update_hessian(bs=self.accumulator.steps)
                
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            
            lr = self.scheduler.step()
            
            if self.ema is not None:
                self.ema.update()
                
            self.state.step_count += 1
            self.state.learning_rate = lr
            self.state.gradient_norm = grad_norm

        return self.state