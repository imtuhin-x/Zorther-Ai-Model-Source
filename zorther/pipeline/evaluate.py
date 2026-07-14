"""Evaluation pipeline."""
import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import platform

@dataclass
class EvaluationConfig:
    batch_size: int = 4
    max_eval_steps: int = 100
    benchmark_iterations: int = 5
    device: str = "cpu"
    output_report_path: str = "evaluation_report.json"


@dataclass
class EvaluationReport:
    average_loss: float = 0.0
    perplexity: float = 0.0
    token_accuracy: float = 0.0
    prefill_tokens_per_sec: float = 0.0
    decode_tokens_per_sec: float = 0.0
    peak_ram_usage_mb: float = 0.0
    evaluated_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)


class LossEvaluator:

    @staticmethod
    @torch.no_grad()
    def evaluate(model: nn.Module, dataloader: DataLoader, max_steps: int, device: torch.device) -> float:
        model.eval()
        total_loss = 0.0
        steps = 0
        
        for batch in dataloader:
            if steps >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            
            logits = model(input_ids)
            loss = F_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
                targets.view(-1),
                ignore_index=-100
            )
            total_loss += loss.item()
            steps += 1
            
        return total_loss / steps if steps > 0 else float("inf")


class PerplexityEvaluator:

    @staticmethod
    def calculate(loss_val: float) -> float:
        try:
            return math.exp(min(loss_val, 20.0))
        except OverflowError:
            return float("inf")


class AccuracyEvaluator:

    @staticmethod
    @torch.no_grad()
    def evaluate(model: nn.Module, dataloader: DataLoader, max_steps: int, device: torch.device) -> float:
        model.eval()
        correct_tokens = 0
        total_tokens = 0
        steps = 0
        
        for batch in dataloader:
            if steps >= max_steps:
                break
            input_ids = batch["input_ids"].to(device)
            targets = batch["targets"].to(device)
            
            logits = model(input_ids)
            preds = torch.argmax(logits, dim=-1)
            
            mask = targets != -100
            correct_tokens += torch.sum((preds == targets) & mask).item()
            total_tokens += torch.sum(mask).item()
            steps += 1
            
        return correct_tokens / total_tokens if total_tokens > 0 else 0.0


class SpeedBenchmark:

    @staticmethod
    def run(model: nn.Module, iterations: int, device: torch.device) -> Tuple[float, float]:
        model.eval()
        vocab_size = model.config.vocab_size
        
        prefill_input = torch.randint(0, vocab_size, (1, 512), dtype=torch.long, device=device)
        start = time.perf_counter()
        for _ in range(iterations):
            with torch.no_grad():
                model(prefill_input)
        prefill_elapsed = time.perf_counter() - start
        prefill_tps = (512 * iterations) / prefill_elapsed
        
        decode_input = torch.randint(0, vocab_size, (1, 1), dtype=torch.long, device=device)
        start = time.perf_counter()
        for _ in range(iterations * 50):
            with torch.no_grad():
                model(decode_input)
        decode_elapsed = time.perf_counter() - start
        decode_tps = (iterations * 50) / decode_elapsed
        
        return prefill_tps, decode_tps


class MemoryBenchmark:

    @staticmethod
    def get_peak_ram() -> float:
        try:
            if platform.system() == "Windows":
                import ctypes
                class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                    _fields_ = [
                        ("cb", ctypes.c_ulong),
                        ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)
                    ]
                get_current_process = ctypes.windll.kernel32.GetCurrentProcess
                get_process_memory_info = ctypes.windll.psapi.GetProcessMemoryInfo
                ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
                
                counters = PROCESS_MEMORY_COUNTERS()
                counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
                get_process_memory_info(get_current_process(), ctypes.byref(counters), counters.cb)
                return counters.PeakWorkingSetSize / (1024 * 1024)
            else:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        except Exception:
            return 0.0


class MetricsAggregator:

    @staticmethod
    def aggregate(loss: float, ppl: float, acc: float, prefill_speed: float, decode_speed: float, ram: float) -> EvaluationReport:
        return EvaluationReport(
            average_loss=loss,
            perplexity=ppl,
            token_accuracy=acc,
            prefill_tokens_per_sec=prefill_speed,
            decode_tokens_per_sec=decode_speed,
            peak_ram_usage_mb=ram
        )


class EvaluationManager:

    def __init__(self, model: nn.Module, config: EvaluationConfig) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

    def evaluate(self, val_loader: DataLoader) -> EvaluationReport:
        loss = LossEvaluator.evaluate(self.model, val_loader, self.config.max_eval_steps, self.device)
        ppl = PerplexityEvaluator.calculate(loss)
        acc = AccuracyEvaluator.evaluate(self.model, val_loader, self.config.max_eval_steps, self.device)
        prefill_tps, decode_tps = SpeedBenchmark.run(self.model, self.config.benchmark_iterations, self.device)
        peak_ram = MemoryBenchmark.get_peak_ram()
        
        report = MetricsAggregator.aggregate(loss, ppl, acc, prefill_tps, decode_tps, peak_ram)
        report.save(self.config.output_report_path)
        return report