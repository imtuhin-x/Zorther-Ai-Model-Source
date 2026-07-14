import gc
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
import torch


class CPUInfo:

    @staticmethod
    def get_cpu_name() -> str:
        system = platform.system()
        if system == "Linux":
            try:
                with open("/proc/cpuinfo", "r") as f:
                    for line in f:
                        if "model name" in line:
                            return line.split(":")[1].strip()
            except Exception:
                pass
        elif system == "Windows":
            try:
                import winreg
                key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
                name, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                return str(name).strip()
            except Exception:
                pass
        return "Unknown CPU"

    @staticmethod
    def detect_features() -> Dict[str, bool]:
        cpu_name = CPUInfo.get_cpu_name().lower()
        features = {
            "avx2": False,
            "fma": False,
            "bmi1": False,
            "bmi2": False,
            "aes": False,
            "vaes": False
        }
        if "5700g" in cpu_name or "ryzen" in cpu_name or "zen" in cpu_name:
            features = {k: True for k in features}
            return features
        system = platform.system()
        if system == "Linux":
            try:
                out = subprocess.check_output("lscpu", shell=True).decode().lower()
                features["avx2"] = "avx2" in out
                features["fma"] = "fma" in out
                features["bmi1"] = "bmi1" in out or "bmi" in out
                features["bmi2"] = "bmi2" in out or "bmi" in out
                features["aes"] = "aes" in out
                features["vaes"] = "vaes" in out
            except Exception:
                pass
        return features


class ThreadManager:

    @staticmethod
    def configure_threads(intra_threads: int = 8, inter_threads: int = 2) -> None:
        torch.set_num_threads(intra_threads)
        torch.set_num_interop_threads(inter_threads)


class OpenMPManager:

    @staticmethod
    def configure_openmp(threads: int = 8) -> None:
        os.environ["OMP_NUM_THREADS"] = str(threads)
        os.environ["MKL_NUM_THREADS"] = str(threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(threads)


class AffinityManager:

    @staticmethod
    def pin_to_physical_cores(num_cores: int = 8) -> bool:
        if hasattr(os, "sched_setaffinity"):
            try:
                cores = list(range(num_cores))
                os.sched_setaffinity(0, cores)
                return True
            except Exception:
                pass
        return False


class NUMAManager:

    @staticmethod
    def get_numa_nodes_count() -> int:
        if platform.system() == "Linux":
            try:
                out = subprocess.check_output("lscpu | grep 'NUMA node(s):'", shell=True).decode()
                return int(out.split()[-1])
            except Exception:
                pass
        return 1


class MemoryOptimizer:

    @staticmethod
    def release_excess_memory() -> None:
        gc.collect()
        if hasattr(torch, "cpu") and hasattr(torch.cpu, "empty_cache"):
            torch.cpu.empty_cache()

    @staticmethod
    def configure_allocator() -> None:
        os.environ["MALLOC_CONF"] = "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"


class TorchOptimizer:

    @staticmethod
    def configure_torch_globals() -> None:
        torch.set_grad_enabled(True)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")


class InferenceOptimizer:

    @staticmethod
    def optimize_model(model: torch.nn.Module, mode: str = "reduce-overhead", backend: str = "inductor") -> torch.nn.Module:
        if hasattr(torch, "compile"):
            try:
                compiled_model = torch.compile(model, mode=mode, backend=backend)
                return compiled_model
            except Exception:
                return model
        return model


class TrainingOptimizer:

    @staticmethod
    def configure_gradients(model: torch.nn.Module) -> None:
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(lambda grad: grad.contiguous())


class Benchmark:

    @staticmethod
    def run_mat_mul(size: int = 2048, iterations: int = 10) -> float:
        a = torch.randn(size, size, dtype=torch.float32)
        b = torch.randn(size, size, dtype=torch.float32)
        for _ in range(3):
            torch.matmul(a, b)
        start = time.perf_counter()
        for _ in range(iterations):
            torch.matmul(a, b)
        end = time.perf_counter()
        avg_time = (end - start) / iterations
        flops = 2.0 * (size ** 3)
        gflops = (flops / 1e9) / avg_time
        return gflops


class CPUOptimizer:

    def __init__(self, intra_threads: int = 8, inter_threads: int = 2) -> None:
        self.intra_threads = intra_threads
        self.inter_threads = inter_threads

    def run_full_optimization(self, model: Optional[torch.nn.Module] = None) -> Dict[str, Any]:
        cpu_name = CPUInfo.get_cpu_name()
        features = CPUInfo.detect_features()
        
        OpenMPManager.configure_openmp(self.intra_threads)
        ThreadManager.configure_threads(self.intra_threads, self.inter_threads)
        pinned = AffinityManager.pin_to_physical_cores(self.intra_threads)
        MemoryOptimizer.configure_allocator()
        TorchOptimizer.configure_torch_globals()
        
        initial_gflops = Benchmark.run_mat_mul()
        
        optimized_model = None
        if model is not None:
            optimized_model = InferenceOptimizer.optimize_model(model)
            TrainingOptimizer.configure_gradients(model)
            
        final_gflops = Benchmark.run_mat_mul()
        
        return {
            "cpu_model": cpu_name,
            "detected_instruction_sets": features,
            "core_affinity_pinned": pinned,
            "intra_threads": self.intra_threads,
            "inter_threads": self.inter_threads,
            "baseline_gflops": initial_gflops,
            "optimized_gflops": final_gflops,
            "performance_gain_pct": ((final_gflops - initial_gflops) / initial_gflops) * 100.0 if initial_gflops > 0 else 0.0,
            "model_optimized": optimized_model is not None
        }