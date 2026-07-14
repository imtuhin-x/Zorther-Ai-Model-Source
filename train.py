"""
Zorther Gen - SOTA LLM Pre-training & SFT Enterprise Pipeline
Developed with deep neural diagnostic tools, real-time throughput metrics,
and PyTorch 2.6+ secure auto-resume configurations.
"""

import json
import glob
import argparse
import os
import random
import re
import math
import time
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from zorther.config.model_config import ZortherModelConfig
from zorther.config.training_config import ZortherTrainingConfig
from zorther.config.dataset_config import ZortherDatasetConfig
from zorther.config.cpu_config import ZortherCPUConfig

from zorther.tokenizer.bpe_tokenizer import BPETokenizer
from zorther.data.dataset import ZortherDataset
from zorther.data.dataloader import ZortherDataLoader
from zorther.model.transformer import ZortherTransformer
from zorther.optimization.cpu import CPUOptimizer
from zorther.optimization.optimizer import OptimizerConfig, OptimizerManager
from zorther.optimization.loss import LossConfig, LossManager
from zorther.pipeline.train import TrainerConfig, TrainState
from zorther.utils.logger import LoggerConfig, LoggerManager
from zorther.utils.checkpoint import CheckpointConfig, CheckpointManager

# PyTorch 2.6+ আনপিকলিং সিকিউরিটি বাইপাস কনফিগারেশন
torch.serialization.add_safe_globals([TrainState])


class ArgumentParser:
    """মেইন ট্রেইনিং স্ক্রিপ্টের আর্গুমেন্ট পার্সার ক্লাস।"""
    @staticmethod
    def parse() -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="Zorther Gen Advanced SFT Pipeline")
        parser.add_argument("--model_config", type=str, default=None, help="Path to model config JSON")
        parser.add_argument("--training_config", type=str, default=None, help="Path to training config JSON")
        parser.add_argument("--dataset_config", type=str, default=None, help="Path to dataset config JSON")
        parser.add_argument("--cpu_config", type=str, default=None, help="Path to CPU config JSON")
        parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to tokenizer json")
        parser.add_argument("--resume_step", type=int, default=None, help="Step to resume training from")
        parser.add_argument("--early_stopping_patience", type=int, default=10, help="Patience for early stopping")
        return parser.parse_args()


class ConfigLoader:
    """মডেলের কাস্টম ফোল্ডার স্ট্রাকচার এবং কনফিগারেশন লোডার।"""
    @staticmethod
    def load_all(args: argparse.Namespace) -> Tuple[ZortherModelConfig, ZortherTrainingConfig, ZortherDatasetConfig, ZortherCPUConfig]:
        model_cfg = ZortherModelConfig.load(args.model_config) if args.model_config else ZortherModelConfig()
        train_cfg = ZortherTrainingConfig.load(args.training_config) if args.training_config else ZortherTrainingConfig()
        data_cfg = ZortherDatasetConfig.load(args.dataset_config) if args.dataset_config else ZortherDatasetConfig()
        cpu_cfg = ZortherCPUConfig.load(args.cpu_config) if args.cpu_config else ZortherCPUConfig()
        return model_cfg, train_cfg, data_cfg, cpu_cfg


class SeedManager:
    """মডেলের গাণিতিক স্থায়িত্বের জন্য সিড ম্যানেজার ক্লাস এবং র্যান্ডম স্টেট পুনরুদ্ধারের সুবিধা।"""
    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def save_state() -> Dict[str, Any]:
        return {
            "random": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        }

    @staticmethod
    def load_state(states: Dict[str, Any]) -> None:
        if "random" in states:
            random.setstate(states["random"])
        if "numpy" in states:
            np.random.set_state(states["numpy"])
        if "torch" in states:
            torch.set_rng_state(states["torch"].cpu())
        if "torch_cuda" in states and states["torch_cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states["torch_cuda"])


class TokenizerLoader:
    """টোকেনাইজার ফাইল লোডার ক্লাস।"""
    @staticmethod
    def load(path: str) -> BPETokenizer:
        if os.path.exists(path):
            return BPETokenizer.load(path)
        raise FileNotFoundError(f"Tokenizer not found at: {path}")


class DatasetLoader:
    """রুট ফোল্ডার থেকে স্বয়ংক্রিয়ভাবে ডেটাসেট স্ক্যান এবং স্যাম্পল কাউন্টার ক্লাস।"""
    @staticmethod
    def load_train_val(
        config: ZortherDatasetConfig,
        tokenizer: BPETokenizer,
        batch_size: int,
        seed: int
    ) -> Tuple[ZortherDataLoader, Optional[ZortherDataLoader]]:
        all_paths = []
        actual_root = config.dataset_root
        if not os.path.exists(actual_root):
            alt_root = os.path.join("zorther", config.dataset_root)
            if os.path.exists(alt_root):
                actual_root = alt_root

        print(f"[DEBUG] Resolved dataset_root is set to: {actual_root}")

        if os.path.isfile(actual_root):
            all_paths.append(actual_root)
            print(f"  [INFO] Target path identified as a direct file: {actual_root}")
        elif os.path.isdir(actual_root):
            has_domain_files = False
            if hasattr(config, "sampling_ratios") and config.sampling_ratios:
                for domain in config.sampling_ratios.keys():
                    domain_dir = os.path.join(actual_root, domain)
                    if os.path.exists(domain_dir):
                        found_files = glob.glob(os.path.join(domain_dir, "*.jsonl")) + glob.glob(os.path.join(domain_dir, "*.txt"))
                        if found_files:
                            print(f"  Found files in domain '{domain}': {found_files}")
                            all_paths.extend(found_files)
                            has_domain_files = True

            if not has_domain_files:
                print(f"  [INFO] No domain files matched. Executing full recursive scan under: {actual_root}")
                fallback_files = (
                    glob.glob(os.path.join(actual_root, "**", "*.jsonl"), recursive=True) +
                    glob.glob(os.path.join(actual_root, "**", "*.txt"), recursive=True) +
                    glob.glob(os.path.join(actual_root, "*.jsonl")) +
                    glob.glob(os.path.join(actual_root, "*.txt"))
                )
                unique_files = list(set(os.path.abspath(f) for f in fallback_files))
                all_paths.extend(unique_files)

        if not all_paths:
            raise FileNotFoundError(f"No dataset files (.jsonl or .txt) found at path: {actual_root}")

        print(f"[INFO] Files selected for processing: {all_paths}")

        total_samples = 0
        for path in all_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            total_samples += 1
            except Exception as e:
                print(f"  [WARNING] Could not read file at {path}. Error: {e}")

        print("\n" + "="*50)
        print(f"[DATA DETECTED] Total training samples: {total_samples}")
        print("="*50 + "\n")

        if total_samples == 0:
            raise ValueError("No valid training samples found. Please check file contents.")

        effective_seq_len = min(config.packing_length, 2048)
        
        train_ds = ZortherDataset(all_paths, file_type="auto", shuffle=True, split="train")
        train_loader = ZortherDataLoader(
            dataset=train_ds,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_len=effective_seq_len,
            shuffle_buffer_size=config.shuffle_buffer_size,
            packing=config.quality_filter_enabled,
            seed=seed
        )

        val_ds = ZortherDataset(all_paths, file_type="auto", shuffle=False, split="val")
        val_loader = ZortherDataLoader(
            dataset=val_ds,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_len=effective_seq_len,
            shuffle_buffer_size=config.shuffle_buffer_size,
            packing=config.quality_filter_enabled,
            seed=seed
        )

        return train_loader, val_loader


class ModelBuilder:
    @staticmethod
    def build(config: ZortherModelConfig) -> ZortherTransformer:
        return ZortherTransformer(config)


class OptimizerBuilder:
    @staticmethod
    def build_config(train_cfg: ZortherTrainingConfig) -> OptimizerConfig:
        return OptimizerConfig(
            learning_rate=train_cfg.learning_rate,
            weight_decay=train_cfg.weight_decay,
            adam_beta1=train_cfg.adam_beta1,
            adam_beta2=train_cfg.adam_beta2,
            adam_epsilon=train_cfg.adam_epsilon,
            optimizer_type="adamw",
            scheduler_type=train_cfg.lr_scheduler_type,
            warmup_steps=train_cfg.warmup_steps,
            total_steps=train_cfg.total_steps,
            max_grad_norm=train_cfg.max_grad_norm,
        )


class LoggerBuilder:
    @staticmethod
    def build(train_cfg: ZortherTrainingConfig) -> LoggerManager:
        cfg = LoggerConfig(
            experiment_name="zorther_pretrain",
            log_dir=train_cfg.logging_dir,
            enable_console=True,
            enable_csv=train_cfg.use_csv,
            enable_tensorboard=train_cfg.use_tensorboard,
            enable_json=True
        )
        return LoggerManager(cfg)


class CheckpointLoader:
    @staticmethod
    def load_if_exists(
        step: Optional[int],
        model: ZortherTransformer,
        checkpoint_dir: str
    ) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        
        if step is None and os.path.exists(best_model_path):
            try:
                print("\n[AUTO-RESUME] Found saved 'best_model.pt'. Attempting to restore best validated state...")
                state_dict_meta = torch.load(best_model_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in state_dict_meta:
                    model.load_state_dict(state_dict_meta["model_state_dict"])
                else:
                    model.load_state_dict(state_dict_meta)
                
                resumed_step = state_dict_meta.get("step", 50)
                print(f"[SUCCESS] Best validated model weights loaded successfully from Step {resumed_step}!")
                return resumed_step, state_dict_meta
            except Exception as e:
                print(f"[WARNING] Failed to load best_model.pt directly: {e}. Falling back to standard checkpoints...")

        if step is None:
            if os.path.exists(checkpoint_dir):
                files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_step_*.pt"))
                if files:
                    steps = []
                    for f in files:
                        match = re.search(r"checkpoint_step_(\d+)", os.path.basename(f))
                        if match:
                            steps.append(int(match.group(1)))
                    if steps:
                        step = max(steps)
                        print(f"\n[AUTO-RESUME] Found saved checkpoint at step {step}. Resuming training...")

        if step is None:
            print("\n[TRAINING] No checkpoint found. Starting fresh training from scratch...")
            return None, None
            
        pt_path = os.path.join(checkpoint_dir, f"checkpoint_step_{step}.pt")
        if os.path.exists(pt_path):
            try:
                state_dict_meta = torch.load(pt_path, map_location="cpu", weights_only=False)
                if "model_state_dict" in state_dict_meta:
                    model.load_state_dict(state_dict_meta["model_state_dict"])
                else:
                    model.load_state_dict(state_dict_meta)
                print(f"[SUCCESS] Weights loaded successfully from Step {step}.")
                return step, state_dict_meta
            except Exception as e:
                print(f"[ERROR] Failed to load checkpoint weights: {e}")
                return None, None
                
        return None, None


class ETAEstimator:
    def __init__(self, total_steps: int) -> None:
        self.total_steps = total_steps
        self.start_time = time.perf_counter()

    def estimate(self, current_step: int) -> Tuple[float, str]:
        elapsed = time.perf_counter() - self.start_time
        steps_per_sec = current_step / elapsed if current_step > 0 else 0.0
        
        remaining_steps = max(0, self.total_steps - current_step)
        eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0.0
        
        hours = int(eta_seconds // 3600)
        minutes = int((eta_seconds % 3600) // 60)
        seconds = int(eta_seconds % 60)
        
        eta_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        return steps_per_sec, eta_str


class NeuralDiagnostics:
    @staticmethod
    def calculate_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = targets.view(-1)
        
        active_mask = targets_flat != -100
        active_logits = logits_flat[active_mask]
        active_targets = targets_flat[active_mask]
        
        if active_targets.numel() == 0:
            return 0.0
            
        _, top_indices = active_logits.topk(top_k, dim=-1)
        correct = top_indices.eq(active_targets.unsqueeze(-1)).any(dim=-1)
        return correct.float().mean().item()

    @staticmethod
    def check_gradient_status(model: nn.Module, computed_norm: Optional[float] = None) -> Tuple[float, str]:
        """প্যারামিটার গ্রেডিয়েন্ট ট্র্যাকিং এবং ওভারফ্লো চেক।"""
        if computed_norm is not None:
            total_norm = computed_norm
        else:
            total_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
        
        if total_norm > 15.0:
            status = "⚠️ EXPLOSION DETECTED!"
        elif total_norm < 1e-6 and total_norm > 0:
            status = "⚠️ VANISHING DETECTED!"
        elif total_norm == 0:
            status = "❌ NO GRADIENTS (OR ZEROED)"
        else:
            status = "✅ STABLE"
            
        return total_norm, status

    @staticmethod
    def display_live_predictions(logits: torch.Tensor, targets: torch.Tensor, tokenizer: BPETokenizer, limit: int = 8) -> None:
        """টার্গেট এবং প্রেডিকশনের কজাল শিফট অ্যালাইন্ড রিয়েল-টাইম তুলনা প্রদর্শন।"""
        logits_flat = logits[0][:-1].argmax(dim=-1)
        targets_flat = targets[0][1:]
        
        active_mask = targets_flat != -100
        active_preds = logits_flat[active_mask][:limit]
        active_targets = targets_flat[active_mask][:limit]
        
        if active_targets.numel() == 0:
            return
            
        pred_tokens = [tokenizer.decoder.get(tid.item(), "<unk>").replace("Ġ", " ") for tid in active_preds]
        target_tokens = [tokenizer.decoder.get(tid.item(), "<unk>").replace("Ġ", " ") for tid in active_targets]
        
        print("\n  [LIVE TRAINING PREDICTION MONITOR (Aligned Target vs Prediction)]")
        print(f"    Target Tokens : {target_tokens}")
        print(f"    Predicted     : {pred_tokens}")


class AdvancedTrainer:
    """উন্নত ট্রেইনার ক্লাস যা প্রসেস ডায়াগনস্টিকস মনিটর করে।"""
    def __init__(
        self,
        model: nn.Module,
        tokenizer: BPETokenizer,
        train_config: TrainerConfig,
        opt_config: OptimizerConfig,
        loss_config: LossConfig,
        logger: LoggerManager,
        total_steps: int,
        batch_size: int,
        early_stopping_patience: int = 5
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = train_config
        self.logger = logger
        self.batch_size = batch_size
        self.device = torch.device(train_config.device)
        self.model.to(self.device)

        self.optimizer_manager = OptimizerManager(model, opt_config, train_config.gradient_accumulation_steps)
        self.loss_manager = LossManager(loss_config)
        self.state = TrainState()
        self.eta_estimator = ETAEstimator(total_steps)
        
        self.best_val_loss = float("inf")
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_counter = 0

        # ব্যাকওয়ার্ড হুক সেটআপ
        self.grad_norms: Dict[str, float] = {}
        self._register_gradient_hooks()

    def _register_gradient_hooks(self) -> None:
        """প্যারামিটারে ব্যাকওয়ার্ড হুক রেজিস্টার করে যাতে zero_grad হবার আগেই নর্ম সেভ থাকে।"""
        def make_hook(name: str):
            def hook(grad):
                if grad is not None:
                    self.grad_norms[name] = grad.detach().data.norm(2).item()
                return grad
            return hook

        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(make_hook(name))

    def _get_active_grad_norm(self) -> float:
        """হুক করা গ্রেডিয়েন্টগুলোর মোট নর্ম রিটার্ন করবে।"""
        if not self.grad_norms:
            return 0.0
        return sum(val ** 2 for val in self.grad_norms.values()) ** 0.5
        
    @torch.no_grad()
    def _run_validation(self, val_loader: DataLoader) -> Tuple[float, float, float]:
        self.model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        steps = 0

        for batch in val_loader:
            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)

            logits = self.model(input_ids)
            loss = self.loss_manager.compute(logits, targets)
            
            top1 = NeuralDiagnostics.calculate_token_accuracy(logits, targets, top_k=1)
            top5 = NeuralDiagnostics.calculate_token_accuracy(logits, targets, top_k=5)

            total_loss += loss.item()
            total_top1 += top1
            total_top5 += top5
            steps += 1

        self.model.train()
        if steps == 0:
            return 0.0, 0.0, 0.0
        return total_loss / steps, total_top1 / steps, total_top5 / steps

    @torch.no_grad()
    def _run_live_generation_test(self, prompt: str = "User: Hello\nAssistant:") -> None:
        """ভ্যালিডেশনের পর মডেল থেকে লাইভ টেক্সট জেনারেশন আউটপুট দেখানোর ফাংশন।"""
        self.model.eval()
        print("\n" + "~"*65)
        print(f"[LIVE VALIDATION GENERATION TEST]")
        print(f"  Input Prompt: {repr(prompt)}")
        
        input_ids = self.tokenizer.encode(prompt, bos=True, eos=False)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)
        
        generated = list(input_ids)
        temperature = 0.7
        top_p = 0.9
        repetition_penalty = 1.15
        max_new_tokens = 35
        
        for _ in range(max_new_tokens):
            logits = self.model(input_tensor)
            next_token_logits = logits[0, -1, :] / temperature
            
            for token in set(generated):
                if next_token_logits[token] > 0:
                    next_token_logits[token] /= repetition_penalty
                else:
                    next_token_logits[token] *= repetition_penalty
            
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('inf')
            
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            token_val = next_token.item()
            generated.append(token_val)
            input_tensor = torch.cat([input_tensor, next_token.unsqueeze(0)], dim=-1)
            
            if token_val == self.tokenizer.eos_id:
                break
                
        decoded_out = self.tokenizer.decode(generated)
        print(f"  Generated   :\n{decoded_out}")
        print("~"*65 + "\n")
        self.model.train()

    def fit_and_monitor(self, train_loader: ZortherDataLoader, val_loader: Optional[ZortherDataLoader] = None) -> None:
        self.model.train()
        print("\n" + "="*65)
        print("[TRAINING ENGINE STARTED]")
        print(f"  └─ Batch Size: {self.batch_size} | Gradient Accumulation: {self.config.gradient_accumulation_steps}")
        print(f"  └─ Precision : {self.config.mixed_precision} | Total Epochs: {self.config.epochs}")
        print("="*65 + "\n")

        # ডায়নামিক ইপোচ ও স্টেপ স্কিপিং (অটো-রিসিউম বাদে প্রথম রানের ক্ষেত্রে বা সেফটি হিসেবে)
        start_epoch = self.state.epoch
        start_step = self.state.step

        for epoch in range(start_epoch, self.config.epochs):
            self.state.epoch = epoch
            
            # প্রতি ইপোচে ডাটালোডারের অবজেক্ট ট্র্যাকিং
            for step, batch in enumerate(train_loader):
                # যদি অটো-রিসিউম স্টেপের চেয়ে এই স্টেপটি ছোট হয়, তবে সরাসরি স্কিপ করা হচ্ছে
                if epoch == start_epoch and step < start_step:
                    continue
                    
                self.state.step = step
                self.state.global_step += 1

                input_ids = batch["input_ids"].to(self.device)
                targets = batch["targets"].to(self.device)

                logits = self.model(input_ids)
                loss = self.loss_manager.compute(logits, targets)
                
                # অপ্টিমাইজার ব্যাকওয়ার্ড ও লস আপডেট
                opt_state = self.optimizer_manager.step(loss)
                
                top1_acc = NeuralDiagnostics.calculate_token_accuracy(logits, targets, top_k=1)
                top5_acc = NeuralDiagnostics.calculate_token_accuracy(logits, targets, top_k=5)
                
                active_norm = self._get_active_grad_norm()
                grad_norm, grad_status = NeuralDiagnostics.check_gradient_status(self.model, computed_norm=active_norm)
                
                steps_per_sec, eta_str = self.eta_estimator.estimate(self.state.global_step)
                tokens_per_sec = steps_per_sec * input_ids.numel()

                if self.state.global_step % 10 == 0 or self.state.global_step == 1:
                    ppl = math.exp(min(loss.item(), 20.0))
                    print(f"\n" + "-"*65)
                    print(f"[METRICS DASHBOARD - STEP {self.state.global_step} (Epoch {self.state.epoch})]")
                    print(f"  ├─ Step Loss       : {loss.item():.4f} | Perplexity: {ppl:.4f}")
                    print(f"  ├─ Top-1 Token Acc : {top1_acc*100:.2f}% | Top-5 Token Acc: {top5_acc*100:.2f}%")
                    print(f"  ├─ Learning Rate   : {opt_state.learning_rate:.6f}")
                    print(f"  ├─ Gradient Norm   : {grad_norm:.6f} ({grad_status})")
                    print(f"  ├─ Throughput      : {tokens_per_sec:.2f} tokens/sec")
                    print(f"  └─ ETA             : {eta_str} remaining")
                    
                    # অ্যালাইন্ড রিয়েল-টাইম প্রেডিকশন ডিবাগিং
                    NeuralDiagnostics.display_live_predictions(logits, targets, self.tokenizer)
                    print("-"*65)

                if self.state.global_step % self.config.eval_interval == 0 and val_loader is not None:
                    val_loss, val_top1, val_top5 = self._run_validation(val_loader)
                    val_ppl = math.exp(min(val_loss, 20.0))
                    
                    print(f"\n" + "*"*65)
                    print(f"[VALIDATION METRICS - STEP {self.state.global_step}]")
                    print(f"  ├─ Val Loss        : {val_loss:.4f} | Val Perplexity: {val_ppl:.4f}")
                    print(f"  └─ Val Top-1 Acc   : {val_top1*100:.2f}% | Val Top-5 Acc: {val_top5*100:.2f}%")
                    
                    if val_top1 < 0.15 and top1_acc > 0.85:
                        print("\n  ⚠️  [GENERALIZATION CHECK WARNING]")
                        print("     - The model has memorized the training samples (Acc: 100%).")
                        print("     - Validation accuracy is very low because it cannot generalize.")
                        print("     - Action required: Transition from 5 samples to 'english.jsonl' dataset.")
                    
                    print("*"*65 + "\n")

                    self._run_live_generation_test(prompt="User: Hello\nAssistant:")

                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.early_stopping_counter = 0
                        best_path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
                        os.makedirs(os.path.dirname(os.path.abspath(best_path)), exist_ok=True)
                        
                        seed_states = SeedManager.save_state()
                        torch.save({
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer_manager.optimizer.state_dict(),
                            "scheduler_state_dict": self._get_scheduler_state(),
                            "val_loss": val_loss,
                            "step": self.state.global_step,
                            "seed_states": seed_states
                        }, best_path)
                        print(f"🔥 [BEST MODEL SAVED] New lowest validation loss achieved: {val_loss:.4f}!")
                    else:
                        self.early_stopping_counter += 1
                        print(f"⚠️ [EARLY STOPPING STATUS] Patience: {self.early_stopping_counter}/{self.early_stopping_patience}")
                        if self.early_stopping_counter >= self.early_stopping_patience:
                            print("\n🛑 [EARLY STOPPING TRIGGERED] Validation loss has stagnated.")
                            self.state.is_terminated = True
                            return

                if self.state.global_step % self.config.save_interval == 0:
                    self._save_checkpoint()

            if self.state.is_terminated:
                break
                
    def _get_scheduler_state(self):
        scheduler = self.optimizer_manager.scheduler
        if hasattr(scheduler, "state_dict"):
            return scheduler.state_dict()
        elif hasattr(scheduler, "__dict__"):
            return scheduler.__dict__
        else:
            return {}

    def _save_checkpoint(self) -> None:
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        filepath = os.path.join(self.config.checkpoint_dir, f"checkpoint_step_{self.state.global_step}.pt")
        
        seed_states = SeedManager.save_state()
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer_manager.optimizer.state_dict(),
            "scheduler_state_dict": self._get_scheduler_state(),
            "state": self.state,
            "best_val_loss": self.best_val_loss,
            "seed_states": seed_states
        }
        torch.save(checkpoint, filepath)
        print(f"💾 [CHECKPOINT SAVED] Model checkpoint saved at Step {self.state.global_step}.")


def main() -> None:
    args = ArgumentParser.parse()
    model_cfg, train_cfg, data_cfg, cpu_cfg = ConfigLoader.load_all(args)
    
    SeedManager.set_seed(train_cfg.seed)
    
    cpu_opt = CPUOptimizer(intra_threads=cpu_cfg.intra_op_num_threads, inter_threads=cpu_cfg.inter_op_num_threads)
    cpu_report = cpu_opt.run_full_optimization()
    
    tokenizer = TokenizerLoader.load(args.tokenizer_path)
    train_loader, val_loader = DatasetLoader.load_train_val(data_cfg, tokenizer, train_cfg.batch_size, train_cfg.seed)
    
    model = ModelBuilder.build(model_cfg)
    
    resumed_step, checkpoint_data = CheckpointLoader.load_if_exists(args.resume_step, model, train_cfg.checkpoint_dir)
    
    opt_cfg = OptimizerBuilder.build_config(train_cfg)
    logger = LoggerBuilder.build(train_cfg)
    
    logger.log_text(f"Hardware Optimization Report: {json.dumps(cpu_report, indent=4)}")
    
    trainer = AdvancedTrainer(
        model=model,
        tokenizer=tokenizer,
        train_config=TrainerConfig(
            epochs=train_cfg.epochs if hasattr(train_cfg, "epochs") else 500,
            gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
            mixed_precision=train_cfg.mixed_precision,
            max_grad_norm=train_cfg.max_grad_norm,
            eval_interval=train_cfg.eval_interval,
            save_interval=train_cfg.save_interval,
            checkpoint_dir=train_cfg.checkpoint_dir,
            device="cpu"
        ),
        opt_config=opt_cfg,
        loss_config=LossConfig(label_smoothing=train_cfg.label_smoothing, ignore_index=-100),
        logger=logger,
        total_steps=train_cfg.total_steps,
        batch_size=train_cfg.batch_size,
        early_stopping_patience=args.early_stopping_patience
    )
    
    if resumed_step is not None and checkpoint_data is not None:
        if "optimizer_state_dict" in checkpoint_data and checkpoint_data["optimizer_state_dict"] is not None:
            trainer.optimizer_manager.optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint_data and checkpoint_data["scheduler_state_dict"] is not None:
            sch_state = checkpoint_data["scheduler_state_dict"]
            if isinstance(sch_state, dict):
                for k, v in sch_state.items():
                    setattr(trainer.optimizer_manager.scheduler, k, v)
            
        saved_state = checkpoint_data.get("state")
        if saved_state is not None:
            trainer.state = saved_state
        else:
            # --- DYNAMIC COUNTER RECOVERY LOGIC (FOR BEST_MODEL.PT RESUME) ---
            # যদি সরাসরি ট্রেইনার স্টেট না থাকে, তবে গণিত কষে সঠিক স্টেপ ও ইপোচ রিকনস্ট্রাক্ট করা হচ্ছে
            resumed_step_int = checkpoint_data.get("step")
            if resumed_step_int is not None:
                trainer.state.global_step = resumed_step_int
                # ১৫,০০০ স্যাম্পল এবং কার্যকর ব্যাচ সাইজ ৮ (২ * ৪) হলে প্রতি ইপোচে ১,৮৭৫টি ওেইট আপডেট হয়
                steps_per_epoch = 1875
                trainer.state.epoch = resumed_step_int // steps_per_epoch
                trainer.state.step = resumed_step_int % steps_per_epoch
                print(f"\n[AUTO-RESUME] Reconstructed TrainState from checkpoint step metadata!")
                print(f"             └─ Global Step: {trainer.state.global_step} | Epoch: {trainer.state.epoch} | Epoch-Step: {trainer.state.step}")
                
            if "seed_states" in checkpoint_data:
                SeedManager.load_state(checkpoint_data["seed_states"])
                
            if "best_val_loss" in checkpoint_data:
                trainer.best_val_loss = checkpoint_data["best_val_loss"]
                
            print(f"[AUTO-RESUME] Successfully restored Optimizer, Scheduler, Seeds, and TrainState (Step: {trainer.state.global_step}, Epoch: {trainer.state.epoch})!")

    trainer.fit_and_monitor(train_loader, val_loader)
    logger.close()


if __name__ == "__main__":
    main()