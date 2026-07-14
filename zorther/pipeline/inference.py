"""Inference pipeline."""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from zorther.tokenizer.base import BaseTokenizer
from zorther.tokenizer.streaming import StreamingDecoder
from zorther.model.cache import CacheConfig, CacheManager


@dataclass
class InferenceConfig:
    max_batch_size: int = 1
    device: str = "cpu"
    cache_type: str = "static"


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.15
    do_sample: bool = True


class PromptProcessor:

    @staticmethod
    def process(prompt: str, format_chat: bool = False, system_prompt: Optional[str] = None) -> str:
        if format_chat:
            sys_part = system_prompt if system_prompt is not None else "You are Zorther, a helpful, honest, and highly logical AI assistant. If you do not have verified information to answer a question, you must politely decline to answer."
            # ডাইনামিকালি আপনার কাস্টম ফরম্যাটে প্রম্পট তৈরি করা হচ্ছে
            return f"Instruction: {sys_part}\nUser: {prompt}\nAssistant:\n"
        return prompt


class InputBuilder:

    @staticmethod
    def build(prompt: str, tokenizer: BaseTokenizer, device: torch.device) -> Tuple[torch.Tensor, List[int]]:
        tokens = tokenizer.encode(prompt, bos=True, eos=False)
        tensor = torch.tensor([tokens], dtype=torch.long, device=device)
        return tensor, tokens


class LogitsProcessor:

    @staticmethod
    def apply_repetition_penalty(logits: torch.Tensor, generated_tokens: List[int], penalty: float) -> torch.Tensor:
        if penalty == 1.0 or not generated_tokens:
            return logits
        
        seen_tokens = set(generated_tokens)
        for token_id in seen_tokens:
            val = logits[0, token_id].item()
            if val > 0:
                logits[0, token_id] /= penalty
            else:
                logits[0, token_id] *= penalty
        return logits

    @staticmethod
    def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        if top_k <= 0:
            return logits
        v, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
        min_values = v[:, -1].unsqueeze(-1)
        return torch.where(logits < min_values, torch.full_like(logits, float("-inf")), logits)

    @staticmethod
    def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p <= 0.0 or top_p >= 1.0:
            return logits
        
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        indices_to_remove = sorted_indices_to_remove.scatter(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        return torch.where(indices_to_remove, torch.full_like(logits, float("-inf")), logits)


class Sampler:

    @staticmethod
    def sample(logits: torch.Tensor, config: GenerationConfig) -> int:
        if not config.do_sample or config.temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())
        
        scaled_logits = logits / config.temperature
        probs = torch.softmax(scaled_logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())


class StoppingCriteria:

    def __init__(self, stop_ids: List[int], max_length: int) -> None:
        self.stop_ids = set(stop_ids)
        self.max_length = max_length

    def should_stop(self, token_id: int, current_length: int) -> bool:
        if token_id in self.stop_ids:
            return True
        if current_length >= self.max_length:
            return True
        return False


class TextPostProcessor:

    @staticmethod
    def post_process(text: str) -> str:
        # ChatML এর পরিবর্তে <eos> ক্যারেক্টার রিমুভ করা হচ্ছে
        text = text.replace("<eos>", "")
        return text.strip()


@dataclass
class GenerationMetrics:
    time_taken: float = 0.0
    tokens_generated: int = 0
    latency_per_token_ms: float = 0.0
    tokens_per_second: float = 0.0


class GenerationSession:

    def __init__(self) -> None:
        self.generated_tokens: List[int] = []
        self.metrics = GenerationMetrics()
        self.start_time = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def record_step(self, token_id: int) -> None:
        self.generated_tokens.append(token_id)

    def stop(self) -> GenerationMetrics:
        end_time = time.perf_counter()
        self.metrics.time_taken = end_time - self.start_time
        self.metrics.tokens_generated = len(self.generated_tokens)
        if self.metrics.tokens_generated > 0:
            self.metrics.latency_per_token_ms = (self.metrics.time_taken / self.metrics.tokens_generated) * 1000.0
            self.metrics.tokens_per_second = self.metrics.tokens_generated / self.metrics.time_taken
        return self.metrics


class InferenceEngine:

    def __init__(self, model: nn.Module, tokenizer: BaseTokenizer, config: InferenceConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        gen_config: Optional[GenerationConfig] = None,
        format_chat: bool = False,
        system_prompt: Optional[str] = None
    ) -> Generator[str, None, str]:
        gen_config = gen_config if gen_config is not None else GenerationConfig()
        
        processed_prompt = PromptProcessor.process(prompt, format_chat, system_prompt)
        input_tensor, prompt_tokens = InputBuilder.build(processed_prompt, self.tokenizer, self.device)
        
        session = GenerationSession()
        session.start()
        
        stream_decoder = StreamingDecoder(self.tokenizer)
        # স্টপ আইডি হিসেবে শুধুমাত্র আসল eos_id চেক করা হবে
        stopping_criteria = StoppingCriteria(
            stop_ids=[self.tokenizer.eos_id],
            max_length=self.model.config.max_seq_len
        )
        
        cache_config = CacheConfig(
            max_batch_size=self.config.max_batch_size,
            max_seq_len=self.model.config.max_seq_len,
            num_key_value_heads=self.model.config.num_key_value_heads,
            head_dim=self.model.config.hidden_size // self.model.config.num_attention_heads,
            dtype=torch.float32
        )
        
        kv_caches = CacheManager(
            num_layers=self.model.config.num_layers,
            cache_type=self.config.cache_type,
            config=cache_config
        )

        curr_pos = 0
        logits = self.model(input_tensor, kv_caches=kv_caches.caches, start_pos=curr_pos)
        next_token_logits = logits[:, -1, :]
        
        next_token_logits = LogitsProcessor.apply_repetition_penalty(next_token_logits, prompt_tokens, gen_config.repetition_penalty)
        next_token_logits = LogitsProcessor.apply_top_k(next_token_logits, gen_config.top_k)
        next_token_logits = LogitsProcessor.apply_top_p(next_token_logits, gen_config.top_p)
        
        next_token = Sampler.sample(next_token_logits, gen_config)
        session.record_step(next_token)
        
        curr_pos += input_tensor.shape[1]
        
        yield stream_decoder.put(next_token)

        for _ in range(gen_config.max_new_tokens - 1):
            if stopping_criteria.should_stop(next_token, curr_pos):
                break
                
            input_tensor = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits = self.model(input_tensor, kv_caches=kv_caches.caches, start_pos=curr_pos)
            next_token_logits = logits[:, -1, :]
            
            next_token_logits = LogitsProcessor.apply_repetition_penalty(next_token_logits, session.generated_tokens, gen_config.repetition_penalty)
            next_token_logits = LogitsProcessor.apply_top_k(next_token_logits, gen_config.top_k)
            next_token_logits = LogitsProcessor.apply_top_p(next_token_logits, gen_config.top_p)
            
            next_token = Sampler.sample(next_token_logits, gen_config)
            session.record_step(next_token)
            
            curr_pos += 1
            
            yield stream_decoder.put(next_token)

        yield stream_decoder.flush()
        
        metrics = session.stop()
        final_text = self.tokenizer.decode(session.generated_tokens)
        return TextPostProcessor.post_process(final_text)