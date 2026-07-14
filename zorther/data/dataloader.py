"""
Zorther Gen - SOTA Enterprise Dynamic DataLoader & Collator
Features:
- Smart SFT Left-Truncation (Always preserves 100% of the Assistant response)
- Dynamic Response Chunking (Splits long responses with continuation markers)
- Sliding Window Pre-training Chunking (32-token overlap stride)
- SFT-Aware Sequence Packing (Tightly packs multiple samples if packing=True)
- Zero-Target Prevention Collator (Filters out empty target batches before training loop)
- Strict max_seq_len + 1 Alignment (Guarantees sliced tensors are exactly max_seq_len)
"""

import random
from typing import Dict, Generator, Iterator, List, Optional, Tuple
import torch

from zorther.tokenizer.base import BaseTokenizer
from zorther.data.dataset import ZortherDataset
from zorther.data.packing import SequencePacker


class ZortherDataLoader:

    def __init__(
        self,
        dataset: ZortherDataset,
        tokenizer: BaseTokenizer,
        batch_size: int,
        max_seq_len: int,
        shuffle_buffer_size: int = 10000,
        packing: bool = True,
        seed: int = 42
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.shuffle_buffer_size = shuffle_buffer_size
        self.packing = packing
        
        self.packer = SequencePacker(
            max_seq_len=max_seq_len + 1,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id
        )
        random.seed(seed)

    def _tokenized_stream(self) -> Generator[Tuple[List[int], List[int]], None, None]:
        for sample in self.dataset:
            # কলামগুলোর ডাইনামিক এক্সট্র্যাকশন
            instruction = sample.get("instruction", "").strip()
            user_input = sample.get("input", "").strip()
            output = sample.get("output", "").strip()
            raw_text = sample.get("text", "").strip()
            
            # ক) SFT / ইন্সট্রাক্ট ডেটা মোড
            if user_input and output:
                if instruction:
                    prompt = f"Instruction: {instruction}\nUser: {user_input}\nAssistant:\n"
                else:
                    prompt = f"User: {user_input}\nAssistant:\n"
                
                response = f"{output}"
                
                prompt_tokens = self.tokenizer.encode(prompt, bos=True, eos=False)
                response_tokens = self.tokenizer.encode(response, bos=False, eos=True)
                
                # কজাল শিফটিং অ্যালাইনমেন্টের জন্য সর্বোচ্চ দৈর্ঘ্য (max_seq_len + 1)
                max_len = self.max_seq_len + 1
                
                # কেস ১: রেসপন্সটি যদি নিজেই সর্বোচ্চ দৈর্ঘ্য অতিক্রম করে
                if len(response_tokens) > max_len - 32:  # ৩২ টোকেন সেফটি বাফার
                    chunk_size = max_len - 64  # প্রম্পটের জন্য ৬৪ টোকেন বাফার রাখা হচ্ছে
                    continuation_prompt = "\nAssistant (continued):\n"
                    continuation_tokens = self.tokenizer.encode(continuation_prompt, bos=False, eos=False)
                    
                    for i in range(0, len(response_tokens), chunk_size):
                        resp_chunk = response_tokens[i : i + chunk_size]
                        
                        if i == 0:
                            allowed_prompt_len = max_len - len(resp_chunk)
                            prompt_chunk = prompt_tokens[-allowed_prompt_len:] if allowed_prompt_len > 0 else []
                        else:
                            # পরবর্তী চাঙ্কগুলোর শুরুতে কন্টিনিউয়েশন মার্কার বসানো হচ্ছে
                            allowed_prompt_len = max_len - len(resp_chunk)
                            prompt_chunk = continuation_tokens[-allowed_prompt_len:] if allowed_prompt_len > 0 else []
                            
                        input_ids = prompt_chunk + resp_chunk
                        targets = [-100] * len(prompt_chunk) + resp_chunk
                        
                        if len(resp_chunk) > 0:
                            yield input_ids, targets
                            
                # কেস ২: রেসপন্সটি max_seq_len এর ছোট, কিন্তু টোটাল সিকোয়েন্সটি বড়
                # রেসপন্স ১০০% অক্ষত রেখে প্রম্পটের শুরুর অংশটি কেটে ছোট করা হচ্ছে (Left-Truncation)
                else:
                    allowed_prompt_len = max_len - len(response_tokens)
                    if allowed_prompt_len > 0:
                        prompt_tokens = prompt_tokens[-allowed_prompt_len:]
                    else:
                        prompt_tokens = []
                        
                    input_ids = prompt_tokens + response_tokens
                    targets = [-100] * len(prompt_tokens) + response_tokens
                    
                    yield input_ids, targets
            
            # খ) প্রি-ট্রেইনিং প্লেইন টেক্সট মোড (বই বা উইকি ডায়নামিক চাংকিং উইথ ওভারল্যাপ)
            elif raw_text:
                text = f"Document:\n{raw_text}"
                tokens = self.tokenizer.encode(text, bos=True, eos=True)
                
                max_len = self.max_seq_len + 1
                overlap = 32
                step_size = max_len - overlap
                
                if len(tokens) > max_len:
                    for i in range(0, len(tokens) - overlap, step_size):
                        chunk = tokens[i : i + max_len]
                        if len(chunk) > 1:
                            yield chunk, list(chunk)
                else:
                    yield tokens, list(tokens)

    def _packed_stream(self, stream: Iterator[Tuple[List[int], List[int]]]) -> Generator[Tuple[List[int], List[int]], None, None]:
        """SFT-Aware মাস্কিং ফ্রেন্ডলি সিকোয়েন্স প্যাকিং মেকানিজম।"""
        current_input: List[int] = []
        current_target: List[int] = []
        max_len = self.max_seq_len + 1
        
        for input_ids, targets in stream:
            # যদি নতুন স্যাম্পল যোগ করার পর তা ৫১৩ এর নিচে থাকে, তবে প্যাক করা হবে
            if len(current_input) + len(input_ids) <= max_len:
                current_input.extend(input_ids)
                current_target.extend(targets)
            else:
                if current_input:
                    yield current_input, current_target
                current_input = list(input_ids)
                current_target = list(targets)
                
        if current_input:
            yield current_input, current_target

    def _shuffled_stream(self, stream: Iterator[Tuple[List[int], List[int]]]) -> Generator[Tuple[List[int], List[int]], None, None]:
        buffer: List[Tuple[List[int], List[int]]] = []
        for item in stream:
            buffer.append(item)
            if len(buffer) >= self.shuffle_buffer_size:
                idx = random.randint(0, len(buffer) - 1)
                yield buffer.pop(idx)
        
        while buffer:
            idx = random.randint(0, len(buffer) - 1)
            yield buffer.pop(idx)

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        raw_stream = self._tokenized_stream()
        
        # প্যাকিং কনফিগারেশন সক্রিয় করা হলো
        if self.packing:
            processed_stream = self._packed_stream(raw_stream)
        else:
            processed_stream = raw_stream
            
        shuffled_stream = self._shuffled_stream(processed_stream)
        
        batch_accumulator: List[Tuple[List[int], List[int]]] = []
        
        for input_ids, targets in shuffled_stream:
            batch_accumulator.append((input_ids, targets))
            if len(batch_accumulator) == self.batch_size:
                batch_dict = self._collate_batch(batch_accumulator)
                # SFT SAFEGUARD: জিরো-টার্গেট ব্যাচ বাতিল করা হচ্ছে
                if batch_dict is not None:
                    yield batch_dict
                batch_accumulator.clear()
                
        if batch_accumulator:
            batch_dict = self._collate_batch(batch_accumulator)
            if batch_dict is not None:
                yield batch_dict

    def _collate_batch(self, batch_data: List[Tuple[List[int], List[int]]]) -> Optional[Dict[str, torch.Tensor]]:
        # কজাল শিফটিং এর কারণে সর্বোচ্চ দৈর্ঘ্য max_seq_len + 1 এ লক করা হলো
        max_len = self.max_seq_len + 1
        
        padded_inputs: List[List[int]] = []
        padded_targets: List[List[int]] = []
        
        for input_ids, targets in batch_data:
            # ট্রাঙ্কেশন (৫১৩ সাইজে এলাইনমেন্ট)
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                targets = targets[:max_len]
                
            # প্যাডিং মেকানিজম
            pad_len = max_len - len(input_ids)
            if pad_len > 0:
                padded_input = input_ids + [self.tokenizer.pad_id] * pad_len
                padded_target = targets + [-100] * pad_len
            else:
                padded_input = input_ids
                padded_target = targets
                
            padded_inputs.append(padded_input)
            padded_targets.append(padded_target)
            
        tensor_inputs = torch.tensor(padded_inputs, dtype=torch.long)
        tensor_targets = torch.tensor(padded_targets, dtype=torch.long)
        
        # শিফটিং লজিক (Autoregressive Next-Token Prediction)
        # ৫১৩ সাইজের টেনসরকে শিফট করার পর ইনপুট ও টার্গেট সাইজ হুবহু max_seq_len (৫১২) হবে!
        input_ids = tensor_inputs[:, :-1].contiguous()
        targets = tensor_targets[:, 1:].contiguous()
        
        # --- SFT SAFEGUARD: জিরো-টার্গেট প্রটেকশন ---
        # যদি পুরো ব্যাচের সব টার্গেট -100 হয়, তবে ডিভিশন বাই জিরো এড়াতে None রিটার্ন করা হবে
        if (targets != -100).sum() == 0:
            return None
            
        return {
            "input_ids": input_ids,
            "targets": targets
        }