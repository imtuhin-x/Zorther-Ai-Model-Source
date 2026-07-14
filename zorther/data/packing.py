"""
Zorther Gen - SOTA Loss-Mask Aware Sequence Packer
Features:
- Parallel packing of input_ids and target_masks (preserving SFT -100 loss masks)
- Explicit document boundary isolation with strict <eos> injection
- Loss-Masked Flushing (appends pad_token_id to input, and -100 to targets)
- Prevents cross-document attention pollution and Pad Token learning
"""

from typing import Generator, Iterator, List, Tuple


class SequencePacker:

    def __init__(self, max_seq_len: int, eos_token_id: int = 1, pad_token_id: int = 2) -> None:
        self.max_seq_len = max_seq_len
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        
        # সমান্তরাল বাফার ট্র্যাকিং
        self.input_buffer: List[int] = []
        self.target_buffer: List[int] = []

    def pack_continuous(self, tokenized_stream: Iterator[Tuple[List[int], List[int]]]) -> Generator[Tuple[List[int], List[int]], None, None]:
        """
        Input_ids এবং Target Loss Masks সমান্তরালভাবে প্যাক করে বাউন্ডারি লস মাস্ক নিশ্চিত করে।
        """
        for input_ids, targets in tokenized_stream:
            if not input_ids or not targets:
                continue
            
            # বাফারে যুক্ত করা
            self.input_buffer.extend(input_ids)
            self.target_buffer.extend(targets)
            
            # ডকুমেন্ট বাউন্ডারি আইসোলেশন: ডকুমেন্টের শেষে EOS এবং Target-এ -100 লস মাস্ক যুক্ত করা
            if self.input_buffer[-1] != self.eos_token_id:
                self.input_buffer.append(self.eos_token_id)
                self.target_buffer.append(-100)  # EOS টোকেনের লস মাস্ক আউট করা হলো

            # নির্দিষ্ট ম্যাক্স সিকোয়েন্স লেন্থ অনুযায়ী চাঙ্ক তৈরি করা
            while len(self.input_buffer) >= self.max_seq_len:
                input_chunk = self.input_buffer[:self.max_seq_len]
                target_chunk = self.target_buffer[:self.max_seq_len]
                
                self.input_buffer = self.input_buffer[self.max_seq_len:]
                self.target_buffer = self.target_buffer[self.max_seq_len:]
                
                yield input_chunk, target_chunk

    def pack_bin_fitting(self, tokenized_docs: List[Tuple[List[int], List[int]]]) -> List[Tuple[List[int], List[int]]]:
        
        sorted_docs = sorted(tokenized_docs, key=lambda x: len(x[0]), reverse=True)
        bins: List[Tuple[List[int], List[int]]] = []

        for input_ids, targets in sorted_docs:
            if not input_ids:
                continue
            
            doc_len = len(input_ids)
            if doc_len > self.max_seq_len:
                input_ids = input_ids[:self.max_seq_len]
                targets = targets[:self.max_seq_len]
                doc_len = self.max_seq_len

            placed = False
            for i, (b_input, b_target) in enumerate(bins):
                if len(b_input) + doc_len <= self.max_seq_len:
                    b_input.extend(input_ids)
                    b_target.extend(targets)
                    bins[i] = (b_input, b_target)
                    placed = True
                    break

            if not placed:
                bins.append((list(input_ids), list(targets)))

        # অবশিষ্ট খালি অংশগুলো প্যাডিং এবং -100 টার্গেট মাস্ক দিয়ে পূরণ করা
        completed_bins = []
        for b_input, b_target in bins:
            padding_needed = self.max_seq_len - len(b_input)
            if padding_needed > 0:
                b_input.extend([self.pad_token_id] * padding_needed)
                b_target.extend([-100] * padding_needed)  # প্যাডের অংশের লস সম্পূর্ণ ইগনোর করা হলো
            completed_bins.append((b_input, b_target))

        return completed_bins

    def flush(self) -> Tuple[List[int], List[int]]:
        """
        বাফারে অবশিষ্ট থাকা অংশটুকুর ইনপুটে PAD এবং টার্গেটে -100 যুক্ত করে ফ্ল্যাশ করে।
        """
        if not self.input_buffer:
            return [], []
            
        padding_needed = self.max_seq_len - len(self.input_buffer)
        
        flushed_input = list(self.input_buffer)
        flushed_target = list(self.target_buffer)
        
        if padding_needed > 0:
            flushed_input.extend([self.pad_token_id] * padding_needed)
            flushed_target.extend([-100] * padding_needed)  # প্যাডের অংশের লস সম্পূর্ণ ইগনোর করা হলো
            
        self.input_buffer.clear()
        self.target_buffer.clear()
        
        return flushed_input, flushed_target