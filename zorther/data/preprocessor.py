import hashlib
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple


class MinHashLSH:

    def __init__(self, num_perm: int = 128, threshold: float = 0.85, char_ngram: int = 5) -> None:
        self.num_perm = num_perm
        self.threshold = threshold
        self.char_ngram = char_ngram
        self.b, self.r = self._optimize_parameters(threshold, num_perm)
        self.hash_tables: List[Dict[Tuple[int, ...], Set[str]]] = [{} for _ in range(self.b)]
        self.prime = 4294967291
        self.a_coefficients = [(i * 2654435761) % self.prime for i in range(1, num_perm + 1)]
        self.b_coefficients = [(i * 1013904223) % self.prime for i in range(1, num_perm + 1)]

    def _optimize_parameters(self, threshold: float, num_perm: int) -> Tuple[int, int]:
        best_b, best_r = num_perm, 1
        min_err = float("inf")
        for b in range(1, num_perm + 1):
            if num_perm % b == 0:
                r = num_perm // b
                sim = (1.0 / b) ** (1.0 / r)
                err = abs(sim - threshold)
                if err < min_err:
                    min_err = err
                    best_b, best_r = b, r
        return best_b, best_r

    def _get_shingles(self, text: str) -> Set[str]:
        shingles = set()
        clean_text = "".join(text.split())
        if len(clean_text) < self.char_ngram:
            shingles.add(clean_text)
            return shingles
        for i in range(len(clean_text) - self.char_ngram + 1):
            shingles.add(clean_text[i:i + self.char_ngram])
        return shingles

    def _compute_minhash(self, shingles: Set[str]) -> List[int]:
        signatures = [self.prime] * self.num_perm
        for shingle in shingles:
            m = hashlib.md5()
            m.update(shingle.encode("utf-8"))
            shingle_hash = int(m.hexdigest(), 16) % self.prime
            for i in range(self.num_perm):
                perm_hash = (self.a_coefficients[i] * shingle_hash + self.b_coefficients[i]) % self.prime
                if perm_hash < signatures[i]:
                    signatures[i] = perm_hash
        return signatures

    def is_duplicate_and_insert(self, doc_id: str, text: str) -> bool:
        shingles = self._get_shingles(text)
        if not shingles:
            return True
        sig = self._compute_minhash(shingles)
        is_dup = False
        for band_idx in range(self.b):
            start = band_idx * self.r
            end = start + self.r
            band_key = tuple(sig[start:end])
            table = self.hash_tables[band_idx]
            if band_key in table:
                is_dup = True
            if band_key not in table:
                table[band_key] = set()
            table[band_key].add(doc_id)
        return is_dup


class DataPreprocessor:

    def __init__(self, lsh_threshold: float = 0.85, min_len: int = 10, max_len: int = 100000) -> None:
        self.lsh = MinHashLSH(threshold=lsh_threshold)
        self.min_len = min_len
        self.max_len = max_len
        self.control_chars_pattern = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]")

    def clean_text(self, text: str) -> str:
        text = self.control_chars_pattern.sub("", text)
        text = unicodedata.normalize("NFC", text)
        text = " ".join(text.split())
        return text

    def is_high_quality(self, text: str) -> bool:
        char_len = len(text)
        if char_len < self.min_len or char_len > self.max_len:
            return False
        alpha_chars = sum(1 for c in text if c.isalpha() or unicodedata.category(c).startswith("L"))
        if char_len > 0 and (alpha_chars / char_len) < 0.25:
            return False
        return True

    def process(self, doc_id: str, sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        text = sample.get("text", "")
        if not text:
            return None
        cleaned = self.clean_text(text)
        if not self.is_high_quality(cleaned):
            return None
        if self.lsh.is_duplicate_and_insert(doc_id, cleaned):
            return None
        processed_sample = dict(sample)
        processed_sample["text"] = cleaned
        return processed_sample
