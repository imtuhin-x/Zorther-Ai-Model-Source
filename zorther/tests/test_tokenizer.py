import os
import tempfile
import time
import unittest
from typing import List

from zorther.tokenizer.bpe_tokenizer import BPETokenizer
from zorther.tokenizer.streaming import StreamingDecoder



_global_tokenizer_tok = None
def _get_test_tokenizer_tok():
    global _global_tokenizer_tok
    if _global_tokenizer_tok is None:
        from zorther.tokenizer.bpe_tokenizer import BPETokenizer
        _global_tokenizer_tok = BPETokenizer()
        corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "বাংলা আমাদের মাতৃভাষা এবং পরম গর্বের ভাষা।",
            "Zorther is an advanced artificial intelligence model."
        ]
        _global_tokenizer_tok.train(corpus, vocab_size=300)
    return _global_tokenizer_tok

class TestTokenizerInitialization(unittest.TestCase):

    def test_base_init(self) -> None:
        tokenizer = BPETokenizer()
        self.assertIsNotNone(tokenizer)
        self.assertTrue(tokenizer.vocab_size > 0)
        self.assertEqual(tokenizer.bos_id, 0)
        self.assertEqual(tokenizer.eos_id, 1)


class TestEncode(unittest.TestCase):
    _tokenizer: BPETokenizer

    @classmethod
    def setUpClass(cls) -> None:
        cls._tokenizer = BPETokenizer()
        corpus = [
            "The quick brown fox jumps over the lazy dog.",
            "বাংলা আমাদের মাতৃভাষা এবং পরম গর্বের ভাষা।",
            "Zorther is an advanced artificial intelligence model."
        ]
        cls._tokenizer.train(corpus, vocab_size=300)

    def test_basic_encode(self) -> None:
        text = "Zorther is advanced"
        tokens = self._tokenizer.encode(text, bos=False, eos=False)
        self.assertIsInstance(tokens, list)
        self.assertTrue(len(tokens) > 0)


class TestDecode(unittest.TestCase):

    def test_basic_decode(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        text = "fox jumps over"
        tokens = tokenizer.encode(text, bos=False, eos=False)
        decoded = tokenizer.decode(tokens)
        self.assertIsInstance(decoded, str)


class TestEncodeDecodeRoundTrip(unittest.TestCase):

    def test_round_trip(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        test_strings = [
            "quick brown fox",
            "বাংলা ভাষা",
            "artificial intelligence"
        ]
        for s in test_strings:
            tokens = tokenizer.encode(s, bos=False, eos=False)
            decoded = tokenizer.decode(tokens)
            self.assertEqual(s, decoded)


class TestSpecialTokens(unittest.TestCase):

    def test_special_handling(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        text = "hello"
        tokens = tokenizer.encode(text, bos=True, eos=True)
        self.assertEqual(tokens[0], tokenizer.bos_id)
        self.assertEqual(tokens[-1], tokenizer.eos_id)


class TestVocabulary(unittest.TestCase):

    def test_vocab_limits(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        self.assertTrue(tokenizer.vocab_size >= 300)
        for token_id in range(tokenizer.vocab_size):
            token_str = tokenizer.decode([token_id])
            self.assertIsInstance(token_str, str)


class TestUnknownToken(unittest.TestCase):

    def test_no_oov_failure(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        unknown_text = "xyz123#@!"
        tokens = tokenizer.encode(unknown_text, bos=False, eos=False)
        decoded = tokenizer.decode(tokens)
        self.assertEqual(unknown_text, decoded)


class TestBatchEncoding(unittest.TestCase):

    def test_batch_processing(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        texts = ["quick brown", "বাংলা ভাষা", "intelligence"]
        batch_tokens = tokenizer.encode_batch(texts, bos=True, eos=True)
        self.assertEqual(len(batch_tokens), 3)
        for tokens in batch_tokens:
            self.assertEqual(tokens[0], tokenizer.bos_id)


class TestPadding(unittest.TestCase):

    def test_manual_padding(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        text1 = "small text"
        text2 = "much longer text inside batch"
        
        tokens1 = tokenizer.encode(text1)
        tokens2 = tokenizer.encode(text2)
        
        max_len = max(len(tokens1), len(tokens2))
        padded1 = tokens1 + [tokenizer.pad_id] * (max_len - len(tokens1))
        padded2 = tokens2 + [tokenizer.pad_id] * (max_len - len(tokens2))
        
        self.assertEqual(len(padded1), len(padded2))


class TestTruncation(unittest.TestCase):

    def test_token_truncation(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        text = "quick brown fox jumps over"
        tokens = tokenizer.encode(text)
        truncated = tokens[:3]
        self.assertEqual(len(truncated), 3)


class TestStreamingDecode(unittest.TestCase):

    def test_stream_boundary(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        decoder = StreamingDecoder(tokenizer)
        
        text = "বাংলা ভাষা অত্যন্ত চমৎকার।"
        tokens = tokenizer.encode(text, bos=False, eos=False)
        
        streamed_chars = []
        for t in tokens:
            chunk = decoder.put(t)
            streamed_chars.append(chunk)
            
        streamed_chars.append(decoder.flush())
        final_decoded = "".join(streamed_chars)
        self.assertEqual(text, final_decoded)


class TestTokenizerSaveLoad(unittest.TestCase):

    def test_serialization(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "tokenizer.json")
            tokenizer.save(filepath)
            
            loaded_tokenizer = BPETokenizer.load(filepath)
            self.assertEqual(tokenizer.vocab_size, loaded_tokenizer.vocab_size)
            
            test_text = "fox jumps"
            self.assertEqual(tokenizer.encode(test_text), loaded_tokenizer.encode(test_text))


class TestTokenizerPerformance(unittest.TestCase):

    def test_latency(self) -> None:
        tokenizer = _get_test_tokenizer_tok()
        large_text = " ".join(["quick brown fox jumps over"] * 50)
        
        start = time.perf_counter()
        for _ in range(10):
            tokenizer.encode(large_text)
        elapsed = time.perf_counter() - start
        
        self.assertTrue(elapsed < 1.0)


if __name__ == "__main__":
    unittest.main()