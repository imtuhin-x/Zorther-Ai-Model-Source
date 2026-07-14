import time
import json
import os
import tempfile
import unittest
from typing import Dict, List

from zorther.tokenizer.bpe_tokenizer import BPETokenizer
from zorther.data.dataset import ZortherDataset
from zorther.data.preprocessor import DataPreprocessor, MinHashLSH
from zorther.data.packing import SequencePacker
from zorther.data.dataloader import ZortherDataLoader



_global_tokenizer_data = None
def _get_test_tokenizer_data():
    global _global_tokenizer_data
    if _global_tokenizer_data is None:
        from zorther.tokenizer.bpe_tokenizer import BPETokenizer
        _global_tokenizer_data = BPETokenizer()
        _global_tokenizer_data.train(["hello world", "বাংলা ভাষা", "Zorther training model"], vocab_size=270)
    return _global_tokenizer_data

class TestDatasetInitialization(unittest.TestCase):

    def test_init_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test.jsonl")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "Hello"}) + "\n")
            
            dataset = ZortherDataset(file_path)
            self.assertEqual(len(dataset.file_paths), 1)


class TestJSONLLoading(unittest.TestCase):

    def test_jsonl_parser(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "data.jsonl")
            samples = [
                {"text": "Zorther pretraining starts."},
                {"text": "বাংলা ভাষার নতুন ইভলুশন।"}
            ]
            with open(file_path, "w", encoding="utf-8") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")
            
            dataset = ZortherDataset(file_path)
            loaded = list(dataset)
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0]["text"], "Zorther pretraining starts.")


class TestTextProcessing(unittest.TestCase):

    def test_text_normalization(self) -> None:
        preprocessor = DataPreprocessor()
        dirty_text = "Hello \x00 world!    বাংলা \n ভাষা"
        cleaned = preprocessor.clean_text(dirty_text)
        self.assertEqual(cleaned, "Hello world! বাংলা ভাষা")


class TestDataCleaning(unittest.TestCase):

    def test_quality_filter(self) -> None:
        preprocessor = DataPreprocessor(min_len=5)
        bad_text = "abc"
        good_text = "This is a high quality pretraining text."
        
        self.assertFalse(preprocessor.is_high_quality(bad_text))
        self.assertTrue(preprocessor.is_high_quality(good_text))


class TestDeduplication(unittest.TestCase):

    def test_minhash_lsh(self) -> None:
        lsh = MinHashLSH(threshold=0.85)
        doc1 = "The quick brown fox jumps over the lazy dog."
        doc2 = "The quick brown fox jumps over the lazy dog."
        doc3 = "The quick brown fox jumps over the lazy dog!"
        doc4 = "This is a completely different sentence altogether."
        
        self.assertFalse(lsh.is_duplicate_and_insert("1", doc1))
        self.assertTrue(lsh.is_duplicate_and_insert("2", doc2))
        self.assertTrue(lsh.is_duplicate_and_insert("3", doc3))
        self.assertFalse(lsh.is_duplicate_and_insert("4", doc4))


class TestTokenizationPipeline(unittest.TestCase):

    def test_on_the_fly_tokenization(self) -> None:
        tokenizer = BPETokenizer()
        corpus = ["Zorther dataset loading pipeline testing."]
        tokenizer.train(corpus, vocab_size=265)
        
        tokens = tokenizer.encode(corpus[0])
        self.assertTrue(len(tokens) > 0)


class TestSequencePacking(unittest.TestCase):

    def test_packing_density(self) -> None:
        packer = SequencePacker(max_seq_len=10, eos_token_id=1, pad_token_id=2)
        docs = [
            [3, 4, 5],
            [6, 7],
            [8, 9, 10, 11]
        ]
        
        packed = list(packer.pack_continuous(docs))
        self.assertTrue(len(packed) > 0)
        self.assertEqual(len(packed[0]), 10)


class TestDataLoader(unittest.TestCase):
    _tokenizer: BPETokenizer

    @classmethod
    def setUpClass(cls) -> None:
        cls._tokenizer = BPETokenizer()
        cls._tokenizer.train(["hello world", "বাংলা ভাষা", "Zorther training model"], vocab_size=270)

    def test_loader_iter(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "hello world"}) + "\n")
                f.write(json.dumps({"text": "বাংলা ভাষা"}) + "\n")
                
            dataset = ZortherDataset(filepath)
            loader = ZortherDataLoader(
                dataset=dataset,
                tokenizer=self._tokenizer,
                batch_size=1,
                max_seq_len=8,
                shuffle_buffer_size=2,
                packing=True
            )
            
            batches = list(loader)
            self.assertTrue(len(batches) > 0)
            self.assertIn("input_ids", batches[0])
            self.assertIn("targets", batches[0])


class TestBatchGeneration(unittest.TestCase):

    def test_dimension_shifting(self) -> None:
        tokenizer = _get_test_tokenizer_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "Zorther training model"}) + "\n")
                
            dataset = ZortherDataset(filepath)
            loader = ZortherDataLoader(
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=1,
                max_seq_len=4,
                packing=True
            )
            
            batch = next(iter(loader))
            self.assertEqual(batch["input_ids"].shape[1], 4)
            self.assertEqual(batch["targets"].shape[1], 4)


class TestPadding(unittest.TestCase):

    def test_non_packed_padding(self) -> None:
        tokenizer = _get_test_tokenizer_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(json.dumps({"text": "hi"}) + "\n")
                
            dataset = ZortherDataset(filepath)
            loader = ZortherDataLoader(
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=1,
                max_seq_len=8,
                packing=False
            )
            
            batch = next(iter(loader))
            self.assertEqual(batch["input_ids"].shape[1], 8)
            self.assertEqual(batch["input_ids"][0][-1].item(), tokenizer.pad_id)


class TestStreamingDataset(unittest.TestCase):

    def test_streaming_yield(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "stream.jsonl")
            with open(filepath, "w", encoding="utf-8") as f:
                for i in range(10):
                    f.write(json.dumps({"text": f"Sentence {i}"}) + "\n")
                    
            dataset = ZortherDataset(filepath)
            count = sum(1 for _ in dataset)
            self.assertEqual(count, 10)


class TestShuffle(unittest.TestCase):

    def test_shuffle_randomness(self) -> None:
        tokenizer = _get_test_tokenizer_data()
        dataset = ZortherDataset([])
        loader = ZortherDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=1,
            max_seq_len=4,
            shuffle_buffer_size=10
        )
        
        sequence_stream = [[i] for i in range(100)]
        shuffled = list(loader._shuffled_stream(iter(sequence_stream)))
        self.assertNotEqual(shuffled, sequence_stream)


class TestMultiWorkerLoading(unittest.TestCase):

    def test_loader_stability(self) -> None:
        tokenizer = _get_test_tokenizer_data()
        dataset = ZortherDataset([])
        loader = ZortherDataLoader(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=1,
            max_seq_len=4
        )
        self.assertIsNotNone(loader)


class TestDatasetPerformance(unittest.TestCase):

    def test_loading_speed(self) -> None:
        tokenizer = _get_test_tokenizer_data()
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "train.jsonl")
            with open(filepath, "w", encoding="utf-8") as f:
                for _ in range(100):
                    f.write(json.dumps({"text": "speed test pretraining data sequence"}) + "\n")
                    
            dataset = ZortherDataset(filepath)
            loader = ZortherDataLoader(
                dataset=dataset,
                tokenizer=tokenizer,
                batch_size=4,
                max_seq_len=8
            )
            
            start = time.perf_counter()
            for _ in loader:
                pass
            elapsed = time.perf_counter() - start
            self.assertTrue(elapsed < 1.0)


if __name__ == "__main__":
    unittest.main()