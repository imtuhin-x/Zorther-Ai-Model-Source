import os
import tempfile
import unittest
import torch
import torch.nn as nn

from zorther.config.model_config import ZortherModelConfig
from zorther.model.transformer import ZortherTransformer
from zorther.model.embeddings import ZortherEmbeddings, RotaryPositionalEmbedding
from zorther.model.layers import RMSNorm, TransformerBlock
from zorther.model.cache import CacheConfig, CacheManager


class TestModelConfig(unittest.TestCase):

    def test_config_initialization(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        self.assertEqual(config.vocab_size, 1000)
        self.assertEqual(config.hidden_size, 128)
        self.assertEqual(config.num_layers, 2)
        self.assertTrue(config.intermediate_size > 0)


class TestModelInitialization(unittest.TestCase):

    def test_model_build(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        self.assertIsNotNone(model)
        self.assertIsInstance(model.embeddings, ZortherEmbeddings)
        self.assertEqual(len(model.layers), 2)


class TestEmbedding(unittest.TestCase):

    def test_embedding_output(self) -> None:
        config = ZortherModelConfig(vocab_size=1000, hidden_size=128)
        embeddings = ZortherEmbeddings(config.vocab_size, config.hidden_size)
        
        input_ids = torch.randint(0, 1000, (2, 10))
        output = embeddings(input_ids)
        
        self.assertEqual(output.shape, (2, 10, 128))


class TestForwardPass(unittest.TestCase):

    def test_forward_logits(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 10))
        
        logits = model(input_ids)
        self.assertEqual(logits.shape, (2, 10, 1000))


class TestOutputShape(unittest.TestCase):

    def test_varying_sequence_lengths(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        
        for seq_len in [1, 5, 32]:
            input_ids = torch.randint(0, 1000, (1, seq_len))
            logits = model(input_ids)
            self.assertEqual(logits.shape, (1, seq_len, 1000))


class TestAttentionIntegration(unittest.TestCase):

    def test_attention_masking(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        input_ids = torch.randint(0, 1000, (1, 10))
        
        logits = model(input_ids)
        self.assertFalse(torch.isnan(logits).any())


class TestGradientFlow(unittest.TestCase):

    def test_backward_pass(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        input_ids = torch.randint(0, 1000, (1, 10))
        targets = torch.randint(0, 1000, (1, 10))
        
        logits, loss = model(input_ids, targets=targets)
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())


class TestLossCompatibility(unittest.TestCase):

    def test_loss_values(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        input_ids = torch.randint(0, 1000, (2, 10))
        targets = torch.randint(0, 1000, (2, 10))
        
        _, loss = model(input_ids, targets=targets)
        self.assertTrue(loss.item() > 0.0)


class TestTrainStep(unittest.TestCase):

    def test_parameter_updates(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        input_ids = torch.randint(0, 1000, (1, 10))
        targets = torch.randint(0, 1000, (1, 10))
        
        before_params = [p.clone() for p in model.parameters() if p.requires_grad]
        
        _, loss = model(input_ids, targets=targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        after_params = [p for p in model.parameters() if p.requires_grad]
        
        updated = False
        for b, a in zip(before_params, after_params):
            if not torch.equal(b, a):
                updated = True
                break
                
        self.assertTrue(updated)


class TestSaveLoad(unittest.TestCase):

    def test_state_serialization(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model1 = ZortherTransformer(config)
        model2 = ZortherTransformer(config)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "model.pt")
            torch.save(model1.state_dict(), filepath)
            
            model2.load_state_dict(torch.load(filepath))
            
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            self.assertTrue(torch.equal(p1, p2))


class TestGeneration(unittest.TestCase):

    def test_generation_step(self) -> None:
        config = ZortherModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_seq_len=256
        )
        model = ZortherTransformer(config)
        input_ids = torch.randint(0, 1000, (1, 5))
        
        cache_config = CacheConfig(
            max_batch_size=1,
            max_seq_len=256,
            num_key_value_heads=2,
            head_dim=32
        )
        kv_caches = CacheManager(num_layers=2, cache_type="dynamic", config=cache_config)
        
        logits = model(input_ids, kv_caches=kv_caches.caches, start_pos=0)
        self.assertEqual(logits.shape, (1, 5, 1000))


if __name__ == "__main__":
    unittest.main()