import os
import tempfile
import unittest
import torch
import torch.nn as nn

from zorther.optimization.optimizer import (
    OptimizerConfig,
    OptimizerFactory,
    SchedulerFactory,
    GradientClipper,
    GradientAccumulator,
    OptimizerManager,
    Lion,
    SophiaG
)


class TestOptimizerCreation(unittest.TestCase):

    def test_factory_creation(self) -> None:
        model = nn.Linear(10, 2)
        
        for opt_type in ["adamw", "sgd", "lion", "sophiag"]:
            config = OptimizerConfig(optimizer_type=opt_type)
            optimizer = OptimizerFactory.create(config, model.parameters())
            self.assertIsNotNone(optimizer)


class TestAdamWOptimizer(unittest.TestCase):

    def test_adamw_step_logic(self) -> None:
        model = nn.Linear(10, 2)
        config = OptimizerConfig(optimizer_type="adamw", learning_rate=1e-3)
        optimizer = OptimizerFactory.create(config, model.parameters())
        
        loss = model(torch.randn(1, 10)).sum()
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
        self.assertTrue(len(optimizer.state) > 0)


class TestOptimizerStep(unittest.TestCase):

    def test_parameter_modification(self) -> None:
        model = nn.Linear(5, 1)
        config = OptimizerConfig(optimizer_type="lion", learning_rate=1e-2)
        optimizer = OptimizerFactory.create(config, model.parameters())
        
        initial_weight = model.weight.clone()
        
        loss = model(torch.randn(1, 5)).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        self.assertFalse(torch.equal(initial_weight, model.weight))


class TestGradientClipping(unittest.TestCase):

    def test_clipping_threshold(self) -> None:
        model = nn.Linear(5, 1)
        nn.init.constant_(model.weight, 10.0)
        
        loss = model(torch.randn(1, 5)).sum() * 1000.0
        loss.backward()
        
        GradientClipper.clip(model.parameters(), 1.0)
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        self.assertTrue(total_norm <= 1.01)


class TestScheduler(unittest.TestCase):

    def test_constant_scheduler(self) -> None:
        model = nn.Linear(5, 1)
        config = OptimizerConfig(scheduler_type="constant", learning_rate=1e-3)
        optimizer = OptimizerFactory.create(config, model.parameters())
        scheduler = SchedulerFactory.create(optimizer, config)
        
        for _ in range(5):
            lr = scheduler.step()
            self.assertEqual(lr, 1e-3)


class TestWarmupScheduler(unittest.TestCase):

    def test_warmup_increments(self) -> None:
        model = nn.Linear(5, 1)
        config = OptimizerConfig(scheduler_type="warmup", learning_rate=1e-3, warmup_steps=10)
        optimizer = OptimizerFactory.create(config, model.parameters())
        scheduler = SchedulerFactory.create(optimizer, config)
        
        prev_lr = 0.0
        for _ in range(5):
            lr = scheduler.step()
            self.assertTrue(lr > prev_lr)
            prev_lr = lr


class TestCosineScheduler(unittest.TestCase):

    def test_cosine_decay(self) -> None:
        model = nn.Linear(5, 1)
        config = OptimizerConfig(scheduler_type="cosine", learning_rate=1e-3, warmup_steps=2, total_steps=10)
        optimizer = OptimizerFactory.create(config, model.parameters())
        scheduler = SchedulerFactory.create(optimizer, config)
        
        for _ in range(2):
            scheduler.step()
            
        lr_at_peak = scheduler.step()
        lr_after_decay = scheduler.step()
        
        self.assertTrue(lr_after_decay < lr_at_peak)


class TestGradientAccumulation(unittest.TestCase):

    def test_step_trigger(self) -> None:
        accumulator = GradientAccumulator(steps=4)
        
        self.assertFalse(accumulator.should_step())
        self.assertFalse(accumulator.should_step())
        self.assertFalse(accumulator.should_step())
        self.assertTrue(accumulator.should_step())


class TestOptimizerStateSaveLoad(unittest.TestCase):

    def test_state_restoration(self) -> None:
        model = nn.Linear(5, 1)
        config = OptimizerConfig(optimizer_type="adamw")
        optimizer1 = OptimizerFactory.create(config, model.parameters())
        optimizer2 = OptimizerFactory.create(config, model.parameters())
        
        loss = model(torch.randn(1, 5)).sum()
        loss.backward()
        optimizer1.step()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "opt.pt")
            torch.save(optimizer1.state_dict(), filepath)
            
            optimizer2.load_state_dict(torch.load(filepath))
            
        self.assertEqual(optimizer1.state_dict().keys(), optimizer2.state_dict().keys())


class TestLossBackward(unittest.TestCase):

    def test_backward_execution(self) -> None:
        model = nn.Linear(5, 1)
        loss_fn = nn.MSELoss()
        
        inputs = torch.randn(1, 5)
        targets = torch.randn(1, 1)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        
        for param in model.parameters():
            self.assertIsNotNone(param.grad)


class TestTrainingStability(unittest.TestCase):

    def test_monotonic_decrease(self) -> None:

        torch.manual_seed(42)

        model = nn.Linear(10, 1)

        nn.init.normal_(
            model.weight,
            std=0.01
        )

        config = OptimizerConfig(
            optimizer_type="adamw",
            learning_rate=1e-3
        )


        manager = OptimizerManager(
            model,
            config,
            accumulation_steps=1
        )


        inputs = torch.randn(
            32,
            10
        )

        targets = torch.randn(
            32,
            1
        )


        losses = []


        for _ in range(100):

            outputs = model(inputs)

            loss = torch.nn.functional.mse_loss(
                outputs,
                targets
            )


            losses.append(
                loss.item()
            )


            manager.step(loss)



        initial_loss = losses[0]

        final_loss = losses[-1]


        # Training improve হয়েছে কিনা check
        self.assertTrue(
            final_loss < initial_loss
        )


if __name__ == "__main__":
    unittest.main()