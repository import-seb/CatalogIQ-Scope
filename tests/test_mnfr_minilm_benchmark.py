"""Step-budget instrumentation tests using recording doubles, not neural models."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from catalogiq.mnfr_minilm_benchmark import BenchmarkComplete, MnfrBenchmark
from catalogiq.segment_transformer import ModelConfig


class Tensor:
    def __init__(self, value):
        self.value = np.asarray(value)
    def detach(self):
        return self
    def cpu(self):
        return self
    def numpy(self):
        return self.value
    def __float__(self):
        return float(self.value)


class Loss:
    weight = None
    label_smoothing = 0
    value = 0.5
    def forward(self, logits, labels):
        return Tensor(self.value)


class Optimizer:
    def __init__(self):
        self.param_groups = [{"lr": 3e-5}]
        self.hooks = []
        self.updates = 0
    def register_step_post_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
    def step(self):
        self.updates += 1
        for hook in self.hooks:
            hook(self, (), {})


class Scheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.calls = 0
        self.step()
    def step(self):
        self.calls += 1


class Generator:
    def __init__(self, device):
        pass
    def manual_seed(self, seed):
        return self


FAKE_TORCH = SimpleNamespace(nn=SimpleNamespace(CrossEntropyLoss=Loss), Generator=Generator,
    randperm=lambda size, generator: Tensor(np.arange(size)),
    optim=SimpleNamespace(lr_scheduler=SimpleNamespace(LambdaLR=Scheduler)),
    cuda=SimpleNamespace(is_available=lambda: False))


class MnfrBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.config = ModelConfig(sequence_length=256)
        self.corpus = SimpleNamespace(label_ids=np.arange(640) % 2, label_names=("All others", "J&J"))
        self.enterContext(patch("catalogiq.mnfr_minilm_benchmark._memory_usage", return_value={"process_rss_bytes": 123}))

    def monitor(self):
        return MnfrBenchmark(self.corpus, np.arange(640), self.config, 20, 5181, torch=FAKE_TORCH)

    def test_exactly_twenty_optimizer_and_scheduler_steps_then_restore_hooks(self):
        original_loss, original_step = Loss.forward, Scheduler.step
        config_before = self.config.to_dict()
        monitor = self.monitor()
        optimizer = Optimizer()
        with self.assertRaises(BenchmarkComplete):
            with monitor:
                scheduler = Scheduler(optimizer)
                monitor.mark_initialized()
                for index in range(100):
                    labels = Tensor(self.corpus.label_ids[monitor.batches[index]])
                    Loss().forward(SimpleNamespace(shape=(32, 2)), labels)
                    optimizer.step()
                    scheduler.step()
        report = monitor.report()
        self.assertEqual(optimizer.updates, 20)
        self.assertEqual(scheduler.calls, 21)  # Includes its construction-time step.
        self.assertEqual(report["completed_optimizer_steps"], 20)
        self.assertEqual(report["completed_scheduler_steps"], 20)
        self.assertTrue(report["all_losses_finite"])
        self.assertTrue(report["all_batch_labels_verified"])
        self.assertEqual(report["full_schedule_optimizer_steps"], 5181)
        self.assertGreater(report["mean_training_step_seconds"], 0)
        self.assertAlmostEqual(report["estimated_full_training_step_seconds"], report["mean_training_step_seconds"] * 5181)
        self.assertEqual(self.config.to_dict(), config_before)
        self.assertIs(Loss.forward, original_loss)
        self.assertIs(Scheduler.step, original_step)
        self.assertEqual(optimizer.hooks, [])

    def test_incorrect_label_mapping_stops_before_optimizer_update(self):
        monitor = self.monitor()
        optimizer = Optimizer()
        with self.assertRaisesRegex(ValueError, "approved seeded Mnfr batches"):
            with monitor:
                Scheduler(optimizer)
                monitor.mark_initialized()
                labels = Tensor(1 - self.corpus.label_ids[monitor.batches[0]])
                Loss().forward(SimpleNamespace(shape=(32, 2)), labels)
        self.assertEqual(optimizer.updates, 0)
        self.assertEqual(optimizer.hooks, [])

    def test_nonfinite_loss_or_class_weighting_is_rejected(self):
        for attribute, value, error in (("value", float("nan"), FloatingPointError), ("weight", object(), ValueError)):
            with self.subTest(attribute=attribute), patch.object(Loss, attribute, value):
                monitor = self.monitor()
                with self.assertRaises(error):
                    with monitor:
                        Scheduler(Optimizer())
                        monitor.mark_initialized()
                        Loss().forward(SimpleNamespace(shape=(32, 2)), Tensor(self.corpus.label_ids[monitor.batches[0]]))

    def test_scheduler_without_optimizer_update_is_not_counted_as_training(self):
        monitor = self.monitor()
        with self.assertRaisesRegex(RuntimeError, "step alignment"):
            with monitor:
                scheduler = Scheduler(Optimizer())
                monitor.mark_initialized()
                Loss().forward(SimpleNamespace(shape=(32, 2)), Tensor(self.corpus.label_ids[monitor.batches[0]]))
                scheduler.step()
        self.assertEqual(monitor.optimizer_steps, 0)


if __name__ == "__main__":
    unittest.main()
