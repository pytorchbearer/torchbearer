import unittest

import torch

import torchbearer
from torchbearer.callbacks import Callback
from torchbearer.metrics.primitives import CategoricalAccuracy


class TestMetric(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.Y_TRUE: torch.LongTensor([0, 1, 2, 2, 1]),
            torchbearer.Y_PRED: torch.FloatTensor([
                [0.9, 0.1, 0.1], # Correct
                [0.1, 0.9, 0.1], # Correct
                [0.1, 0.1, 0.9], # Correct
                [0.9, 0.1, 0.1], # Incorrect
                [0.9, 0.1, 0.1], # Incorrect
            ])
        }
        self._state[torchbearer.Y_PRED].requires_grad = True
        self._targets = [1, 1, 1, 0, 0]
        self._metric = CategoricalAccuracy().root

    def test_requires_grad(self):
        result = self._metric.process(self._state)
        self.assertTrue(self._state[torchbearer.Y_PRED].requires_grad is True)
        self.assertTrue(result.requires_grad is False)


class TestCallback(unittest.TestCase):
    def test_state_dict(self):
        callback = Callback()

        self.assertEqual(callback.state_dict(), {})
        self.assertEqual(callback.load_state_dict({}), callback)

    def test_str(self):
        callback = Callback()
        self.assertEqual(str(callback).strip(), "torchbearer.bases.Callback")

    def test_empty_methods(self):
        callback = Callback()

        self.assertIsNone(callback.on_start({}))
        self.assertIsNone(callback.on_start_epoch({}))
        self.assertIsNone(callback.on_start_training({}))
        self.assertIsNone(callback.on_sample({}))
        self.assertIsNone(callback.on_forward({}))
        self.assertIsNone(callback.on_criterion({}))
        self.assertIsNone(callback.on_backward({}))
        self.assertIsNone(callback.on_step_training({}))
        self.assertIsNone(callback.on_end_training({}))
        self.assertIsNone(callback.on_end_epoch({}))
        self.assertIsNone(callback.on_checkpoint({}))
        self.assertIsNone(callback.on_end({}))
        self.assertIsNone(callback.on_start_validation({}))
        self.assertIsNone(callback.on_sample_validation({}))
        self.assertIsNone(callback.on_forward_validation({}))
        self.assertIsNone(callback.on_end_validation({}))
        self.assertIsNone(callback.on_step_validation({}))
        self.assertIsNone(callback.on_criterion_validation({}))
