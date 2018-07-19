import unittest

from unittest.mock import Mock, call

from torchbearer.metrics import RunningMean, Metric, RunningMetric

import torch


class TestRunningMetric(unittest.TestCase):
    def setUp(self):
        self._metric = RunningMetric('test', batch_size=5, step_size=5)
        self._metric.reset({})
        self._metric._process_train = Mock(return_value=3)
        self._metric._step = Mock(return_value='output')

    def test_train_called_with_state(self):
        self._metric.train()
        self._metric.process({'test': -1})
        self._metric._process_train.assert_called_with({'test': -1})

    def test_cache_one_step(self):
        self._metric.train()
        for i in range(6):
            self._metric.process({})
        self._metric._step.assert_has_calls([call([3]), call([3, 3, 3, 3, 3])])

    def test_empty_methods(self):
        metric = RunningMetric('test')
        self.assertTrue(metric._step(['test']) is None)
        self.assertTrue(metric._process_train(['test']) is None)


class TestRunningMean(unittest.TestCase):
    def setUp(self):
        self._metric = Metric('test')
        self._mean = RunningMean('test')
        self._cache = [1.0, 1.5, 2.0]
        self._target = 1.5

    def test_train(self):
        result = self._mean._process_train(torch.FloatTensor([1.0, 1.5, 2.0]))
        self.assertAlmostEqual(self._target, result, 3, 0.002)

    def test_step(self):
        result = self._mean._step(self._cache)
        self.assertEqual(self._target, result)