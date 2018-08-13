import unittest

from unittest.mock import Mock, call

from torchbearer.metrics import RunningMean, Metric, RunningMetric, Mean, Std

import torch


class TestStd(unittest.TestCase):
    def setUp(self):
        self._metric = Metric('test')
        self._metric.process = Mock()
        self._metric.process.side_effect = [torch.zeros(torch.Size([])),
                                            torch.FloatTensor([0.1, 0.2, 0.3]),
                                            torch.FloatTensor([0.4, 0.5, 0.6]),
                                            torch.FloatTensor([0.7, 0.8, 0.9]),
                                            torch.ones(torch.Size([]))]

        self._std = Std('test')
        self._std.reset({})
        self._target = 0.31622776601684

    def test_train(self):
        self.setUp()
        self._std.train()
        for i in range(5):
            self._std.process(self._metric.process())
        result = self._std.process_final({})
        self.assertAlmostEqual(self._target, result)

    def test_validate(self):
        self.setUp()
        self._std.eval()
        for i in range(5):
            self._std.process(self._metric.process())
        result = self._std.process_final({})
        self.assertAlmostEqual(self._target, result)

    def test_precision_error(self):
        self.setUp()
        self._std.train()
        val = torch.tensor([0.55])
        for i in range(2):
            self._std.process(val)

        result = self._std.process_final({})
        self.assertEqual(0, result)

    def setUpMoreDims(self):
        self._metric = Metric('test')
        self._metric.process = Mock()
        self._metric.process.side_effect = [torch.zeros(torch.Size([])),
                                            torch.FloatTensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]),
                                            torch.FloatTensor([[0.4, 0.5, 0.6], [1.4, 1.5, 1.6]]),
                                            torch.FloatTensor([[0.7, 0.8, 0.9], [1.7, 1.8, 1.9]]),
                                            torch.ones(torch.Size([]))]
        self._std = Std('test')
        self._std.reset({})
        self._target = 0.57662804083742

    def test_more_dims(self):
        self.setUpMoreDims()
        for i in range(5):
            self._std.process(self._metric.process())
        result = self._std.process_final({})
        self.assertAlmostEqual(self._target, result)


class TestMean(unittest.TestCase):
    def setUp(self):
        self._metric = Metric('test')
        self._metric.process = Mock()
        self._metric.process.side_effect = [torch.zeros(torch.Size([])),
                                            torch.FloatTensor([0.1, 0.2, 0.3]),
                                            torch.FloatTensor([0.4, 0.5, 0.6]),
                                            torch.FloatTensor([0.7, 0.8, 0.9]),
                                            torch.ones(torch.Size([]))]

        self._mean = Mean('test')
        self._mean.reset({})
        self._target = 0.5

    def test_train_dict(self):
        self.setUp()
        self._mean.train()
        for i in range(5):
            self._mean.process(self._metric.process())
        result = self._mean.process_final({})
        self.assertAlmostEqual(self._target, result)

    def test_validate_dict(self):
        self.setUp()
        self._mean.eval()
        for i in range(5):
            self._mean.process(self._metric.process())
        result = self._mean.process_final({})
        self.assertAlmostEqual(self._target, result)

    def setUpMoreDims(self):
        self._metric = Metric('test')
        self._metric.process = Mock()
        self._metric.process.side_effect = [torch.zeros(torch.Size([])),
                                            torch.FloatTensor([[0.1, 0.2, 0.3], [1.1, 1.2, 1.3]]),
                                            torch.FloatTensor([[0.4, 0.5, 0.6], [1.4, 1.5, 1.6]]),
                                            torch.FloatTensor([[0.7, 0.8, 0.9], [1.7, 1.8, 1.9]]),
                                            torch.ones(torch.Size([]))]
        self._mean = Mean('test')
        self._mean.reset({})
        self._target = 0.95

    def test_more_dims(self):
        self.setUpMoreDims()
        for i in range(5):
            self._mean.process(self._metric.process())
        result = self._mean.process_final({})
        self.assertAlmostEqual(self._target, result)


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