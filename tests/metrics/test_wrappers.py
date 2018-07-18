import unittest

from unittest.mock import Mock, call

from torch.autograd import Variable

import torchbearer
from torchbearer.metrics import Std, Metric, Mean, BatchLambda, EpochLambda

import torch


class TestStd(unittest.TestCase):
    def setUp(self):
        self._metric = Metric('test')
        self._metric.process = Mock()
        self._metric.process.side_effect = [torch.FloatTensor([0.1, 0.2, 0.3]),
                                            torch.FloatTensor([0.4, 0.5, 0.6]),
                                            torch.FloatTensor([0.7, 0.8, 0.9])]

        self._std = Std(self._metric)
        self._std.reset({})
        self._target = 0.25819888974716

    def test_train(self):
        self._std.train()
        for i in range(3):
            self._std.process({})
        result = self._std.process_final({})
        self.assertAlmostEqual(self._target, result)

    def test_validate(self):
        self._std.eval()
        for i in range(3):
            self._std.process({})
        result = self._std.process_final({})
        self.assertAlmostEqual(self._target, result)


class TestMean(unittest.TestCase):
    def setUp(self):
        self._metric = Metric('test')
        self._metric.process = Mock()
        self._metric.process.side_effect = [torch.FloatTensor([0.1, 0.2, 0.3]),
                                            torch.FloatTensor([0.4, 0.5, 0.6]),
                                            torch.FloatTensor([0.7, 0.8, 0.9])]

        self._mean = Mean(self._metric)
        self._mean.reset({})
        self._target = 0.5

    def test_train_dict(self):
        self._mean.train()
        for i in range(3):
            self._mean.process({})
        result = self._mean.process_final({})
        self.assertAlmostEqual(self._target, result)

    def test_validate_dict(self):
        self._mean.eval()
        for i in range(3):
            self._mean.process({})
        result = self._mean.process_final({})
        self.assertAlmostEqual(self._target, result)


class TestBatchLambda(unittest.TestCase):
    def setUp(self):
        self._metric_function = Mock(return_value='test')
        self._metric = BatchLambda('test', self._metric_function)
        self._states = [{torchbearer.Y_TRUE: Variable(torch.FloatTensor([1])), torchbearer.Y_PRED: Variable(torch.FloatTensor([2]))},
                        {torchbearer.Y_TRUE: Variable(torch.FloatTensor([3])), torchbearer.Y_PRED: Variable(torch.FloatTensor([4]))},
                        {torchbearer.Y_TRUE: Variable(torch.FloatTensor([5])), torchbearer.Y_PRED: Variable(torch.FloatTensor([6]))}]

    def test_train(self):
        self._metric.train()
        calls = []
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
            calls.append(call(self._states[i][torchbearer.Y_PRED].data, self._states[i][torchbearer.Y_TRUE].data))
        self._metric_function.assert_has_calls(calls)

    def test_validate(self):
        self._metric.eval()
        calls = []
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
            calls.append(call(self._states[i][torchbearer.Y_PRED].data, self._states[i][torchbearer.Y_TRUE].data))
        self._metric_function.assert_has_calls(calls)


class TestEpochLambda(unittest.TestCase):
    def setUp(self):
        self._metric_function = Mock(return_value='test')
        self._metric = EpochLambda('test', self._metric_function, step_size=3)
        self._metric.reset({torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float32})
        self._states = [{torchbearer.BATCH: 0, torchbearer.Y_TRUE: torch.LongTensor([0]), torchbearer.Y_PRED: torch.FloatTensor([0.0]), torchbearer.DEVICE: 'cpu'},
                        {torchbearer.BATCH: 1, torchbearer.Y_TRUE: torch.LongTensor([1]), torchbearer.Y_PRED: torch.FloatTensor([0.1]), torchbearer.DEVICE: 'cpu'},
                        {torchbearer.BATCH: 2, torchbearer.Y_TRUE: torch.LongTensor([2]), torchbearer.Y_PRED: torch.FloatTensor([0.2]), torchbearer.DEVICE: 'cpu'},
                        {torchbearer.BATCH: 3, torchbearer.Y_TRUE: torch.LongTensor([3]), torchbearer.Y_PRED: torch.FloatTensor([0.3]), torchbearer.DEVICE: 'cpu'},
                        {torchbearer.BATCH: 4, torchbearer.Y_TRUE: torch.LongTensor([4]), torchbearer.Y_PRED: torch.FloatTensor([0.4]), torchbearer.DEVICE: 'cpu'}]

    def test_train(self):
        self._metric.train()
        calls = [[torch.FloatTensor([0.0]), torch.LongTensor([0])],
                 [torch.FloatTensor([0.0, 0.1, 0.2, 0.3]), torch.LongTensor([0, 1, 2, 3])]]
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
        self.assertEqual(2, len(self._metric_function.call_args_list))
        for i in range(len(self._metric_function.call_args_list)):
            self.assertTrue(torch.eq(self._metric_function.call_args_list[i][0][0], calls[i][0]).all)
            self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[i][0][1], -calls[i][1])), 1e-12).all)
        self._metric_function.reset_mock()
        self._metric.process_final({})

        self._metric_function.assert_called_once()
        self.assertTrue(torch.eq(self._metric_function.call_args_list[0][0][1], torch.LongTensor([0, 1, 2, 3, 4])).all)
        self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[0][0][0], -torch.FloatTensor([0.0, 0.1, 0.2, 0.3, 0.4]))), 1e-12).all)

    def test_validate(self):
        self._metric.eval()
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
        self._metric_function.assert_not_called()
        self._metric.process_final_validate({})

        self._metric_function.assert_called_once()
        self.assertTrue(torch.eq(self._metric_function.call_args_list[0][0][1], torch.LongTensor([0, 1, 2, 3, 4])).all)
        self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[0][0][0], -torch.FloatTensor([0.0, 0.1, 0.2, 0.3, 0.4]))), 1e-12).all)
