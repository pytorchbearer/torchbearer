import unittest

from unittest.mock import Mock, call

from torch.autograd import Variable

from bink.metrics import Std, Metric, Mean, BatchLambda, EpochLambda

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
        self._states = [{'y_true': Variable(torch.FloatTensor([1])), 'y_pred': Variable(torch.FloatTensor([2]))},
                        {'y_true': Variable(torch.FloatTensor([3])), 'y_pred': Variable(torch.FloatTensor([4]))},
                        {'y_true': Variable(torch.FloatTensor([5])), 'y_pred': Variable(torch.FloatTensor([6]))}]

    def test_train(self):
        self._metric.train()
        calls = []
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
            calls.append(call(self._states[i]['y_true'].data, self._states[i]['y_pred'].data))
        self._metric_function.assert_has_calls(calls)

    def test_validate(self):
        self._metric.eval()
        calls = []
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
            calls.append(call(self._states[i]['y_true'].data, self._states[i]['y_pred'].data))
        self._metric_function.assert_has_calls(calls)


class TestEpochLambda(unittest.TestCase):
    def setUp(self):
        self._metric_function = Mock(return_value='test')
        self._metric = EpochLambda('test', self._metric_function, step_size=3)
        self._metric.reset({})
        self._states = [{'t': 0, 'y_true': torch.LongTensor([0]), 'y_pred': torch.FloatTensor([0.0]), 'device': 'cpu'},
                        {'t': 1, 'y_true': torch.LongTensor([1]), 'y_pred': torch.FloatTensor([0.1]), 'device': 'cpu'},
                        {'t': 2, 'y_true': torch.LongTensor([2]), 'y_pred': torch.FloatTensor([0.2]), 'device': 'cpu'},
                        {'t': 3, 'y_true': torch.LongTensor([3]), 'y_pred': torch.FloatTensor([0.3]), 'device': 'cpu'},
                        {'t': 4, 'y_true': torch.LongTensor([4]), 'y_pred': torch.FloatTensor([0.4]), 'device': 'cpu'}]

    def test_train(self):
        self._metric.train()
        calls = [[torch.LongTensor([0]), torch.FloatTensor([0.0])],
                 [torch.LongTensor([0, 1, 2, 3]), torch.FloatTensor([0.0, 0.1, 0.2, 0.3])]]
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
        self.assertEqual(2, len(self._metric_function.call_args_list))
        for i in range(len(self._metric_function.call_args_list)):
            self.assertTrue(torch.eq(self._metric_function.call_args_list[i][0][0], calls[i][0]).all)
            self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[i][0][1], -calls[i][1])), 1e-12).all)
        self._metric_function.reset_mock()
        self._metric.process_final({})

        self._metric_function.assert_called_once()
        self.assertTrue(torch.eq(self._metric_function.call_args_list[0][0][0], torch.LongTensor([0, 1, 2, 3, 4])).all)
        self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[0][0][1], -torch.FloatTensor([0.0, 0.1, 0.2, 0.3, 0.4]))), 1e-12).all)

    def test_validate(self):
        self._metric.eval()
        for i in range(len(self._states)):
            self._metric.process(self._states[i])
        self._metric_function.assert_not_called()
        self._metric.process_final_validate({})

        self._metric_function.assert_called_once()
        self.assertTrue(torch.eq(self._metric_function.call_args_list[0][0][0], torch.LongTensor([0, 1, 2, 3, 4])).all)
        self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[0][0][1], -torch.FloatTensor([0.0, 0.1, 0.2, 0.3, 0.4]))), 1e-12).all)
