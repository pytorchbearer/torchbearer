import unittest

from unittest.mock import Mock, call

from bink.metrics import Std, BasicMetric, Mean, BatchLambda, EpochLambda

import torch

class TestStd(unittest.TestCase):
    def setUp(self):
        self._metric = BasicMetric('test')
        self._metric.train = Mock()
        self._metric.train.side_effect = [torch.FloatTensor([0.1, 0.2, 0.3]),
                                          torch.FloatTensor([0.4, 0.5, 0.6]),
                                          torch.FloatTensor([0.7, 0.8, 0.9])]

        self._metric.validate = Mock()
        self._metric.validate.side_effect = [torch.FloatTensor([0.1, 0.2, 0.3]),
                                          torch.FloatTensor([0.4, 0.5, 0.6]),
                                          torch.FloatTensor([0.7, 0.8, 0.9])]

        self._std = Std(self._metric)
        self._std.reset({})
        self._target = 0.25819888974716

    def test_train_dict(self):
        for i in range(3):
            self._std.train_dict({})
        result = self._std.final_train_dict({})
        self.assertTrue('train_test_std' in result, msg='train_test_std is not a key in: ' + str(result))
        self.assertAlmostEqual(self._target, result['train_test_std'])

    def test_validate_dict(self):
        for i in range(3):
            self._std.validate_dict({})
        result = self._std.final_validate_dict({})
        self.assertTrue('val_test_std' in result, msg='val_test_std is not a key in: ' + str(result))
        self.assertAlmostEqual(self._target, result['val_test_std'])


class TestMean(unittest.TestCase):
    def setUp(self):
        self._metric = BasicMetric('test')
        self._metric.train = Mock()
        self._metric.train.side_effect = [torch.FloatTensor([0.1, 0.2, 0.3]),
                                          torch.FloatTensor([0.4, 0.5, 0.6]),
                                          torch.FloatTensor([0.7, 0.8, 0.9])]

        self._metric.validate = Mock()
        self._metric.validate.side_effect = [torch.FloatTensor([0.1, 0.2, 0.3]),
                                          torch.FloatTensor([0.4, 0.5, 0.6]),
                                          torch.FloatTensor([0.7, 0.8, 0.9])]

        self._std = Mean(self._metric)
        self._std.reset({})
        self._target = 0.5

    def test_train_dict(self):
        for i in range(3):
            self._std.train_dict({})
        result = self._std.final_train_dict({})
        self.assertTrue('train_test' in result, msg='train_test is not a key in: ' + str(result))
        self.assertAlmostEqual(self._target, result['train_test'])

    def test_validate_dict(self):
        for i in range(3):
            self._std.validate_dict({})
        result = self._std.final_validate_dict({})
        self.assertTrue('val_test' in result, msg='val_test is not a key in: ' + str(result))
        self.assertAlmostEqual(self._target, result['val_test'])


class TestBatchLambda(unittest.TestCase):
    def setUp(self):
        self._metric_function = Mock(return_value='test')
        self._metric = BatchLambda('test', self._metric_function)
        self._states = [{'y_true': 1, 'y_pred': 2},
                        {'y_true': 3, 'y_pred': 4},
                        {'y_true': 5, 'y_pred': 6}]

    def test_train(self):
        calls = []
        for i in range(len(self._states)):
            self._metric.train(self._states[i])
            calls.append(call(self._states[i]['y_true'], self._states[i]['y_pred']))
        self._metric_function.assert_has_calls(calls)

    def test_validate(self):
        calls = []
        for i in range(len(self._states)):
            self._metric.validate(self._states[i])
            calls.append(call(self._states[i]['y_true'], self._states[i]['y_pred']))
        self._metric_function.assert_has_calls(calls)


class TestEpochLambda(unittest.TestCase):
    def setUp(self):
        self._metric_function = Mock(return_value='test')
        self._metric = EpochLambda('test', self._metric_function, step_size=3)
        self._metric.reset({})
        self._states = [{'t': 0, 'y_true': torch.LongTensor([0]), 'y_pred': torch.FloatTensor([0.0])},
                        {'t': 1, 'y_true': torch.LongTensor([1]), 'y_pred': torch.FloatTensor([0.1])},
                        {'t': 2, 'y_true': torch.LongTensor([2]), 'y_pred': torch.FloatTensor([0.2])},
                        {'t': 3, 'y_true': torch.LongTensor([3]), 'y_pred': torch.FloatTensor([0.3])},
                        {'t': 4, 'y_true': torch.LongTensor([4]), 'y_pred': torch.FloatTensor([0.4])}]

    def test_train(self):
        calls = [[torch.LongTensor([0]), torch.FloatTensor([0.0])],
                 [torch.LongTensor([0, 1, 2, 3]), torch.FloatTensor([0.0, 0.1, 0.2, 0.3])]]
        for i in range(len(self._states)):
            self._metric.train(self._states[i])
        self.assertEqual(2, len(self._metric_function.call_args_list))
        for i in range(len(self._metric_function.call_args_list)):
            self.assertTrue(torch.eq(self._metric_function.call_args_list[i][0][0], calls[i][0]).all)
            self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[i][0][1], -calls[i][1])), 1e-12).all)
        self._metric_function.reset_mock()
        self._metric.final_train({})

        self._metric_function.assert_called_once()
        self.assertTrue(torch.eq(self._metric_function.call_args_list[0][0][0], torch.LongTensor([0, 1, 2, 3, 4])).all)
        self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[0][0][1], -torch.FloatTensor([0.0, 0.1, 0.2, 0.3, 0.4]))), 1e-12).all)

    def test_validate(self):
        for i in range(len(self._states)):
            self._metric.validate(self._states[i])
        self._metric_function.assert_not_called()
        self._metric.final_validate({})

        self._metric_function.assert_called_once()
        self.assertTrue(torch.eq(self._metric_function.call_args_list[0][0][0], torch.LongTensor([0, 1, 2, 3, 4])).all)
        self.assertTrue(torch.lt(torch.abs(torch.add(self._metric_function.call_args_list[0][0][1], -torch.FloatTensor([0.0, 0.1, 0.2, 0.3, 0.4]))), 1e-12).all)
