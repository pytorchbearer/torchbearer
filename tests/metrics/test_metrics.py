import unittest
from unittest.mock import Mock

from torchbearer.metrics import MetricList, Metric


class TestMetricList(unittest.TestCase):
    def test_default_acc(self):
        metric = MetricList(['acc'])
        self.assertTrue(metric.metric_list[0].name == 'running_acc', msg='running_acc not in: ' + str(metric.metric_list))

    def test_default_loss(self):
        metric = MetricList(['loss'])
        self.assertTrue(metric.metric_list[0].name == 'running_loss', msg='running_loss not in: ' + str(metric.metric_list))

    def test_process(self):
        my_mock = Metric('test')
        my_mock.process = Mock(return_value=-1)
        metric = MetricList([my_mock])
        result = metric.process({'state': -1})
        self.assertEqual({'test': -1}, result)
        my_mock.process.assert_called_once_with({'state': -1})

    def test_process_final(self):
        my_mock = Metric('test')
        my_mock.process_final = Mock(return_value=-1)
        metric = MetricList([my_mock])
        result = metric.process_final({'state': -1})
        self.assertEqual({'test': -1}, result)
        my_mock.process_final.assert_called_once_with({'state': -1})

    def test_train(self):
        my_mock = Metric('test')
        my_mock.train = Mock(return_value=None)
        metric = MetricList([my_mock])
        metric.train()
        my_mock.train.assert_called_once()

    def test_eval(self):
        my_mock = Metric('test')
        my_mock.eval = Mock(return_value=None)
        metric = MetricList([my_mock])
        metric.eval()
        my_mock.eval.assert_called_once()

    def test_reset(self):
        my_mock = Metric('test')
        my_mock.reset = Mock(return_value=None)
        metric = MetricList([my_mock])
        metric.reset({'state': -1})
        my_mock.reset.assert_called_once_with({'state': -1})
