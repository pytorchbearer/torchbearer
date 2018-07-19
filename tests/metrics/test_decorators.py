import unittest
from unittest.mock import patch, Mock

import torchbearer.metrics as metrics
from torchbearer.metrics import default_for_key, lambda_metric, EpochLambda, BatchLambda


class TestDecorators(unittest.TestCase):
    def test_default_for_key_class(self):
        metric = metrics.Loss
        metric = default_for_key('test')(metric)
        self.assertTrue('test' in metrics.DEFAULT_METRICS)
        self.assertTrue(metrics.DEFAULT_METRICS['test'].name == 'loss')
        self.assertTrue(metric == metrics.Loss)

    def test_default_for_key_metric(self):
        metric = metrics.Loss()
        metric = default_for_key('test')(metric)
        self.assertTrue('test' in metrics.DEFAULT_METRICS)
        self.assertTrue(metrics.DEFAULT_METRICS['test'].name == 'loss')
        self.assertTrue(metric.name == 'loss')

    def test_lambda_metric_epoch(self):
        metric = 'test'
        metric = lambda_metric('test', on_epoch=True)(metric)().build()
        self.assertTrue(isinstance(metric, EpochLambda))
        self.assertTrue(metric._final == 'test')

    def test_lambda_metric_batch(self):
        metric = 'test'
        metric = lambda_metric('test')(metric)().build()
        self.assertTrue(isinstance(metric, BatchLambda))
        self.assertTrue(metric._metric_function == 'test')

    def test_to_dict_metric(self):
        metric = metrics.Metric
        out = metrics.to_dict(metric)('test').build()
        self.assertTrue(out.metric.name == 'test')
        self.assertTrue(isinstance(out, metrics.ToDict))

    @patch('torchbearer.metrics.MetricFactory.build')
    def test_to_dict_metric_factory(self, build_mock):
        metric = metrics.MetricFactory
        build_mock.return_value = metrics.Metric('test')
        out = metrics.to_dict(metric)().build()
        self.assertTrue(out.metric.name == 'test')
        self.assertTrue(isinstance(out, metrics.ToDict))
        build_mock.assert_called_once()

    def test_mean_metric(self):
        metric = metrics.Metric
        out = metrics.mean(metric)('test').build()
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Mean))
        self.assertTrue(out.children[0].metric.name == 'test')
        self.assertTrue(out.root.name == 'test')

    @patch('torchbearer.metrics.MetricFactory.build')
    def test_mean_metric_factory(self, build_mock):
        metric = metrics.MetricFactory
        build_mock.return_value = metrics.Metric('test')
        out = metrics.mean(metric)().build()
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Mean))
        self.assertTrue(out.children[0].metric.name == 'test')
        self.assertTrue(out.root.name == 'test')

    def test_std_metric(self):
        metric = metrics.Metric
        out = metrics.std(metric)('test').build()
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Std))
        self.assertTrue(out.children[0].metric.name == 'test_std')
        self.assertTrue(out.root.name == 'test')

    @patch('torchbearer.metrics.MetricFactory.build')
    def test_std_metric_factory(self, build_mock):
        metric = metrics.MetricFactory
        build_mock.return_value=metrics.Metric('test')
        out = metrics.std(metric)().build()
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Std))
        self.assertTrue(out.children[0].metric.name == 'test_std')
        self.assertTrue(out.root.name == 'test')

    def test_running_mean_metric(self):
        metric = metrics.Metric
        out = metrics.running_mean(batch_size=40, step_size=20)(metric)('test').build()
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.RunningMean))
        self.assertTrue(out.children[0].metric._batch_size == 40)
        self.assertTrue(out.children[0].metric._step_size == 20)
        self.assertTrue(out.children[0].metric.name == 'running_test')
        self.assertTrue(out.root.name == 'test')

    @patch('torchbearer.metrics.MetricFactory.build')
    def test_running_mean_metric_factory(self, build_mock):
        metric = metrics.MetricFactory
        build_mock.return_value=metrics.Metric('test')
        out = metrics.running_mean(metric)().build()
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.RunningMean))
        self.assertTrue(out.children[0].metric.name == 'running_test')
        self.assertTrue(out.root.name == 'test')
