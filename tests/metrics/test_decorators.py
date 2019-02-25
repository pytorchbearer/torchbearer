import unittest

from mock import Mock

import torchbearer.metrics as metrics
from torchbearer.metrics import default_for_key, lambda_metric, EpochLambda, BatchLambda


class TestDecorators(unittest.TestCase):
    def test_default_for_key_class(self):
        metric = metrics.Loss
        metric = default_for_key('test')(metric)
        self.assertTrue(metrics.get_default('test').name == 'loss')
        self.assertTrue(metric == metrics.Loss)

    def test_default_for_key_metric(self):
        metric = metrics.Loss()
        metric = default_for_key('test')(metric)
        self.assertTrue(metrics.get_default('test').name == 'loss')
        self.assertTrue(metric.name == 'loss')

    def test_default_for_key_args(self):
        mock = Mock()

        class MyMetric(metrics.Metric):
            def __init__(self, *args, **kwargs):
                super(MyMetric, self).__init__('test')
                mock(*args, **kwargs)

        default_for_key('test', 10, some_arg='a test')(MyMetric)
        metrics.get_default('test')
        mock.assert_called_once_with(10, some_arg='a test')

    def test_lambda_metric_epoch(self):
        metric = 'test'
        metric = lambda_metric('test', on_epoch=True)(metric)
        self.assertTrue(isinstance(metric, EpochLambda))
        self.assertTrue(metric._final == 'test')

    def test_lambda_metric_batch(self):
        metric = 'test'
        metric = lambda_metric('test')(metric)
        self.assertTrue(isinstance(metric, BatchLambda))
        self.assertTrue(metric._metric_function == 'test')

    def test_to_dict_metric_class(self):
        metric = metrics.Metric
        out = metrics.to_dict(metric)('test')
        self.assertTrue(out.metric.name == 'test')
        self.assertTrue(isinstance(out, metrics.ToDict))

    def test_to_dict_metric_instance(self):
        metric = metrics.Metric('test')
        out = metrics.to_dict(metric)
        self.assertTrue(out.metric.name == 'test')
        self.assertTrue(isinstance(out, metrics.ToDict))

    def test_mean_metric_class(self):
        metric = metrics.Metric
        out = metrics.mean(dim=10)(metric)('test')
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Mean))
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(out.children[0].metric.name == 'test')
        self.assertTrue(out.root.name == 'test')

    def test_mean_metric_instance(self):
        metric = metrics.Metric('test')
        out = metrics.mean(metric, dim=10)
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Mean))
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(out.children[0].metric.name == 'test')
        self.assertTrue(out.root.name == 'test')

    def test_std_metric_class(self):
        metric = metrics.Metric
        out = metrics.std(dim=10, unbiased=False)(metric)('test')
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Std))
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(not out.children[0].metric._unbiased)
        self.assertTrue(out.children[0].metric.name == 'test_std')
        self.assertTrue(out.root.name == 'test')

    def test_std_metric_instance(self):
        metric = metrics.Metric('test')
        out = metrics.std(metric, dim=10, unbiased=False)
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Std))
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(not out.children[0].metric._unbiased)
        self.assertTrue(out.children[0].metric.name == 'test_std')
        self.assertTrue(out.root.name == 'test')

    def test_var_metric_class(self):
        metric = metrics.Metric
        out = metrics.var(dim=10, unbiased=False)(metric)('test')
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Var))
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(not out.children[0].metric._unbiased)
        self.assertTrue(out.children[0].metric.name == 'test_var')
        self.assertTrue(out.root.name == 'test')

    def test_var_metric_instance(self):
        metric = metrics.Metric('test')
        out = metrics.var(metric, dim=10, unbiased=False)
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.Var))
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(not out.children[0].metric._unbiased)
        self.assertTrue(out.children[0].metric.name == 'test_var')
        self.assertTrue(out.root.name == 'test')

    def test_running_mean_metric_class(self):
        metric = metrics.Metric
        out = metrics.running_mean(batch_size=40, step_size=20, dim=10)(metric)('test')
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.RunningMean))
        self.assertTrue(out.children[0].metric._batch_size == 40)
        self.assertTrue(out.children[0].metric._step_size == 20)
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(out.children[0].metric.name == 'running_test')
        self.assertTrue(out.root.name == 'test')

    def test_running_mean_metric_instance(self):
        metric = metrics.Metric('test')
        out = metrics.running_mean(batch_size=40, step_size=20, dim=10)(metric)
        self.assertTrue(isinstance(out, metrics.MetricTree))
        self.assertTrue(isinstance(out.children[0], metrics.ToDict))
        self.assertTrue(isinstance(out.children[0].metric, metrics.RunningMean))
        self.assertTrue(out.children[0].metric._batch_size == 40)
        self.assertTrue(out.children[0].metric._step_size == 20)
        self.assertTrue(out.children[0].metric._kwargs['dim'] == 10)
        self.assertTrue(out.children[0].metric.name == 'running_test')
        self.assertTrue(out.root.name == 'test')
