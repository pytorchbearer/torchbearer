import unittest
from unittest.mock import Mock

from torchbearer.metrics import MetricFactory, MetricList, Metric, MetricTree, AdvancedMetric


class TestMetricFactory(unittest.TestCase):
    def test_empty_build(self):
        factory = MetricFactory()
        self.assertTrue(factory.build() is None)


class TestMetricTree(unittest.TestCase):
    def test_process(self):
        root = Metric('test')
        root.process = Mock(return_value='test')
        leaf1 = Metric('test')
        leaf1.process = Mock(return_value={'test': 10})
        leaf2 = Metric('test')
        leaf2.process = Mock(return_value=None)

        tree = MetricTree(root)
        tree.add_child(leaf1)
        tree.add_child(leaf2)

        self.assertTrue(tree.process('args') == {'test': 10})

        root.process.assert_called_once_with('args')
        leaf1.process.assert_called_once_with('test')
        leaf2.process.assert_called_once_with('test')

    def test_process_final(self):
        root = Metric('test')
        root.process_final = Mock(return_value='test')
        leaf1 = Metric('test')
        leaf1.process_final = Mock(return_value={'test': 10})
        leaf2 = Metric('test')
        leaf2.process_final = Mock(return_value=None)

        tree = MetricTree(root)
        tree.add_child(leaf1)
        tree.add_child(leaf2)

        self.assertTrue(tree.process_final('args') == {'test': 10})

        root.process_final.assert_called_once_with('args')
        leaf1.process_final.assert_called_once_with('test')
        leaf2.process_final.assert_called_once_with('test')

    def test_train(self):
        root = Metric('test')
        root.train = Mock()
        leaf = Metric('test')
        leaf.train = Mock()

        tree = MetricTree(root)
        tree.add_child(leaf)

        tree.train()
        root.train.assert_called_once()
        leaf.train.assert_called_once()

    def test_eval(self):
        root = Metric('test')
        root.eval = Mock()
        leaf = Metric('test')
        leaf.eval = Mock()

        tree = MetricTree(root)
        tree.add_child(leaf)

        tree.eval()
        root.eval.assert_called_once()
        leaf.eval.assert_called_once()

    def test_reset(self):
        root = Metric('test')
        root.reset = Mock()
        leaf = Metric('test')
        leaf.reset = Mock()

        tree = MetricTree(root)
        tree.add_child(leaf)

        tree.reset({})
        root.reset.assert_called_once_with({})
        leaf.reset.assert_called_once_with({})


class TestMetricList(unittest.TestCase):
    def test_list_in_list(self):
        metric = MetricList(['acc', MetricList(['loss'])])
        self.assertTrue(metric.metric_list[0].name == 'acc')
        self.assertTrue(metric.metric_list[1].name == 'loss')

    def test_default_acc(self):
        metric = MetricList(['acc'])
        self.assertTrue(metric.metric_list[0].name == 'acc', msg='acc not in: ' + str(metric.metric_list))

    def test_default_loss(self):
        metric = MetricList(['loss'])
        self.assertTrue(metric.metric_list[0].name == 'loss', msg='loss not in: ' + str(metric.metric_list))

    def test_default_epoch(self):
        metric = MetricList(['epoch'])
        self.assertTrue(metric.metric_list[0].name == 'epoch', msg='loss not in: ' + str(metric.metric_list))

    def test_process(self):
        my_mock = Metric('test')
        my_mock.process = Mock(return_value={'test': -1})
        metric = MetricList([my_mock])
        result = metric.process({'state': -1})
        self.assertEqual({'test': -1}, result)
        my_mock.process.assert_called_once_with({'state': -1})

    def test_process_final(self):
        my_mock = Metric('test')
        my_mock.process_final = Mock(return_value={'test': -1})
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


class TestAdvancedMetric(unittest.TestCase):
    def test_empty_methods(self):
        metric = AdvancedMetric('test')

        self.assertTrue(metric.process_train() is None)
        self.assertTrue(metric.process_final_train() is None)
        self.assertTrue(metric.process_validate() is None)
        self.assertTrue(metric.process_final_validate() is None)

    def test_train(self):
        metric = AdvancedMetric('test')
        metric.process_train = Mock()
        metric.process_final_train = Mock()

        metric.train()
        metric.process('testing')
        metric.process_train.assert_called_once_with('testing')

        metric.process_final('testing')
        metric.process_final_train.assert_called_once_with('testing')

    def test_eval(self):
        metric = AdvancedMetric('test')
        metric.process_validate = Mock()
        metric.process_final_validate = Mock()

        metric.eval()
        metric.process('testing')
        metric.process_validate.assert_called_once_with('testing')

        metric.process_final('testing')
        metric.process_final_validate.assert_called_once_with('testing')
