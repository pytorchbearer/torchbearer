import unittest
from mock import Mock, patch

import torch.nn as nn
import torch.nn.functional as F

import torchbearer
from torchbearer.metrics import DefaultAccuracy


class TestDefaultAccuracy(unittest.TestCase):
    def test_defaults(self):

        state = {torchbearer.CRITERION: 'not a criterion'}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'acc')

        state = {torchbearer.CRITERION: nn.CrossEntropyLoss()}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'acc')

        state = {torchbearer.CRITERION: nn.NLLLoss()}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'acc')

        state = {torchbearer.CRITERION: F.cross_entropy}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'acc')

        state = {torchbearer.CRITERION: F.nll_loss}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'acc')

        state = {torchbearer.CRITERION: nn.MSELoss()}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'mse')

        state = {torchbearer.CRITERION: F.mse_loss}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'mse')

        state = {torchbearer.CRITERION: nn.BCELoss()}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'binary_acc')

        state = {torchbearer.CRITERION: nn.BCEWithLogitsLoss()}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'binary_acc')

        state = {torchbearer.CRITERION: F.binary_cross_entropy}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'binary_acc')

        state = {torchbearer.CRITERION: F.binary_cross_entropy_with_logits}
        metric = DefaultAccuracy()
        metric.reset(state)
        self.assertEqual(metric.name, 'binary_acc')

    @patch('torchbearer.metrics.default.CategoricalAccuracy')
    def test_pass_through(self, cat_acc):
        mock = Mock()
        cat_acc.return_value = mock
        mock.reset = Mock()
        mock.process = Mock()
        mock.process_final = Mock()
        mock.eval = Mock()
        mock.train = Mock()

        metric = DefaultAccuracy()
        metric.reset({torchbearer.CRITERION: None})
        metric.process(1, 2, 3)
        metric.process_final(4, 5, 6)
        metric.eval()
        metric.train()

        self.assertEqual(cat_acc.call_count, 1)
        mock.reset.assert_called_once_with({torchbearer.CRITERION: None})
        mock.process.assert_called_once_with(1, 2, 3)
        mock.process_final.assert_called_once_with(4, 5, 6)
        self.assertEqual(mock.eval.call_count, 1)
        self.assertEqual(mock.train.call_count, 1)

    @patch('torchbearer.metrics.default.CategoricalAccuracy')
    def test_reset_after_eval(self, cat_acc):
        metric = DefaultAccuracy()
        self.assertTrue(cat_acc.call_count == 1)
        cat_acc.reset_mock()
        metric.eval()

        state = {torchbearer.CRITERION: F.cross_entropy, torchbearer.DATA: 'test'}
        mock = Mock()
        mock.eval = Mock()
        torchbearer.metrics.default.__loss_map__[F.cross_entropy.__name__] = lambda: mock

        metric.reset(state)

        mock.eval.assert_called_once_with(data_key='test')
        self.assertTrue(mock.eval.call_count == 1)
