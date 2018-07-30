import unittest

from unittest.mock import Mock, patch

import torchbearer
from torchbearer.metrics import RocAucScore, MetricList

import torch

import numpy as np


class TestRocAucScore(unittest.TestCase):
    @patch('sklearn.metrics')
    def test_one_hot(self, mock_sklearn_metrics):
        mock_sklearn_metrics.roc_auc_score = Mock()
        metric = RocAucScore(one_hot_classes=3, one_hot_offset=1).build()
        metric.reset({torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float32})
        res = metric.process({torchbearer.BATCH: 0, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float32,
                        torchbearer.Y_TRUE: torch.LongTensor([1, 2, 3]),
                        torchbearer.Y_PRED: torch.FloatTensor([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]])})
        self.assertTrue('roc_auc_score' in res)
        mock_sklearn_metrics.roc_auc_score.assert_called_once()
        self.assertTrue(np.array_equal(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                                       mock_sklearn_metrics.roc_auc_score.call_args_list[0][0][0]))
        try:
            np.testing.assert_array_almost_equal(np.array([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]]),
                                                                                   mock_sklearn_metrics.roc_auc_score.call_args_list[0][0][1])
        except AssertionError:
            self.fail('y_pred not correctly passed to sklearn')

    @patch('sklearn.metrics')
    def test_non_one_hot(self, mock_sklearn_metrics):
        mock_sklearn_metrics.roc_auc_score = Mock()
        metric = RocAucScore(one_hot_labels=False).build()
        metric.reset({torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float32})
        res = metric.process({torchbearer.BATCH: 0, torchbearer.DEVICE: 'cpu', torchbearer.DATA_TYPE: torch.float32,
                        torchbearer.Y_TRUE: torch.LongTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                        torchbearer.Y_PRED: torch.FloatTensor([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]])})
        self.assertTrue('roc_auc_score' in res)
        mock_sklearn_metrics.roc_auc_score.assert_called_once()
        self.assertTrue(np.array_equal(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                                       mock_sklearn_metrics.roc_auc_score.call_args_list[0][0][0]))
        try:
            np.testing.assert_array_almost_equal(np.array([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]]),
                                                                                   mock_sklearn_metrics.roc_auc_score.call_args_list[0][0][1])
        except AssertionError:
            self.fail('y_pred not correctly passed to sklearn')

    def test_default_roc(self):
        mlist = MetricList(['roc_auc'])
        self.assertTrue(mlist.metric_list[0].name == 'roc_auc_score')

        mlist = MetricList(['roc_auc_score'])
        self.assertTrue(mlist.metric_list[0].name == 'roc_auc_score')
