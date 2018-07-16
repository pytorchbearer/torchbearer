import unittest

from unittest.mock import Mock, patch

import sconce
from sconce.metrics import RocAucScore

import torch

import numpy as np


class TestRocAucScore(unittest.TestCase):
    @patch('sklearn.metrics')
    def test_one_hot(self, mock_sklearn_metrics):
        mock_sklearn_metrics.roc_auc_score = Mock()
        metric = RocAucScore(one_hot_classes=3, one_hot_offset=1)
        metric.reset({sconce.DEVICE: 'cpu', sconce.DATA_TYPE: torch.float32})
        metric.process({sconce.BATCH: 0, sconce.DEVICE: 'cpu', sconce.DATA_TYPE: torch.float32,
                        sconce.Y_TRUE: torch.LongTensor([1, 2, 3]),
                        sconce.Y_PRED: torch.FloatTensor([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]])})
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
        metric = RocAucScore(one_hot_labels=False)
        metric.reset({sconce.DEVICE: 'cpu', sconce.DATA_TYPE: torch.float32})
        metric.process({sconce.BATCH: 0, sconce.DEVICE: 'cpu', sconce.DATA_TYPE: torch.float32,
                        sconce.Y_TRUE: torch.LongTensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                        sconce.Y_PRED: torch.FloatTensor([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]])})
        mock_sklearn_metrics.roc_auc_score.assert_called_once()
        self.assertTrue(np.array_equal(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]),
                                       mock_sklearn_metrics.roc_auc_score.call_args_list[0][0][0]))
        try:
            np.testing.assert_array_almost_equal(np.array([[0.0, 0.0, 0.0], [1.1, 1.1, 1.1], [2.2, 2.2, 2.2]]),
                                                                                   mock_sklearn_metrics.roc_auc_score.call_args_list[0][0][1])
        except AssertionError:
            self.fail('y_pred not correctly passed to sklearn')
