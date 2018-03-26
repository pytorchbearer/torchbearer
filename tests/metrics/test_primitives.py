import unittest

from bink.metrics import Loss, Epoch, CategoricalAccuracy

import torch


class TestLoss(unittest.TestCase):
    def setUp(self):
        self._state = {
            'loss': torch.FloatTensor([2.35])
        }
        self._metric = Loss()

    def test_train_evaluate_dict(self):
        self._metric.train()
        result = self._metric.evaluate_dict(self._state)
        self.assertTrue('train_loss' in result, msg='train_loss is not a key in: ' + str(result))
        self.assertAlmostEqual(2.35, result['train_loss'][0], 3, 0.002)

    def test_validate_evaluate_dict(self):
        self._metric.eval()
        result = self._metric.evaluate_dict(self._state)
        self.assertTrue('val_loss' in result, msg='val_loss is not a key in: ' + str(result))
        self.assertAlmostEqual(2.35, result['val_loss'][0], 3, 0.002)


class TestEpoch(unittest.TestCase):
    def setUp(self):
        self._state = {
            'epoch': 101
        }
        self._metric = Epoch()

    def test_train_evaluate_dict(self):
        self._metric.train()
        result = self._metric.evaluate_dict(self._state)
        self.assertTrue('epoch' in result, msg='epoch is not a key in: ' + str(result))
        self.assertEqual(101, result['epoch'])

    def test_validate_evaluate_dict(self):
        self._metric.eval()
        result = self._metric.evaluate_dict(self._state)
        self.assertTrue('epoch' in result, msg='epoch is not a key in: ' + str(result))
        self.assertEqual(101, result['epoch'])


class TestCategoricalAccuracy(unittest.TestCase):
    def setUp(self):
        self._state = {
            'y_true':torch.LongTensor([0, 1, 2, 2, 1]),
            'y_pred':torch.FloatTensor([
                [0.9, 0.1, 0.1], # Correct
                [0.1, 0.9, 0.1], # Correct
                [0.1, 0.1, 0.9], # Correct
                [0.9, 0.1, 0.1], # Incorrect
                [0.9, 0.1, 0.1], # Incorrect
            ])
        }
        self._targets = [1, 1, 1, 0, 0]
        self._metric = CategoricalAccuracy()

    def test_train_evaluate_dict_key_exists(self):
        self._metric.train()
        result = self._metric.evaluate_dict(self._state)
        self.assertTrue('train_acc' in result, msg='train_acc is not a key in: ' + str(result))

    def test_validate_evaluate_dict_key_exists(self):
        self._metric.eval()
        result = self._metric.evaluate_dict(self._state)
        self.assertTrue('val_acc' in result, msg='val_acc is not a key in: ' + str(result))

    def test_train_evaluate_correct_output(self):
        self._metric.train()
        result = self._metric.evaluate(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_validate_evaluate_correct_output(self):
        self._metric.eval()
        result = self._metric.evaluate(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))
