import unittest

import torch
from torch.autograd import Variable

import torchbearer
from torchbearer.metrics import Loss, Epoch, CategoricalAccuracy, TopKCategoricalAccuracy, BinaryAccuracy, MeanSquaredError


class TestLoss(unittest.TestCase):
    def setUp(self):
        with torch.no_grad():
            self._state = {
                torchbearer.LOSS: torch.FloatTensor([2.35])
            }
        self._metric = Loss().root  # Get root node of Tree for testing

    def test_train_process(self):
        self._metric.train()
        result = self._metric.process(self._state)
        self.assertAlmostEqual(2.35, result[0], 3, 0.002)

    def test_validate_process(self):
        self._metric.eval()
        result = self._metric.process(self._state)
        self.assertAlmostEqual(2.35, result[0], 3, 0.002)


class TestEpoch(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.EPOCH: 101
        }
        self._metric = Epoch().metric  # Get wrapped metric for testing

    def test_process(self):
        result = self._metric.process(self._state)
        self.assertEqual(101, result)

    def test_process_final(self):
        result = self._metric.process_final(self._state)
        self.assertEqual(101, result)


class TestBinaryAccuracy(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.Y_TRUE: torch.LongTensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0]
            ]),
            torchbearer.Y_PRED: torch.FloatTensor([
                [0.9, 0.1, 0.1],  # Correct
                [0.1, 0.9, 0.1],  # Correct
                [0.1, 0.1, 0.9],  # Correct
                [0.9, 0.1, 0.1],  # Incorrect
                [0.9, 0.1, 0.1]  # Incorrect
            ])
        }
        self._targets = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1]
        self._metric = BinaryAccuracy().root  # Get root node of Tree for testing

    def test_train_process(self):
        self._metric.train()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_validate_process(self):
        self._metric.eval()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_weird_types(self):
        state = {
            torchbearer.Y_TRUE: torch.LongTensor([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 0]
            ]).byte(),
            torchbearer.Y_PRED: torch.FloatTensor([
                [0.9, 0.1, 0.1],  # Correct
                [0.1, 0.9, 0.1],  # Correct
                [0.1, 0.1, 0.9],  # Correct
                [0.9, 0.1, 0.1],  # Incorrect
                [0.9, 0.1, 0.1]  # Incorrect
            ]).double()
        }

        self._metric.train()
        result = self._metric.process(state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_threshold(self):
        state = {
            torchbearer.Y_TRUE: torch.FloatTensor([
                [0.9, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0.3, 1],
                [0, 0.6, 0]
            ]),
            torchbearer.Y_PRED: torch.FloatTensor([
                [0.9, 0.1, 0.1],  # Correct
                [0.1, 0.9, 0.1],  # Correct
                [0.1, 0.1, 0.9],  # Correct
                [0.9, 0.1, 0.1],  # Incorrect
                [0.9, 0.1, 0.1]  # Incorrect
            ])
        }

        metric = BinaryAccuracy(threshold=0.4).root  # Get root node of Tree for testing
        metric.train()
        result = metric.process(state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))


class TestCategoricalAccuracy(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.Y_TRUE: Variable(torch.LongTensor([0, 1, 2, 2, 1])),
            torchbearer.Y_PRED: Variable(torch.FloatTensor([
                [0.9, 0.1, 0.1], # Correct
                [0.1, 0.9, 0.1], # Correct
                [0.1, 0.1, 0.9], # Correct
                [0.9, 0.1, 0.1], # Incorrect
                [0.9, 0.1, 0.1], # Incorrect
            ])),
        }
        self._targets = [1, 1, 1, 0, 0]
        self._metric = CategoricalAccuracy().root  # Get root node of Tree for testing

    def test_ignore_index(self):
        metric = CategoricalAccuracy(ignore_index=1).root  # Get root node of Tree for testing
        targets = [1, 1, 0]

        metric.train()
        result = metric.process(self._state)
        for i in range(0, len(targets)):
            self.assertEqual(result[i], targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(targets[i])
                                 + ' in: ' + str(result))

    def test_train_process(self):
        self._metric.train()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_train_process_soft(self):
        self._metric.train()
        soft_targets = torch.FloatTensor([
                [0.98, 0.01, 0.01], # Correct
                [0.01, 0.98, 0.01], # Correct
                [0.01, 0.01, 0.98], # Correct
                [0.01, 0.01, 0.98], # Incorrect
                [0.01, 0.98, 0.01], # Incorrect
            ])
        state = self._state.copy()
        state[torchbearer.Y_TRUE] = soft_targets
        result = self._metric.process(state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_validate_process(self):
        self._metric.eval()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))


class TestTopKCategoricalAccuracy(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.Y_TRUE: Variable(torch.LongTensor([0, 5, 2, 3, 1])),
            torchbearer.Y_PRED: Variable(torch.FloatTensor([
                [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],  # Correct
                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # Correct
                [0.6, 0.5, 0.4, 0.7, 0.8, 0.9],  # Incorrect
                [0.6, 0.5, 0.7, 0.4, 0.8, 0.9],  # Incorrect
                [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]  # Correct
            ]))
        }
        self._targets = [1, 1, 0, 0, 1]
        self._metric = TopKCategoricalAccuracy(k=5).root  # Get root node of Tree for testing

    def test_ignore_index(self):
        metric = TopKCategoricalAccuracy(ignore_index=1).root  # Get root node of Tree for testing
        targets = [1, 1, 0, 0]

        metric.train()
        result = metric.process(self._state)
        for i in range(0, len(targets)):
            self.assertEqual(result[i], targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(targets[i])
                                 + ' in: ' + str(result))

    def test_train_process(self):
        self._metric.train()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_validate_process(self):
        self._metric.eval()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_top_ten_default(self):
        metric = torchbearer.metrics.get_default('top_10_acc').root
        self.assertEqual(metric.k, 10)


class TestMeanSquaredError(unittest.TestCase):
    def setUp(self):
        self._state = {
            torchbearer.Y_TRUE: Variable(torch.FloatTensor(
                [0.8, 0.2, 0.0, 0.4, 0.3, 0.7]
            )),
            torchbearer.Y_PRED: Variable(torch.FloatTensor(
                [0.9, 0.1, 0.1, 0.7, 0.5, 0.6]
            ))
        }
        self._targets = [0.01, 0.01, 0.01, 0.09, 0.04, 0.01]
        self._metric = MeanSquaredError().root  # Get root node of Tree for testing

    def test_train_process(self):
        self._metric.train()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertAlmostEqual(result[i].item(), self._targets[i], places=3,
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_validate_process(self):
        self._metric.eval()
        result = self._metric.process(self._state)
        for i in range(0, len(self._targets)):
            self.assertAlmostEqual(result[i].item(), self._targets[i], places=3,
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))