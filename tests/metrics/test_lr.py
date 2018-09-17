import unittest
from unittest.mock import Mock, patch

import torch.nn as nn
import torch.nn.functional as F

import torchbearer
from torchbearer.metrics import LR

import torch.optim as optim


class TestLR(unittest.TestCase):
    def test_simple(self):
        state = {torchbearer.OPTIMIZER: optim.SGD(nn.Linear(10, 10).parameters(), lr=0.01)}
        metric = LR()
        metric.reset(state)
        self.assertDictEqual(metric.process(state), {'lr': 0.01})
        self.assertDictEqual(metric.process_final(state), {'lr': 0.01})

    def test_groups(self):
        state = {
            torchbearer.OPTIMIZER: optim.SGD([
                {'params': nn.Linear(10, 10).parameters()},
                {'params': nn.Linear(10, 10).parameters(), 'lr': 1e-3}
            ], lr=1e-2, momentum=0.9)
        }
        metric = LR()
        metric.reset(state)
        self.assertListEqual(metric.process(state)['lr'], [1e-2, 1e-3])
        self.assertListEqual(metric.process_final(state)['lr'], [1e-2, 1e-3])
