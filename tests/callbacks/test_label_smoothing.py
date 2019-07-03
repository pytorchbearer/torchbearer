from unittest import TestCase

import torch

import torchbearer
from torchbearer.callbacks import LabelSmoothingRegularisation


class TestLabelSmoothingRegularisation(TestCase):
    def test_with_long_target(self):
        callback = LabelSmoothingRegularisation(epsilon=0.4, classes=4)
        state = {torchbearer.TARGET: torch.Tensor([1, 3]).long()}

        target = torch.Tensor([
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7]
        ])

        callback.on_sample(state)
        self.assertTrue(((state[torchbearer.TARGET] - target).abs() < 0.00001).all())

    def test_with_hot_target(self):
        callback = LabelSmoothingRegularisation(epsilon=0.4)
        state = {torchbearer.TARGET: torch.Tensor([
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ]).long()}

        target = torch.Tensor([
            [0.1, 0.7, 0.1, 0.1],
            [0.1, 0.1, 0.1, 0.7]
        ])

        callback.on_sample(state)
        self.assertTrue(((state[torchbearer.TARGET] - target).abs() < 0.00001).all())
