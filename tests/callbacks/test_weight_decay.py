from unittest import TestCase
from unittest.mock import patch, Mock

import torch.nn as nn
import torch

import torchbearer
from torchbearer.callbacks import WeightDecay


class TestWeightDecay(TestCase):

    def test_single_parameter(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=[nn.Parameter(torch.Tensor([2.0]))])
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = WeightDecay(1, 1.0)
        decay.on_start(state)
        decay.on_criterion(state)

        self.assertTrue(state[torchbearer.LOSS].item() == 3)

    def test_multiple_parameters(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=[nn.Parameter(torch.Tensor([2.0, 1.0]))])
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = WeightDecay(1, 1.0)
        decay.on_start(state)
        decay.on_criterion(state)

        self.assertTrue(state[torchbearer.LOSS].item() == 4)

    def test_rate(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=[nn.Parameter(torch.Tensor([2.0]))])
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = WeightDecay(0.5, 1.0)
        decay.on_start(state)
        decay.on_criterion(state)

        self.assertTrue(state[torchbearer.LOSS].item() == 2)

    def test_pow(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=[nn.Parameter(torch.Tensor([2.0]))])
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = WeightDecay(2, 2)
        decay.on_start(state)
        decay.on_criterion(state)

        self.assertTrue(state[torchbearer.LOSS].item() == 5)

    def test_given_params(self):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=-1)
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = WeightDecay(2, 2, params=model.parameters())
        decay.on_start(state)

        self.assertTrue(decay.params == -1)
