from unittest import TestCase

import torch
import torch.nn as nn
from mock import Mock, patch

import torchbearer
from torchbearer.callbacks import WeightDecay, L1WeightDecay, L2WeightDecay


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

    @patch('torchbearer.callbacks.weight_decay.WeightDecay.__init__')
    def test_L1(self, mock_wd):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=-1)
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = L1WeightDecay()
        self.assertTrue(mock_wd.call_args[1]['p'] == 1)

    @patch('torchbearer.callbacks.weight_decay.WeightDecay.__init__')
    def test_L2(self, mock_wd):
        model = nn.Sequential(nn.Conv2d(3, 3, 3))
        model.parameters = Mock(return_value=-1)
        state = {
            torchbearer.MODEL: model,
            torchbearer.LOSS: 1
        }

        decay = L2WeightDecay()
        self.assertTrue(mock_wd.call_args[1]['p'] == 2)
