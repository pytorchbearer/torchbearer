import torch
from unittest import TestCase

from mock import MagicMock, patch

import torchbearer
import torchbearer.callbacks.init as init


class TestWeightInit(TestCase):
    def test_modules_from_state(self):
        callback = init.WeightInit(targets=['Mock'])
        model = MagicMock()
        state = {torchbearer.MODEL: model}
        callback.on_init(state)
        self.assertTrue(model.modules.call_count == 1)

    def test_filter(self):
        mock = MagicMock()
        callback = init.WeightInit(initialiser=lambda m: m.test(), modules=[mock], targets=['Mock'])
        callback.on_init({})
        self.assertTrue(mock.test.call_count == 1)

        mock = MagicMock()
        callback = init.WeightInit(initialiser=lambda m: m.test(), modules=[mock], targets=['Not'])
        callback.on_init({})
        self.assertTrue(mock.test.call_count == 0)

    def test_module_list(self):
        mock = MagicMock()
        callback = init.WeightInit(initialiser=lambda m: m.test(), modules=[mock], targets=['Mock'])
        model = MagicMock()
        state = {torchbearer.MODEL: model}
        callback.on_init(state)
        self.assertTrue(model.modules.call_count == 0)


class TestSimpleInits(TestCase):
    @patch('torchbearer.callbacks.init.init')
    def test_kaiming(self, nn_init):
        callback = init.KaimingNormal(a=1, mode='test', nonlinearity='test2')
        mock = MagicMock()
        callback.initialiser(mock)
        nn_init.kaiming_normal_.assert_called_once_with(mock.weight.data, a=1, mode='test', nonlinearity='test2')

        callback = init.KaimingUniform(a=1, mode='test', nonlinearity='test2')
        mock = MagicMock()
        callback.initialiser(mock)
        nn_init.kaiming_uniform_.assert_called_once_with(mock.weight.data, a=1, mode='test', nonlinearity='test2')

    @patch('torchbearer.callbacks.init.init')
    def test_xavier(self, nn_init):
        callback = init.XavierNormal(gain=100)
        mock = MagicMock()
        callback.initialiser(mock)
        nn_init.xavier_normal_.assert_called_once_with(mock.weight.data, gain=100)

        callback = init.XavierUniform(gain=100)
        mock = MagicMock()
        callback.initialiser(mock)
        nn_init.xavier_uniform_.assert_called_once_with(mock.weight.data, gain=100)

    def test_bias(self):
        callback = init.ZeroBias()
        mock = MagicMock()
        callback.initialiser(mock)
        self.assertTrue(mock.bias.data.zero_.call_count == 1)


class TestLsuv(TestCase):
    def test_end_to_end(self):
        import numpy as np
        np.random.seed(7)
        torch.manual_seed(7)

        class Flatten(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1)

        from torch.utils.data import DataLoader
        model = torch.nn.Sequential(
            torch.nn.Conv2d(1,1,1),
            Flatten(),
            torch.nn.Linear(4, 2),
        )

        state = {torchbearer.MODEL: model}
        data = torch.rand(2, 1, 2, 2)

        init.LsuvInit(data).on_init(state)
        correct_conv_weight = torch.FloatTensor([[[[3.4462]]]])
        correct_linear_weight = torch.FloatTensor([[0.0817, -0.0061, -0.8176, 0.5700],
                                                   [-0.6918, 0.0483, -0.4575, -0.5566]])

        conv_weight = list(model.modules())[1].weight
        linear_weight = list(model.modules())[3].weight

        diff_conv = (conv_weight-correct_conv_weight) < 0.0001
        diff_linear = (linear_weight - correct_linear_weight) < 0.0001
        self.assertTrue(diff_conv.all().item)
        self.assertTrue(diff_linear.all().item)
