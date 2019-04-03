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
        from torchbearer.callbacks.init import ZeroBias

        np.random.seed(7)
        torch.manual_seed(7)

        class Flatten(torch.nn.Module):
            def forward(self, x):
                return x.view(x.shape[0], -1)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(1,1,1),
            Flatten(),
            torch.nn.Linear(4, 2),
        )

        state = {torchbearer.MODEL: model}
        data = torch.rand(2, 1, 2, 2)
        ZeroBias(model.modules()).on_init(state)  # LSUV expects biases to be zero
        init.LsuvInit(data).on_init(state)

        correct_conv_weight = torch.FloatTensor([[[[3.2236]]]])
        correct_linear_weight = torch.FloatTensor([[-0.3414, -0.5503, -0.4402, -0.4367],
                                                   [0.3425, -0.0697, -0.6646, 0.4900]])

        conv_weight = list(model.modules())[1].weight
        linear_weight = list(model.modules())[3].weight
        diff_conv = (conv_weight-correct_conv_weight) < 0.0001
        diff_linear = (linear_weight - correct_linear_weight) < 0.0001
        self.assertTrue(diff_conv.all().item())
        self.assertTrue(diff_linear.all().item())

    def test_break(self):
        import numpy as np
        from torchbearer.callbacks.init import ZeroBias

        np.random.seed(7)
        torch.manual_seed(7)

        model = torch.nn.Sequential(
            torch.nn.Conv2d(1,1,1),
        )

        with patch('torchbearer.callbacks.lsuv.LSUV.apply_weights_correction') as awc:
            state = {torchbearer.MODEL: model}
            data = torch.rand(2, 1, 2, 2)
            ZeroBias(model.modules()).on_init(state)  # LSUV expects biases to be zero
            init.LsuvInit(data, std_tol=1e-20, max_attempts=0, do_orthonorm=False).on_init(state)

            # torchbearer.callbacks.lsuv.apply_weights_correction = old_fun
            self.assertTrue(awc.call_count == 2)

