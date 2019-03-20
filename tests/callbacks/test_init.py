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
