from unittest import TestCase
from mock import patch, Mock

from torch import nn
from torchbearer.callbacks.manifold_mixup import ManifoldMixup
import torchbearer
import torch


class TestModule(nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
        self.conv = nn.Conv1d(1, 1, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(1)

    def forward(self, x):
        x = self.conv(x.view(-1, 1, 1))
        x = self.relu(x)
        x = self.bn(x)
        return x


class TestModule2(nn.Module):
    def __init__(self):
        super(TestModule2, self).__init__()
        self.layer1 = TestModule()

    def forward(self, x):
        return self.layer1(x)


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.fc1 = nn.Linear(1, 1)
        self.conv1 = nn.Conv1d(1, 1, 1)
        self.relu = nn.ReLU()
        self.layer1 = TestModule()
        self.layer2 = TestModule2()

    def forward(self, x):
        x = self.fc1(x)
        x = self.conv1(x.view(-1,1,1))
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class TestManifoldMixup(TestCase):
    def setUp(self):
        super(TestManifoldMixup, self).setUp()
        self.model = TestModel()

    def test_depth_none(self):
        mm = ManifoldMixup().at_depth(None)
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        self.assertTrue(len(mm._layers) == 12)
        
    def test_depth_0(self):
        mm = ManifoldMixup().at_depth(0)
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        checks = [
            self.model.fc1 in mm._layers,
            self.model.conv1 in mm._layers,
            self.model.relu in mm._layers,
            self.model.layer1 in mm._layers,
            self.model.layer2 in mm._layers,
            ]

        self.assertTrue(all(checks))  # Top level modules in
        self.assertFalse(self.model.layer1.conv in mm._layers)  # Depth 1 modules not in

    def test_depth_1(self):
        mm = ManifoldMixup().at_depth(1)
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        top_checks = [
            self.model.fc1 in mm._layers,
            self.model.conv1 in mm._layers,
            self.model.relu in mm._layers,
            self.model.layer1 in mm._layers,
            self.model.layer2 in mm._layers,
        ]

        first_checks = [
            self.model.layer1.conv in mm._layers,
            self.model.layer1.relu in mm._layers,
            self.model.layer1.bn in mm._layers,
            self.model.layer2.layer1 in mm._layers,
        ]

        self.assertFalse(any(top_checks))  # Top level modules not in
        self.assertTrue(all(first_checks))  # Depth 1 modules in

    def test_depth_2(self):
        mm = ManifoldMixup().at_depth(2)
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        top_checks = [
            self.model.fc1 in mm._layers,
            self.model.conv1 in mm._layers,
            self.model.relu in mm._layers,
            self.model.layer1 in mm._layers,
            self.model.layer2 in mm._layers,
        ]

        first_checks = [
            self.model.layer1.conv in mm._layers,
            self.model.layer1.relu in mm._layers,
            self.model.layer1.bn in mm._layers,
            self.model.layer2.layer1 in mm._layers,
        ]

        second_checks = [
            self.model.layer2.layer1.conv in mm._layers,
            self.model.layer2.layer1.relu in mm._layers,
            self.model.layer2.layer1.bn in mm._layers,
        ]

        self.assertFalse(any(top_checks))  # Top level modules not in
        self.assertFalse(any(first_checks))  # Depth 1 modules not in
        self.assertTrue(all(second_checks))  # Depth 2 modules in

    def test_for_layers(self):
        mm = ManifoldMixup().at_depth(None).for_layers(['conv1', 'layer1_conv', 'layer2_layer1_conv'])
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)
        
        self.assertTrue(self.model.conv1 in mm._layers, self.model.layer1.conv in mm._layers and self.model.layer2.layer1.conv in mm._layers)
        self.assertTrue(len(mm._layers) == 3)

    def test_get_selected_layers(self):
        mm = ManifoldMixup().at_depth(None).for_layers(['conv1', 'layer1_conv', 'layer2_layer1_conv'])
        found_layers = mm.get_selected_layers(self.model)
        self.assertTrue(len(found_layers) == 3)
        self.assertTrue('conv1' in found_layers)
        self.assertTrue('layer1_conv' in found_layers)
        self.assertTrue('layer2_layer1_conv' in found_layers)

    def test_layer_filter(self):
        mm = ManifoldMixup().at_depth(None).with_layer_filter(['conv1', 'layer1_conv', 'layer2_layer1_conv'])
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        self.assertFalse(self.model.conv1 in mm._layers,
                        self.model.layer1.conv in mm._layers and self.model.layer2.layer1.conv in mm._layers)
        self.assertTrue(len(mm._layers) == 12-3)

    def test_layer_type_filter(self):
        mm = ManifoldMixup().at_depth(None).with_layer_type_filter([nn.Conv1d])
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        self.assertFalse(self.model.conv1 in mm._layers,
                        self.model.layer1.conv in mm._layers and self.model.layer2.layer1.conv in mm._layers)
        self.assertTrue(len(mm._layers) == 12-3)

    def test_wrap(self):
        mm = ManifoldMixup().at_depth(None).for_layers(['conv1', 'layer1_relu', 'layer2_layer1_conv'])
        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        self.model.conv1.mixup()
        self.model.layer1.relu.mixup()
        self.model.layer2.layer1.conv.mixup()

        self.assertRaises(AttributeError, lambda: self.model.relu.mixup())

    @patch('torchbearer.callbacks.manifold_mixup._mixup_inputs', side_effect=lambda x, _: x)
    def test_call_mix(self, _):
        mm = ManifoldMixup().at_depth(None).for_layers(['conv1', 'layer1_relu', 'layer2_layer1_conv'])

        state = {torchbearer.MODEL: self.model}
        mm.on_start(state)

        self.model.conv1.mixup()
        self.assertTrue(self.model.conv1.do_mixup)
        self.model(torch.rand(3, 1))
        self.assertFalse(self.model.conv1.do_mixup)

    @patch('torchbearer.callbacks.manifold_mixup._mixup')
    def test_on_sample(self, mix):
        mm = ManifoldMixup().at_depth(None).for_layers(['conv1', 'layer1_relu', 'layer2_layer1_conv'])

        state = {torchbearer.MODEL: self.model, torchbearer.X: torch.rand(3, 1), torchbearer.Y_TRUE: torch.rand(3, 1)}
        mm.on_start(state)

        mm.on_sample(state)
        self.assertTrue(mix.call_count == 1)
        
        self.assertTrue(torchbearer.MIXUP_PERMUTATION in state)
        self.assertTrue(torchbearer.MIXUP_LAMBDA in state)

        state = {torchbearer.MODEL: self.model, torchbearer.X: torch.rand(3, 1), torchbearer.Y_TRUE: torch.rand(3, 1)}
        mm.on_sample(state)
        self.assertTrue(mix.call_count == 2)

    @patch('torchbearer.callbacks.manifold_mixup._mixup_inputs', side_effect=lambda x, _: x)
    def test_eval(self, mix):
        mm = ManifoldMixup().at_depth(None).for_layers(['conv1', 'layer1_relu', 'layer2_layer1_conv'])

        self.model.eval()
        state = {torchbearer.MODEL: self.model, torchbearer.X: torch.rand(3, 1), torchbearer.Y_TRUE: torch.rand(3, 1)}
        mm.on_start(state)

        mm.on_sample(state)
        self.model(torch.rand(3, 1))
        self.assertTrue(mix.call_count == 0)

        state = {torchbearer.MODEL: self.model, torchbearer.X: torch.rand(3, 1), torchbearer.Y_TRUE: torch.rand(3, 1)}
        mm.on_sample(state)
        self.model = self.model.train()
        self.model(torch.rand(3, 1))
        self.assertTrue(mix.call_count == 1)

    def test_mixup_inputs(self):
        from torchbearer.callbacks.manifold_mixup import _mixup_inputs

        x = torch.Tensor([[1, 2], [2, 3]])
        perm = torch.Tensor([1, 0]).long()
        lam = torch.Tensor([0.1])

        state = {torchbearer.X: x, torchbearer.MIXUP_PERMUTATION: perm, torchbearer.MIXUP_LAMBDA: lam}
        mixed = _mixup_inputs(x, state)

        self.assertFalse((mixed - torch.Tensor([[1.9, 2.9], [1.1, 2.1]]) > 1e-6).any())

    @patch('torchbearer.callbacks.manifold_mixup.Beta')
    def test_sample_lam_random(self, beta):
        mm = ManifoldMixup()
        sl = mm._sample_lam
        sl()

        self.assertTrue(beta.mock_calls[0][1] == (1., 1.))
        self.assertTrue(beta.mock_calls[1][0] == '().sample')

    def test_sample_lam_negative(self):
        mm = ManifoldMixup(alpha=-1)
        sl = mm._sample_lam
        lam = sl()

        self.assertTrue(lam == 1.)

    def test_sample_lam_fixed(self):
        mm = ManifoldMixup(lam=2.)
        sl = mm._sample_lam
        lam = sl()

        self.assertTrue(lam == 2.)
        
    def test_single_to_list(self):
        mm = ManifoldMixup()
        sl = mm._single_to_list

        item = 1.
        self.assertTrue(sl(item) == [item, ])




