from unittest import TestCase

import torch

import torchbearer
from torchbearer.callbacks import BCPlus


class TestBCPlus(TestCase):
    def test_on_val(self):
        bcplus = BCPlus(classes=4)
        state = {torchbearer.TARGET: torch.tensor([1, 3, 2])}
        bcplus.on_sample_validation(state)
        self.assertTrue((state[torchbearer.TARGET] -
                         torch.tensor([[0, 1, 0, 0],
                                       [0, 0, 0, 1],
                                       [0, 0, 1, 0]]).float()).abs().lt(1e-4).all())

        bcplus = BCPlus(classes=4)
        state = {torchbearer.TARGET: torch.tensor([[0, 1, 0, 0],
                                                   [0, 0, 0, 1],
                                                   [0, 0, 1, 0]])}
        bcplus.on_sample_validation(state)
        self.assertTrue((state[torchbearer.TARGET] -
                         torch.tensor([[0, 1, 0, 0],
                                       [0, 0, 0, 1],
                                       [0, 0, 1, 0]]).float()).abs().lt(1e-4).all())

    def test_bc_loss(self):
        prediction = torch.tensor([[10.0, 0.01]])
        target = torch.tensor([[0., 0.8]])
        loss = BCPlus.bc_loss({torchbearer.PREDICTION: prediction, torchbearer.TARGET: target})
        self.assertTrue((loss - 7.81).abs().le(1e-2).all())

    def test_sample_targets(self):
        # Test mixup
        bcplus = BCPlus(classes=4, mixup_loss=True)
        state = {torchbearer.INPUT: torch.zeros(3, 10, 10), torchbearer.TARGET: torch.tensor([1, 3, 2]), torchbearer.DEVICE: 'cpu'}
        bcplus.on_sample(state)

        self.assertTrue(torchbearer.MIXUP_LAMBDA in state)
        self.assertTrue(torchbearer.MIXUP_PERMUTATION in state)
        self.assertTrue(len(state[torchbearer.TARGET]) == 2)

        # Test bcplus
        bcplus = BCPlus(classes=4)
        state = {torchbearer.INPUT: torch.zeros(3, 10, 10), torchbearer.TARGET: torch.tensor([1, 3, 2]),
                 torchbearer.DEVICE: 'cpu'}
        bcplus.on_sample(state)

        self.assertTrue(state[torchbearer.TARGET].dim() == 2)
        self.assertTrue(not (torchbearer.MIXUP_PERMUTATION in state))

    def test_sample_inputs(self):
        torch.manual_seed(7)

        batch = torch.tensor([[
            [0.1, 0.5, 0.6],
            [0.8, 0.6, 0.5],
            [0.2, 0.4, 0.7]
        ]])
        target = torch.tensor([1])
        state = {torchbearer.INPUT: batch, torchbearer.TARGET: target, torchbearer.DEVICE: 'cpu'}

        bcplus = BCPlus(classes=4)
        bcplus.on_sample(state)

        lam = torch.ones(1) * 0.2649

        self.assertTrue(((state[torchbearer.INPUT] * (lam.pow(2) + (1 - lam).pow(2)).sqrt()) - (batch - batch.mean())).abs().le(1e-4).all())
