from unittest import TestCase
from mock import patch, call
import torch

import torchbearer
from torchbearer.callbacks import SamplePairing


class TestSamplePairing(TestCase):
    @patch('torchbearer.callbacks.sample_pairing.torch')
    def test_on_sample(self, mock_torch):
        mock_torch.randperm.return_value = torch.Tensor([3, 2, 1, 0]).long()
        callback = SamplePairing(policy=SamplePairing.default_policy(0, 10, 10, 10))

        state = {torchbearer.INPUT: torch.Tensor([0.25, 0.5, 0.75, 1]), torchbearer.EPOCH: 0}
        callback.on_sample(state)
        self.assertTrue(((state[torchbearer.INPUT] - 0.625).abs() < 0.0001).all())

    def test_default_policy(self):
        policy = SamplePairing.default_policy(10, 80, 3, 2)

        state = {torchbearer.EPOCH: 0}
        self.assertFalse(policy(state))
        state = {torchbearer.EPOCH: 10}
        self.assertTrue(policy(state))

        for i in range(2):
            state = {torchbearer.EPOCH: 11 + i}
            policy(state)
        state = {torchbearer.EPOCH: 14}
        self.assertFalse(policy(state))
        state = {torchbearer.EPOCH: 15}
        self.assertFalse(policy(state))
        state = {torchbearer.EPOCH: 16}
        self.assertTrue(policy(state))
