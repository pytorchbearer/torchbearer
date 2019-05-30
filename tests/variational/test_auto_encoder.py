import torch
import unittest
import torchbearer
from mock import Mock

import torchbearer.variational.auto_encoder as ae


class SimpleAE(ae.AutoEncoderBase):
    def encode(self, x, state=None):
        return x, state

    def decode(self, sample, state=None):
        return sample, state


class TestAutoEncoder(unittest.TestCase):
    def test_empty_methods(self):
        base = ae.AutoEncoderBase(10)

        self.assertRaises(NotImplementedError, lambda: base.encode(0))
        self.assertRaises(NotImplementedError, lambda: base.decode(0))

    def test_forward_call_counts(self):
        latents = 10
        x = torch.rand(1)
        state = {torchbearer.X: x, torchbearer.Y_TRUE: 1}

        model = SimpleAE(latents)
        model.encode = Mock()
        model.decode = Mock()

        model.forward(x, state)
        self.assertTrue(model.encode.call_count == 1)
        self.assertTrue(model.decode.call_count == 1)

        model.forward(x, state)
        self.assertTrue(model.encode.call_count == 2)
        self.assertTrue(model.decode.call_count == 2)

    def test_forward_call_args(self):
        latents = 10
        x = torch.rand(1)
        state = {torchbearer.X: x, torchbearer.Y_TRUE: 1}

        model = SimpleAE(latents)
        model.encode = Mock()
        model.encode.return_value = None
        model.decode = Mock()

        model.forward(x, state)
        self.assertTrue(model.encode.call_args[0] == (x, state))
        self.assertTrue(model.decode.call_args[0] == (None, state))

    def test_forward_replace_y_true(self):
        latents = 10
        x = torch.rand(1)
        state = {torchbearer.X: x, torchbearer.Y_TRUE: 1}

        model = SimpleAE(latents)
        model.encode = Mock()
        model.encode.return_value = None
        model.decode = Mock()

        model.forward(x, state)
        self.assertTrue(model.decode.call_args[0][1][torchbearer.Y_TRUE] == x)

    def test_latent_dims(self):
        latents = 10
        x = torch.rand(1)
        state = {torchbearer.X: x, torchbearer.Y_TRUE: 1}

        model = SimpleAE(latents)
        self.assertTrue(model.latent_dims == latents)