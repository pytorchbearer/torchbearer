import math
import unittest
from unittest.mock import patch

import torch

from torchbearer.variational import SimpleNormal, SimpleUniform


class TestSimpleNormal(unittest.TestCase):
    @patch('torchbearer.variational.distributions.torch.normal')
    def test_rsample_tensor(self, normal):
        mu = torch.ones(2, 2)
        logvar = torch.zeros(2, 2)

        dist = SimpleNormal(mu, logvar)

        normal.side_effect = lambda mu, std: mu
        self.assertTrue(dist.rsample(sample_shape=torch.Size([2])).sum() == 8)

        normal.side_effect = lambda mu, std: std
        self.assertTrue(dist.rsample(sample_shape=torch.Size([2])).sum() == 16)

    @patch('torchbearer.variational.distributions.torch.normal')
    def test_rsample_number(self, normal):
        dist = SimpleNormal(1, 0)

        normal.side_effect = lambda mu, std: mu
        self.assertTrue(dist.rsample(sample_shape=torch.Size([2])).sum() == 2)

        normal.side_effect = lambda mu, std: std
        self.assertTrue(dist.rsample(sample_shape=torch.Size([2])).sum() == 4)

    def test_logprob_tensor(self):
        mu = torch.ones(2, 2)
        logvar = (torch.zeros(2, 2) + 2).log()

        dist = SimpleNormal(mu, logvar)

        self.assertTrue(((dist.log_prob(torch.ones(2, 2)) + 1.2655).abs() < 0.0001).all())

    def test_logprob_number(self):
        logvar = math.log(2)

        dist = SimpleNormal(1, logvar)

        self.assertTrue(((dist.log_prob(1) + 1.2655).abs() < 0.0001).all())


class TestSimpleUniform(unittest.TestCase):
    @patch('torchbearer.variational.distributions.torch.rand')
    def test_rsample_tensor(self, rand):
        low = torch.ones(2, 2)
        high = torch.ones(2, 2) + 1

        dist = SimpleUniform(low, high)

        rand.side_effect = lambda shape, dtype, device: torch.ones(shape) / 2
        self.assertTrue(((dist.rsample(sample_shape=torch.Size([2])) - 1.5).abs() < 0.0001).all())

    @patch('torchbearer.variational.distributions.torch.rand')
    def test_rsample_number(self, rand):
        dist = SimpleUniform(1, 2)

        rand.side_effect = lambda shape, dtype, device: torch.ones(shape) / 2
        self.assertTrue(((dist.rsample(sample_shape=torch.Size([2])) - 1.5).abs() < 0.0001).all())

    def test_logprob_tensor(self):
        low = torch.ones(2, 2)
        high = torch.ones(2, 2) + 1

        dist = SimpleUniform(low, high)

        self.assertTrue((dist.log_prob(torch.ones(2, 2)).abs() < 0.0001).all())
        self.assertTrue((dist.log_prob(torch.ones(2, 2) + 2) == float('-inf')).all())

    def test_logprob_number(self):
        dist = SimpleUniform(1, 2)

        self.assertTrue((dist.log_prob(1).abs() < 0.0001).all())
        self.assertTrue((dist.log_prob(3) == float('-inf')).all())
