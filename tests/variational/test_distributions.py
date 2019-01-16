import math
import unittest
from unittest.mock import patch

import torch

from torchbearer.variational import SimpleNormal


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
        mu = torch.ones(2, 2)

        dist = SimpleNormal(mu, 0)

        normal.side_effect = lambda mu, std: mu
        self.assertTrue(dist.rsample(sample_shape=torch.Size([2])).sum() == 8)

        normal.side_effect = lambda mu, std: std
        self.assertTrue(dist.rsample(sample_shape=torch.Size([2])).sum() == 16)

    def test_logprob_tensor(self):
        mu = torch.ones(2, 2)
        logvar = (torch.zeros(2, 2) + 2).log()

        dist = SimpleNormal(mu, logvar)

        self.assertTrue(((dist.log_prob(torch.ones(2, 2)) + 1.2655).abs() < 0.0001).sum() == 4)

    def test_logprob_number(self):
        mu = torch.ones(2, 2)
        logvar = math.log(2)

        dist = SimpleNormal(mu, logvar)

        self.assertTrue(((dist.log_prob(torch.ones(2, 2)) + 1.2655).abs() < 0.0001).sum() == 4)