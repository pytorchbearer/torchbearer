import math
import unittest
from mock import patch, Mock

import torch

from torchbearer.variational import SimpleDistribution, SimpleNormal, SimpleUniform, SimpleExponential, SimpleWeibull

class TestEmptyMethods(unittest.TestCase):
    def test_methods(self):
        dist = SimpleDistribution()

        self.assertTrue(dist.support is None)
        self.assertTrue(dist.arg_constraints is None)
        self.assertTrue(dist.mean is None)
        self.assertTrue(dist.variance is None)

        self.assertTrue(dist.expand(torch.Size()) is None)
        self.assertTrue(dist.cdf(1) is None)
        self.assertTrue(dist.icdf(1) is None)
        self.assertTrue(dist.enumerate_support() is None)
        self.assertTrue(dist.entropy() is None)

        self.assertRaises(NotImplementedError, lambda: dist.rsample())
        self.assertRaises(NotImplementedError, lambda: dist.log_prob(1))


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

    def test_log_prob_tensor(self):
        mu = torch.ones(2, 2)
        logvar = (torch.zeros(2, 2) + 2).log()

        dist = SimpleNormal(mu, logvar)

        self.assertTrue(((dist.log_prob(torch.ones(2, 2)) + 1.2655).abs() < 0.0001).all())

    def test_log_prob_number(self):
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

    def test_log_prob_tensor(self):
        low = torch.ones(2, 2)
        high = torch.ones(2, 2) + 1

        dist = SimpleUniform(low, high)

        self.assertTrue((dist.log_prob(torch.ones(2, 2)).abs() < 0.0001).all())
        self.assertTrue((dist.log_prob(torch.ones(2, 2) + 2) == float('-inf')).all())

    def test_log_prob_number(self):
        dist = SimpleUniform(1, 2)

        self.assertTrue((dist.log_prob(1).abs() < 0.0001).all())
        self.assertTrue((dist.log_prob(3) == float('-inf')).all())


class TestSimpleExponential(unittest.TestCase):
    @patch('torchbearer.variational.distributions.broadcast_all')
    def test_rsample_tensor(self, broadcast_all):
        rate = (torch.ones(2, 2) / 2).log()

        def new_mock_tensor(shape):
            x = torch.ones(shape)
            x.exponential_ = Mock(return_value=x)
            return x

        rate.new = Mock(side_effect=new_mock_tensor)
        broadcast_all.return_value = (rate,)
        dist = SimpleExponential(rate)

        self.assertTrue(((dist.rsample(sample_shape=torch.Size([2])) - 2.0).abs() < 0.0001).all())

    @patch('torchbearer.variational.distributions.broadcast_all')
    def test_rsample_number(self, broadcast_all):
        rate = (torch.ones(1) / 2).log()

        def new_mock_tensor(shape):
            x = torch.ones(shape)
            x.exponential_ = Mock(return_value=x)
            return x

        rate.new = Mock(side_effect=new_mock_tensor)
        broadcast_all.return_value = (rate,)
        dist = SimpleExponential(0.5)

        self.assertTrue(((dist.rsample(sample_shape=torch.Size([2])) - 2.0).abs() < 0.0001).all())

    def test_log_prob_tensor(self):
        dist = SimpleExponential((torch.ones(2, 2) / 2).log())

        self.assertTrue(((dist.log_prob(torch.ones(2, 2)) + 1.1931).abs() < 0.0001).all())

    def test_log_prob_number(self):
        dist = SimpleExponential(math.log(0.5))

        self.assertTrue(((dist.log_prob(torch.ones(2, 2)) + 1.1931).abs() < 0.0001).all())


class TestSimpleWeibull(unittest.TestCase):
    @patch('torchbearer.variational.distributions.torch.rand')
    def test_rsample_tensor(self, rand):
        l = torch.ones(2, 2)
        k = torch.ones(2, 2)

        dist = SimpleWeibull(l, k)

        rand.side_effect = lambda shape, dtype, device: torch.ones(shape) / 2
        self.assertTrue(((dist.rsample(sample_shape=torch.Size([2])) - 0.6931).abs() < 0.0001).all())

    @patch('torchbearer.variational.distributions.torch.rand')
    def test_rsample_number(self, rand):
        dist = SimpleWeibull(1, 1)

        rand.side_effect = lambda shape, dtype, device: torch.ones(shape) / 2
        self.assertTrue(((dist.rsample(sample_shape=torch.Size([2])) - 0.6931).abs() < 0.0001).all())

    def test_log_prob_tensor(self):
        l = torch.ones(2, 2)
        k = torch.ones(2, 2)

        dist = SimpleWeibull(l, k)
        self.assertTrue((dist.log_prob(torch.ones(2, 2)) < 0.0001).all())
        self.assertTrue((dist.log_prob(torch.ones(2, 2) - 2) == float('-inf')).all())

    def test_log_prob_number(self):
        dist = SimpleWeibull(1, 1)

        self.assertTrue((dist.log_prob(1) < 0.0001).all())
        self.assertTrue((dist.log_prob(-1) == float('-inf')).all())

