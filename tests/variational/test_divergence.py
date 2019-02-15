import unittest
from mock import Mock

import torch

import torchbearer
from torchbearer.variational import DivergenceBase, SimpleNormalUnitNormalKL, SimpleNormalSimpleNormalKL, SimpleNormal, SimpleWeibull, SimpleWeibullSimpleWeibullKL, SimpleExponentialSimpleExponentialKL, SimpleExponential

key = torchbearer.state_key('divergence_test')


class TestDivergenceBase(unittest.TestCase):
    def test_on_criterion(self):
        divergence = DivergenceBase({'test': key}).with_sum_sum_reduction()
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 4)
        self.assertTrue(state[torchbearer.LOSS].requires_grad)
        divergence.compute.assert_called_once_with(test=1)

    def test_on_criterion_validation(self):
        divergence = DivergenceBase({'test': key}).with_sum_sum_reduction()
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 4)
        divergence.compute.assert_called_once_with(test=1)

    def test_empty_compute(self):
        divergence = DivergenceBase({'test': key})
        self.assertRaises(NotImplementedError, divergence.compute)

    def test_custom_post_fcn(self):
        divergence = DivergenceBase({'test': key}).with_sum_sum_reduction().with_post_function(lambda loss: loss * 2)
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 8)
        divergence.compute.assert_called_once_with(test=1)

    def test_store(self):
        divergence = DivergenceBase({'test': key}, state_key=key).with_sum_sum_reduction().with_post_function(lambda loss: loss * 2)
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion_validation(state)
        self.assertTrue(state[key].item() == 4)
        self.assertTrue(state[torchbearer.LOSS].item() == 8)
        divergence.compute.assert_called_once_with(test=1)

    def test_reductions(self):
        divergence = DivergenceBase({'test': key}).with_sum_sum_reduction()
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 4)

        divergence = DivergenceBase({'test': key}).with_sum_mean_reduction()
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 2)

    def test_with_beta(self):
        divergence = DivergenceBase({'test': key}).with_sum_sum_reduction().with_beta(4)
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 16)
        divergence.compute.assert_called_once_with(test=1)

    def test_with_linear_capacity(self):
        divergence = DivergenceBase({'test': key}).with_sum_sum_reduction().with_linear_capacity(min_c=0, max_c=6, steps=6, gamma=2)
        divergence.compute = Mock(return_value=torch.ones((2, 2), requires_grad=True))

        state = {
            torchbearer.LOSS: torch.zeros(1, requires_grad=True),
            key: 1
        }

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 8)
        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_step_training(state)

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 6)
        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_step_training(state)

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 4)
        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_step_training(state)

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 2)
        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_step_training(state)

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 0)
        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_step_training(state)

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 2)
        state[torchbearer.LOSS] = torch.zeros(1)
        divergence.on_step_training(state)

        divergence.on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS].item() == 4)


class TestSimpleNormalUnitNormalKL(unittest.TestCase):
    def test_divergence(self):
        callback = SimpleNormalUnitNormalKL(input_key=key)
        dist = SimpleNormal(torch.zeros(2, 2), torch.ones(2, 2) * -1.3863)
        self.assertTrue(((callback.compute(dist) - 0.3181).abs() < 0.0001).all())


class TestSimpleNormalSimpleNormalKL(unittest.TestCase):
    def test_divergence(self):
        callback = SimpleNormalSimpleNormalKL(key, key)
        input = SimpleNormal(torch.zeros(2, 2), torch.ones(2, 2) * -1.3863)
        target = SimpleNormal(torch.ones(2, 2), torch.ones(2, 2) * 1.3863)
        self.assertTrue(((callback.compute(input, target) - 1.0425).abs() < 0.0001).all())

class TestSimpleWeibullSimpleWeibullKL(unittest.TestCase):
    def test_divergence(self):
        callback = SimpleWeibullSimpleWeibullKL(key, key)
        input = SimpleWeibull(torch.ones(2, 2), torch.zeros(2, 2) + 0.5)
        target = SimpleWeibull(torch.ones(2, 2), torch.ones(2, 2) * 5)
        self.assertTrue(((callback.compute(input, target) - 3628803.7500).abs() < 0.0001).all())


class TestSimpleExponentialSimpleExponentialKL(unittest.TestCase):
    def test_divergence(self):
        callback = SimpleExponentialSimpleExponentialKL(key, key)
        input = SimpleExponential(torch.ones(2, 2) - 1.6931)
        target = SimpleExponential(torch.zeros(2, 2))
        self.assertTrue(((callback.compute(input, target) - 0.3068).abs() < 0.0001).all())