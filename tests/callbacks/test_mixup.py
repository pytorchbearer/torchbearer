from unittest import TestCase
import torch

import torchbearer
from torchbearer.callbacks import MixupInputs

class TestMixupInputs(TestCase):

    def test_input(self):

        mixup = MixupInputs()
        X = torch.Tensor([[1.,2.,3.],[6.,7.,8.]])
        Y_true = torch.Tensor([0.,1.])

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true
        }

        mixup.on_sample(state)
        self.assertTrue(torch.all(torch.eq(state[torchbearer.X], X * state[torchbearer.MIXUP_LAMBDA] + X[state[torchbearer.MIXUP_PERMUTATION], :] * (1-state[torchbearer.MIXUP_LAMBDA]))))


    def test_alpha(self):
        mixup = MixupInputs(-0.1)
        X = torch.Tensor([[1.,2.,3.],[6.,7.,8.]])
        Y_true = torch.Tensor([0.,1.])

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true
        }

        mixup.on_sample(state)
        self.assertTrue(state[torchbearer.MIXUP_LAMBDA] == 1.0)

    def test_fixed_lambda(self):
        mixup = MixupInputs(-0.1)
        X = torch.Tensor([[1.,2.,3.],[6.,7.,8.]])
        Y_true = torch.Tensor([0.,1.])
        lam = 0.3

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true,
            torchbearer.MIXUP_LAMBDA : lam
        }

        mixup.on_sample(state, lam)
        self.assertTrue(state[torchbearer.X][0][1] == 0.3 * 2 + 0.7 * 2 or state[torchbearer.X][0][1] == 0.3 * 2 + 0.7 * 7)

    def test_target(self):
        mixup = MixupInputs(-0.1)
        X = torch.Tensor([[1.,2.,3.],[6.,7.,8.]])
        Y_true = torch.Tensor([0.,1.])

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true
        }

        mixup.on_sample(state)
        print(state[torchbearer.Y_TRUE])
        self.assertTrue(state[torchbearer.Y_TRUE] == (torch.Tensor([0.,1.]), torch.Tensor([0.,1.])) or state[torchbearer.Y_TRUE] == (torch.Tensor([0.,1.]), torch.Tensor([1.,0.])))
