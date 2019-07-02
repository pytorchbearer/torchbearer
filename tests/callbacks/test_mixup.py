from unittest import TestCase
from mock import patch, call
import torch

import torchbearer
from torchbearer.callbacks import Mixup


class TestMixupInputs(TestCase):
    def test_input(self):
        mixup = Mixup()
        X = torch.Tensor([[1., 2., 3.], [6., 7., 8.]])
        Y_true = torch.Tensor([0., 1.])

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true
        }

        mixup.on_sample(state)
        self.assertTrue((torch.eq(state[torchbearer.X], X * state[torchbearer.MIXUP_LAMBDA] + X[state[torchbearer.MIXUP_PERMUTATION], :] * (1-state[torchbearer.MIXUP_LAMBDA]))).all())

    def test_alpha(self):
        mixup = Mixup(-0.1)
        X = torch.Tensor([[1., 2., 3.], [6., 7., 8.]])
        Y_true = torch.Tensor([0., 1.])

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true
        }

        mixup.on_sample(state)
        self.assertTrue(state[torchbearer.MIXUP_LAMBDA] == 1.0)

    def test_fixed_lambda(self):
        mixup = Mixup(-0.1)
        X = torch.Tensor([[1., 2., 3.], [6., 7., 8.]])
        Y_true = torch.Tensor([0., 1.])
        lam = 0.3

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true,
            torchbearer.MIXUP_LAMBDA : lam
        }

        mixup.on_sample(state, lam)
        self.assertTrue(state[torchbearer.X][0][1] == 0.3 * 2 + 0.7 * 2 or state[torchbearer.X][0][1] == 0.3 * 2 + 0.7 * 7)

    def test_target(self):
        mixup = Mixup(-0.1)
        X = torch.Tensor([[1., 2., 3.], [6., 7., 8.]])
        Y_true = torch.Tensor([0., 1.])

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true
        }

        mixup.on_sample(state)

        self.assertTrue((state[torchbearer.Y_TRUE][0] == torch.Tensor([0., 1.])).all())
        self.assertTrue((state[torchbearer.Y_TRUE][1] == torch.Tensor([0., 1.])).all() or (state[torchbearer.Y_TRUE][1] == torch.Tensor([1., 0.])).all())

    @patch('torchbearer.callbacks.mixup.F.cross_entropy')
    def test_loss(self, mock_cross_entropy):
        mock_cross_entropy.return_value = 1.0

        loss = Mixup.loss
        res = loss({torchbearer.Y_PRED: 'test1', torchbearer.Y_TRUE: ('target1', 'target2'), torchbearer.MIXUP_LAMBDA: 0.7})

        mock_cross_entropy.assert_has_calls([call('test1', 'target1'), call('test1', 'target2')])
        self.assertTrue(res == 1.0)
