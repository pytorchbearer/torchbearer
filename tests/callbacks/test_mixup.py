from unittest import TestCase
from mock import patch, call, MagicMock
import torch

import torchbearer
from torchbearer.callbacks import Mixup, MixupAcc


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
        mixup = Mixup(-0.1, 0.3)
        X = torch.Tensor([[1., 2., 3.], [6., 7., 8.]])
        Y_true = torch.Tensor([0., 1.])
        lam = 0.3

        state = {
            torchbearer.X : X,
            torchbearer.Y_TRUE : Y_true,
            torchbearer.MIXUP_LAMBDA : lam
        }

        mixup.on_sample(state)
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

        loss = Mixup.mixup_loss
        res = loss({torchbearer.DATA: torchbearer.TRAIN_DATA, torchbearer.Y_PRED: 'test1', torchbearer.Y_TRUE: ('target1', 'target2'), torchbearer.MIXUP_LAMBDA: 0.7})

        mock_cross_entropy.assert_has_calls([call('test1', 'target1'), call('test1', 'target2')])
        self.assertTrue(res == 1.0)

class TestMixupAcc(TestCase):
    def setUp(self):
        self.lam = 0.8

        self._state = {
            torchbearer.Y_TRUE: (torch.LongTensor([0, 1, 2, 2, 1]), torch.LongTensor([1, 2, 1, 0, 1])),
            torchbearer.Y_PRED: torch.FloatTensor([
                [0.9, 0.1, 0.1],  # Correct
                [0.1, 0.9, 0.1],  # Correct
                [0.1, 0.1, 0.9],  # Correct
                [0.9, 0.1, 0.1],  # Incorrect
                [0.9, 0.1, 0.1],  # Incorrect
            ]),
            torchbearer.MIXUP_LAMBDA: self.lam,
        }
        self._targets = [0.8, 0.8, 0.8, 0.2, 0]
        self._true_targets = [1, 1, 1, 0, 0]
        self._metric = MixupAcc().root  # Get root node of Tree for testing

    def test_train_process(self):
        self._metric.train()
        result = self._metric.process_train(self._state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_val_process(self):
        self._metric.train()
        _state = self._state.copy()
        _state[torchbearer.Y_TRUE] = _state[torchbearer.Y_TRUE][0]
        result = self._metric.process_validate(_state)
        for i in range(0, len(self._targets)):
            self.assertEqual(result[i], self._true_targets[i],
                             msg='returned: ' + str(result[i]) + ' expected: ' + str(self._targets[i])
                                 + ' in: ' + str(result))

    def test_reset(self):
        self._metric.cat_acc = MagicMock()
        self._metric.reset({})
        self.assertTrue(self._metric.cat_acc.reset.call_count == 1)