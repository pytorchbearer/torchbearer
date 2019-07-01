import torch
from unittest import TestCase

from mock import MagicMock, patch

import torchbearer
from torchbearer.callbacks import UnpackState


class TestUnpackState(TestCase):
    def reset_state(self):
        return {
            torchbearer.X: 'data',
            torchbearer.Y_TRUE: 'targets'
        }

    def test_sample(self):
        packer = UnpackState(keys=[torchbearer.X, torchbearer.Y_TRUE])
        state = self.reset_state()

        packer.on_sample(state)
        self.assertTrue(state[torchbearer.X] == {torchbearer.X: 'data', torchbearer.Y_TRUE: 'targets'})

        state = self.reset_state()
        packer.on_sample_validation(state)
        self.assertTrue(state[torchbearer.X] == {torchbearer.X: 'data', torchbearer.Y_TRUE: 'targets'})

    def test_no_key(self):
        packer = UnpackState()
        state = self.reset_state()

        packer.on_sample(state)
        self.assertTrue(state[torchbearer.X] == 'data')

        state = self.reset_state()
        packer.on_sample_validation(state)
        self.assertTrue(state[torchbearer.X] == 'data')

    def test_fill_X(self):
        packer = UnpackState(keys=[torchbearer.Y_TRUE])
        state = self.reset_state()

        packer.on_sample(state)
        self.assertTrue(state[torchbearer.X] == {torchbearer.X: 'data', torchbearer.Y_TRUE: 'targets'})

        state = self.reset_state()
        packer.on_sample_validation(state)
        self.assertTrue(state[torchbearer.X] == {torchbearer.X: 'data', torchbearer.Y_TRUE: 'targets'})

    def test_forward_no_dict(self):
        packer = UnpackState(keys=[torchbearer.Y_TRUE])
        state = self.reset_state()

        state[torchbearer.Y_PRED] = 1
        packer.on_forward(state)
        self.assertTrue(state[torchbearer.Y_PRED] == 1)

        state = self.reset_state()

        state[torchbearer.Y_PRED] = 1
        packer.on_forward_validation(state)
        self.assertTrue(state[torchbearer.Y_PRED] == 1)

    def test_forward_list(self):
        packer = UnpackState(keys=[torchbearer.Y_TRUE])
        state = self.reset_state()

        state[torchbearer.Y_PRED] = [1, 2, 3]
        packer.on_forward(state)
        self.assertTrue(state[torchbearer.Y_PRED] == [1, 2, 3])

        state = self.reset_state()

        state[torchbearer.Y_PRED] = [1, 2, 3]
        packer.on_forward_validation(state)
        self.assertTrue(state[torchbearer.Y_PRED] == [1, 2, 3])

    def test_forward_dict_no_y_pred(self):
        packer = UnpackState(keys=[torchbearer.Y_TRUE])
        state = self.reset_state()

        state[torchbearer.Y_PRED] = {'one': 1, 'two': 2}
        packer.on_forward(state)
        self.assertTrue(state[torchbearer.Y_PRED] == {'one': 1, 'two': 2})

        state = self.reset_state()

        state[torchbearer.Y_PRED] = {'one': 1, 'two': 2}
        packer.on_forward_validation(state)
        self.assertTrue(state[torchbearer.Y_PRED] == {'one': 1, 'two': 2})

    def test_forward_dict_y_pred(self):
        packer = UnpackState(keys=[torchbearer.Y_TRUE])
        state = self.reset_state()

        state[torchbearer.Y_PRED] = {torchbearer.Y_PRED: 1, 'two': 2}
        packer.on_forward(state)
        self.assertTrue(state[torchbearer.Y_PRED] == 1)
        self.assertTrue(state['two'] == 2)

        state = self.reset_state()

        state[torchbearer.Y_PRED] = {torchbearer.Y_PRED: 1, 'two': 2}
        packer.on_forward_validation(state)
        self.assertTrue(state[torchbearer.Y_PRED] == 1)
        self.assertTrue(state['two'] == 2)