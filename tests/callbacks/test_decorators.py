import unittest

import torchbearer.callbacks as callbacks
import torchbearer


class TestDecorators(unittest.TestCase):

    def test_on_start(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_start(example).on_start(state) == state)

    def test_on_start_epoch(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_start_epoch(example).on_start_epoch(state) == state)

    def test_on_start_training(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_start_training(example).on_start_training(state) == state)

    def test_on_sample(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_sample(example).on_sample(state) == state)

    def test_on_forward(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_forward(example).on_forward(state) == state)

    def test_on_criterion(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_criterion(example).on_criterion(state) == state)

    def test_on_backward(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_backward(example).on_backward(state) == state)

    def test_on_step_training(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_step_training(example).on_step_training(state) == state)

    def test_on_end_training(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_end_training(example).on_end_training(state) == state)

    def test_on_end_epoch(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_end_epoch(example).on_end_epoch(state) == state)

    def test_on_end(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_end(example).on_end(state) == state)

    def test_on_start_validation(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_start_validation(example).on_start_validation(state) == state)

    def test_on_sample_validation(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_sample_validation(example).on_sample_validation(state) == state)

    def test_on_forward_validation(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_forward_validation(example).on_forward_validation(state) == state)

    def test_on_criterion_validation(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_criterion_validation(example).on_criterion_validation(state) == state)

    def test_on_end_validation(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_end_validation(example).on_end_validation(state) == state)

    def test_on_step_validation(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_step_validation(example).on_step_validation(state) == state)

    def test_add_to_loss(self):
        def example(state):
            return 1
        state = {'test': 'test', torchbearer.LOSS: 0}
        callbacks.add_to_loss(example).on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS] == 1)
        callbacks.add_to_loss(example).on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS] == 2)


