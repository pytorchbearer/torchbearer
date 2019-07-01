import unittest

import torchbearer.callbacks as callbacks
import torchbearer


class TestDecorators(unittest.TestCase):

    def test_multi(self):
        def example(state):
            return state
        state = 'test'
        c = callbacks.on_backward(callbacks.on_sample(callbacks.on_start(example)))
        self.assertTrue(c.on_backward(state) == state)
        self.assertTrue(c.on_sample(state) == state)
        self.assertTrue(c.on_start(state) == state)

    def test_target(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.bind_to(callbacks.on_start)(example).on_start(state) == state)

    def test_on_init(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_init(example).on_init(state) == state)

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

    def test_on_checkpoint(self):
        def example(state):
            return state
        state = 'test'
        self.assertTrue(callbacks.on_checkpoint(example).on_checkpoint(state) == state)

    def test_add_to_loss(self):
        def example(state):
            return 1
        state = {'test': 'test', torchbearer.LOSS: 0}
        callbacks.add_to_loss(example).on_criterion(state)
        self.assertTrue(state[torchbearer.LOSS] == 1)
        callbacks.add_to_loss(example).on_criterion_validation(state)
        self.assertTrue(state[torchbearer.LOSS] == 2)

    def test_once(self):
        class Example(callbacks.Callback):
            @callbacks.once
            def on_step_validation(self, state):
                state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = Example()

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb2 = Example()

        cb2.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb2.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state[torchbearer.EPOCH] += 1
        cb2.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

    def test_once_per_epoch(self):
        class Example(callbacks.Callback):
            @callbacks.once_per_epoch
            def on_step_validation(self, state):
                state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = Example()

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb2 = Example()

        cb2.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb2.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state[torchbearer.EPOCH] += 1
        cb2.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

    def test_only_if(self):
        class Example(callbacks.Callback):
            @callbacks.only_if(lambda s: s[torchbearer.EPOCH] % 2 == 0)
            def on_step_validation(self, state):
                state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = Example()

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 3)

    def test_once_lambda(self):
        @callbacks.once
        @callbacks.on_step_validation
        def callback_func(state):
            state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = callback_func

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

    def test_once_per_epoch_lambda(self):
        @callbacks.once_per_epoch
        @callbacks.on_step_validation
        def callback_func(state):
            state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = callback_func

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

    def test_only_if_lambda(self):
        @callbacks.only_if(lambda s: s[torchbearer.EPOCH] % 2 == 0)
        @callbacks.on_step_validation
        def callback_func(state):
            state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = callback_func

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 3)

    def test_lambda_only_if(self):
        @callbacks.on_step_validation
        @callbacks.only_if(lambda s: s[torchbearer.EPOCH] % 2 == 0)
        def callback_func(state):
            state['value'] += 1

        state = {torchbearer.EPOCH: 0, 'value': 0}

        cb = callback_func

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 1)

        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 2)

        state[torchbearer.EPOCH] += 1
        cb.on_step_validation(state)
        self.assertTrue(state['value'] == 3)