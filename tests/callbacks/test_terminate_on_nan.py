from unittest import TestCase

import torchbearer
from torchbearer.callbacks import TerminateOnNaN


class TestTerminateOnNaN(TestCase):
    def test_should_terminate(self):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 0,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'running_loss': float('nan')}
        }

        terminator = TerminateOnNaN()

        terminator.on_step_training(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])
        terminator.on_step_validation(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])
        terminator.on_end_epoch(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_should_not_terminate(self):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 0,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'running_loss': 0.01}
        }

        terminator = TerminateOnNaN()

        terminator.on_step_training(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
        terminator.on_step_validation(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
        terminator.on_end_epoch(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_monitor_should_terminate(self):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 0,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'running_loss': 0.001, 'test_metric': float('nan')}
        }

        terminator = TerminateOnNaN(monitor='test_metric')

        terminator.on_step_training(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])
        terminator.on_step_validation(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])
        terminator.on_end_epoch(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_monitor_should_not_terminate(self):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 0,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'running_loss': 0.01, 'test_metric': 0.001}
        }

        terminator = TerminateOnNaN(monitor='test_metric')

        terminator.on_step_training(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
        terminator.on_step_validation(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
        terminator.on_end_epoch(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_not_found_metric(self):
        state = {
            torchbearer.EPOCH: 0,
            torchbearer.BATCH: 0,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'running_loss': 0.001, 'test_metric': float('nan')}
        }

        terminator = TerminateOnNaN(monitor='test')

        terminator.on_step_training(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
        terminator.on_step_validation(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
        terminator.on_end_epoch(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])
