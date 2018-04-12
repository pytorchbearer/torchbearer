from io import StringIO
from unittest import TestCase
from unittest.mock import patch

from bink.callbacks import TerminateOnNaN


class TestTerminateOnNaN(TestCase):

    @patch('sys.stdout', new_callable=StringIO)
    def test_should_terminate(self, _):
        state = {
            'epoch': 0,
            't': 0,
            'stop_training': False,
            'metrics': {'running_loss': float('nan')}
        }

        terminator = TerminateOnNaN()

        terminator.on_step_training(state)
        self.assertTrue(state['stop_training'])
        terminator.on_end_epoch(state)
        self.assertTrue(state['stop_training'])

    def test_should_not_terminate(self):
        state = {
            'epoch': 0,
            't': 0,
            'stop_training': False,
            'metrics': {'running_loss': 0.01}
        }

        terminator = TerminateOnNaN()

        terminator.on_step_training(state)
        self.assertFalse(state['stop_training'])
        terminator.on_end_epoch(state)
        self.assertFalse(state['stop_training'])

    @patch('sys.stdout', new_callable=StringIO)
    def test_nan_as_string(self, _):
        state = {
            'epoch': 0,
            't': 0,
            'stop_training': False,
            'metrics': {'running_loss': 'nan'}
        }

        terminator = TerminateOnNaN()

        terminator.on_step_training(state)
        self.assertTrue(state['stop_training'])
        terminator.on_end_epoch(state)
        self.assertTrue(state['stop_training'])

    @patch('sys.stdout', new_callable=StringIO)
    def test_monitor_should_terminate(self, _):
        state = {
            'epoch': 0,
            't': 0,
            'stop_training': False,
            'metrics': {'running_loss': 0.001, 'test_metric': float('nan')}
        }

        terminator = TerminateOnNaN(monitor='test_metric')

        terminator.on_step_training(state)
        self.assertTrue(state['stop_training'])
        terminator.on_end_epoch(state)
        self.assertTrue(state['stop_training'])

    def test_monitor_should_not_terminate(self):
        state = {
            'epoch': 0,
            't': 0,
            'stop_training': False,
            'metrics': {'running_loss': 0.01, 'test_metric': 0.001}
        }

        terminator = TerminateOnNaN(monitor='test_metric')

        terminator.on_step_training(state)
        self.assertFalse(state['stop_training'])
        terminator.on_end_epoch(state)
        self.assertFalse(state['stop_training'])
