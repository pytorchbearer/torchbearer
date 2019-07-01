from unittest import TestCase
from mock import MagicMock

import torchbearer
from torchbearer.callbacks import EarlyStopping


class TestEarlyStopping(TestCase):
    def test_step_on_batch(self):
        stopper = EarlyStopping(monitor='test_metric', mode='min', step_on_batch=True)

        stopper.step = MagicMock()

        stopper.on_step_training('test')
        self.assertTrue(stopper.step.call_count == 1)

        stopper.on_end_epoch('test')
        self.assertTrue(stopper.step.call_count == 1)

    def test_min_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.01
        stopper.on_end_epoch(state)

        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_min_should_continue(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.0001

        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_max_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric',  mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.0001
        stopper.on_end_epoch(state)

        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_max_should_continue(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.01
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_max_equal_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        stopper.on_end_epoch(state)

        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_in_equal_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        stopper.on_end_epoch(state)

        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_patience_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', patience=3)

        stopper.on_start(state)

        for i in range(3):
            stopper.on_end_epoch(state)
            self.assertFalse(state[torchbearer.STOP_TRAINING])

        stopper.on_end_epoch(state)
        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_patience_should_continue(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', patience=3)

        stopper.on_start(state)

        for i in range(3):
            stopper.on_end_epoch(state)
            self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.0001
        stopper.on_end_epoch(state)
        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_min_delta_should_continue(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max', min_delta=0.1)

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.102
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_min_delta_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max', min_delta=0.1)

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric'] = 0.10
        stopper.on_end_epoch(state)

        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_auto_should_be_min(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertTrue(stopper.mode == 'min')

    def test_auto_should_be_max(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'acc_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='acc_metric')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertTrue(stopper.mode == 'max')

    def test_monitor_should_continue(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric_1': 0.001, 'test_metric_2': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric_2', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric_1'] = 0.0001
        state[torchbearer.METRICS]['test_metric_2'] = 0.01
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

    def test_monitor_should_stop(self):
        state = {
            torchbearer.EPOCH: 1,
            torchbearer.STOP_TRAINING: False,
            torchbearer.METRICS: {'test_metric_1': 0.001, 'test_metric_2': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric_2', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[torchbearer.STOP_TRAINING])

        state[torchbearer.METRICS]['test_metric_1'] = 0.1
        state[torchbearer.METRICS]['test_metric_2'] = 0.0001
        stopper.on_end_epoch(state)

        self.assertTrue(state[torchbearer.STOP_TRAINING])

    def test_state_dict(self):
        stopper = EarlyStopping(monitor='test_metric_1')
        stopper.wait = 10
        stopper.best = 20
        state = stopper.state_dict()

        stopper = EarlyStopping(monitor='test_metric_1')
        self.assertNotEqual(stopper.wait, 10)

        stopper.load_state_dict(state)
        self.assertEqual(stopper.wait, 10)
        self.assertEqual(stopper.best, 20)
