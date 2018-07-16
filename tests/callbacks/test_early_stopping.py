from unittest import TestCase

import sconce
from sconce.callbacks import EarlyStopping


class TestEarlyStopping(TestCase):
    
    def test_min_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric'] = 0.01
        stopper.on_end_epoch(state)

        self.assertTrue(state[sconce.STOP_TRAINING])

    def test_min_should_continue(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric'] = 0.0001

        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

    def test_max_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric',  mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric'] = 0.0001
        stopper.on_end_epoch(state)

        self.assertTrue(state[sconce.STOP_TRAINING])

    def test_max_should_continue(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric'] = 0.01
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

    def test_max_equal_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        stopper.on_end_epoch(state)

        self.assertTrue(state[sconce.STOP_TRAINING])

    def test_in_equal_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        stopper.on_end_epoch(state)

        self.assertTrue(state[sconce.STOP_TRAINING])

    def test_patience_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', patience=3)

        stopper.on_start(state)

        for i in range(3):
            stopper.on_end_epoch(state)
            self.assertFalse(state[sconce.STOP_TRAINING])

        stopper.on_end_epoch(state)
        self.assertTrue(state[sconce.STOP_TRAINING])

    def test_patience_should_continue(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', patience=3)

        stopper.on_start(state)

        for i in range(3):
            stopper.on_end_epoch(state)
            self.assertFalse(state['stop_training'])

        state[sconce.METRICS]['test_metric'] = 0.0001
        stopper.on_end_epoch(state)
        self.assertFalse(state[sconce.STOP_TRAINING])

    def test_min_delta_should_continue(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max', min_delta=0.1)

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric'] = 0.102
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

    def test_min_delta_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max', min_delta=0.1)

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric'] = 0.10
        stopper.on_end_epoch(state)

        self.assertTrue(state[sconce.STOP_TRAINING])

    def test_auto_should_be_min(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertTrue(stopper.mode == 'min')

    def test_auto_should_be_max(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'acc_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='acc_metric')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertTrue(stopper.mode == 'max')

    def test_monitor_should_continue(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric_1': 0.001, 'test_metric_2': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric_2', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric_1'] = 0.0001
        state[sconce.METRICS]['test_metric_2'] = 0.01
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

    def test_monitor_should_stop(self):
        state = {
            sconce.EPOCH: 1,
            sconce.STOP_TRAINING: False,
            sconce.METRICS: {'test_metric_1': 0.001, 'test_metric_2': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric_2', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state[sconce.STOP_TRAINING])

        state[sconce.METRICS]['test_metric_1'] = 0.1
        state[sconce.METRICS]['test_metric_2'] = 0.0001
        stopper.on_end_epoch(state)

        self.assertTrue(state[sconce.STOP_TRAINING])
