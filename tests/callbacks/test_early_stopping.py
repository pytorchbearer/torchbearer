from unittest import TestCase
from bink.callbacks import EarlyStopping

class TestEarlyStopping(TestCase):
    
    def test_min_should_stop(self):
        state = {
            'epoch':1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.01
        stopper.on_end_epoch(state)

        self.assertTrue(state['stop_training'])

    def test_min_should_continue(self):
        state = {
            'epoch':1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.0001

        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

    def test_max_should_stop(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric',  mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.0001
        stopper.on_end_epoch(state)

        self.assertTrue(state['stop_training'])

    def test_max_should_continue(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.01
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

    def test_max_equal_should_stop(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        stopper.on_end_epoch(state)

        self.assertTrue(state['stop_training'])

    def test_in_equal_should_stop(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='min')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        stopper.on_end_epoch(state)

        self.assertTrue(state['stop_training'])

    def test_patience_should_stop(self):
        state = {
            'epoch':1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', patience=3)

        stopper.on_start(state)

        for i in range(3):
            stopper.on_end_epoch(state)
            self.assertFalse(state['stop_training'])

        stopper.on_end_epoch(state)
        self.assertTrue(state['stop_training'])

    def test_patience_should_continue(self):
        state = {
            'epoch':1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', patience=3)

        stopper.on_start(state)

        for i in range(3):
            stopper.on_end_epoch(state)
            self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.0001
        stopper.on_end_epoch(state)
        self.assertFalse(state['stop_training'])

    def test_min_delta_should_continue(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max', min_delta=0.1)

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.102
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

    def test_min_delta_should_stop(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric', mode='max', min_delta=0.1)

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric'] = 0.10
        stopper.on_end_epoch(state)

        self.assertTrue(state['stop_training'])

    def test_auto_should_be_min(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertTrue(stopper.mode == 'min')

    def test_auto_should_be_max(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'acc_metric': 0.001}
        }

        stopper = EarlyStopping(monitor='acc_metric')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertTrue(stopper.mode == 'max')

    def test_monitor_should_continue(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric_1': 0.001, 'test_metric_2': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric_2', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric_1'] = 0.0001
        state['metrics']['test_metric_2'] = 0.01
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

    def test_monitor_should_stop(self):
        state = {
            'epoch': 1,
            'stop_training': False,
            'metrics': {'test_metric_1': 0.001, 'test_metric_2': 0.001}
        }

        stopper = EarlyStopping(monitor='test_metric_2', mode='max')

        stopper.on_start(state)
        stopper.on_end_epoch(state)

        self.assertFalse(state['stop_training'])

        state['metrics']['test_metric_1'] = 0.1
        state['metrics']['test_metric_2'] = 0.0001
        stopper.on_end_epoch(state)

        self.assertTrue(state['stop_training'])