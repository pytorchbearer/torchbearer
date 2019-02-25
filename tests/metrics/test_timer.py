from unittest import TestCase

from mock import Mock, MagicMock, patch

import torchbearer
from torchbearer.metrics.timer import TimerMetric, _TimerMetric


class TestTimer(TestCase):
    def test_update_time(self):
        timer = TimerMetric('test')
        timerMetric = TimerMetric('test2')
        timerMetric.process = Mock(return_value=1)
        timer.update_time('test', timerMetric, {})
        self.assertTrue(timer.get_timings()['test'] == 1)

        timerMetric.process = Mock(return_value=2)
        timer.update_time('test_2', timerMetric, {})
        self.assertTrue(timer.get_timings()['test_2'] == 2)

    def test_calls(self):
        timer = TimerMetric('test')
        timer.batch_timer = MagicMock()
        timer.epoch_timer = MagicMock()
        timer.train_timer = MagicMock()
        timer.total_timer = MagicMock()
        timer.valid_timer = MagicMock()

        timer.on_start({})
        timer.on_start_training({})
        timer.on_start_epoch({})
        timer.on_sample({})
        timer.on_forward({})
        timer.on_criterion({})
        timer.on_backward({})
        timer.on_step_training({})
        timer.on_start_validation({})
        timer.on_sample_validation({})
        timer.on_forward_validation({})
        timer.on_criterion_validation({})
        timer.on_step_validation({})
        timer.on_end_training({})
        timer.on_end_validation({})
        timer.on_end_epoch({})
        timer.on_end({})

        self.assertTrue(timer.batch_timer.process.call_count == 11)
        self.assertTrue(timer.total_timer.process.call_count == 1)
        self.assertTrue(timer.epoch_timer.process.call_count == 1)
        self.assertTrue(timer.train_timer.process.call_count == 2)
        self.assertTrue(timer.valid_timer.process.call_count == 1)

    def test_process(self):
        timer = TimerMetric((torchbearer.metrics.timer.ON_FORWARD, ))
        timer.time_dict = {torchbearer.metrics.timer.ON_FORWARD: 1, 'test': 2}
        self.assertTrue(timer.process({})[torchbearer.metrics.timer.ON_FORWARD] == 1)

    def test_reset(self):
        state = {torchbearer.CALLBACK_LIST: torchbearer.callbacks.CallbackList([])}

        timer = TimerMetric()
        self.assertTrue(state[torchbearer.CALLBACK_LIST].callback_list == [])
        timer.reset(state)
        self.assertIsInstance(state[torchbearer.CALLBACK_LIST].callback_list[0], TimerMetric)

        timer.reset(state)
        self.assertTrue(len(state[torchbearer.CALLBACK_LIST].callback_list) == 1)


class TestTimerMetric(TestCase):
    @patch('time.time')
    def test_process(self, time):
        time.return_value = 1
        timer_metric = _TimerMetric('test')
        time.return_value = 2
        dt = timer_metric.process({})

        self.assertTrue(dt == 1)

    @patch('time.time')
    def test_reset(self, time):
        time.return_value = 1
        timer_metric = _TimerMetric('test')

        time.return_value = 3
        timer_metric.reset({})
        self.assertTrue(timer_metric.t == 3)

