from unittest import TestCase
from unittest.mock import patch, Mock

from torchbearer.callbacks import TimerCallback


class TestTimer(TestCase):
    @patch('time.time')
    def test_update_time(self, time):
        time.return_value = 0
        timer = TimerCallback()
        time.return_value = 1
        timer.update_time('test', {})
        self.assertTrue(timer.get_timings()['test'] == 1)

        time.return_value = 3
        timer.update_time('test_2', {})
        self.assertTrue(timer.get_timings()['test_2'] == 2)

    def test_calls(self):
        timer = TimerCallback()
        timer.update_time = Mock()

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
        self.assertTrue(timer.update_time.call_count == 13)
