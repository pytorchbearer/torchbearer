from __future__ import print_function
import sys
if sys.version_info[0] < 3:
    old_super = super

    def super(_, obj):
        return old_super(obj.__class__, obj)

import time
from torchbearer.callbacks import Callback
import torchbearer
from torchbearer.metrics import Metric

ON_START_TRAINING = 'on_start_training'
ON_START_EPOCH = 'on_start_epoch'
ON_SAMPLE = 'on_sample'
ON_FORWARD = 'on_forward'
ON_CRITERION = 'on_criterion'
ON_BACKWARD = 'on_backward'
ON_STEP_TRAINING = 'on_step_training'
ON_START_VALIDATION = 'on_start_validation'
ON_SAMPLE_VALIDATION = 'on_sample_validation'
ON_FORWARD_VALIDATION = 'on_forward_validaiton'
ON_CRITERION_VALIDATION = 'on_criterion_validation'
ON_STEP_VALIDATION = 'on_step_validation'
TRAIN_TIME = 'train_time'
TOTAL_TIME = 'total_time'
VALIDATION_TIME = 'validation_time'


class TimerMetric(Callback, Metric):
    def __init__(self, time_keys=()):
        """ Timer callback that aggregates timings for each stage of model execution
        """
        super(TimerMetric, self).__init__(name='timer')
        self.t0 = time.time()
        self.time_dict = {}
        # self.init_keys()
        self.batch_timer = _TimerMetric('t_batch')
        self.epoch_timer = _TimerMetric('t_epoch')
        self.train_timer = _TimerMetric('t_train')
        self.valid_timer = _TimerMetric('t_valid')
        self.total_timer = _TimerMetric('t_total')
        self.time_keys = time_keys
        self.added_callback = False

    def update_time(self, text, metric, state):
        self.time_dict[text] = metric.process(state)
        state[torchbearer.TIMINGS] = self.time_dict

    def process(self, *args):
        super(TimerMetric, self).process(*args)
        d_out = {key: self.time_dict[key] for key in self.time_keys if key in self.time_dict}
        return d_out

    def reset(self, state):
        super(TimerMetric, self).reset(state)
        if not self.added_callback:
            state[torchbearer.CALLBACK_LIST].append([self])
            self.added_callback = True

    def on_start(self, state):
        self.t0 = time.time()
        self.batch_timer.reset(state)
        self.epoch_timer.reset(state)
        self.train_timer.reset(state)
        self.valid_timer.reset(state)
        self.total_timer.reset(state)

    def on_start_training(self, state):
        super(TimerMetric, self).on_start_training(state)
        self.update_time(ON_START_TRAINING, self.batch_timer, state)
        self.update_time(ON_START_TRAINING, self.train_timer, state)

    def on_start_epoch(self, state):
        super(TimerMetric, self).on_start_epoch(state)
        self.update_time(ON_START_EPOCH, self.epoch_timer, state)

    def on_sample(self, state):
        super(TimerMetric, self).on_sample(state)
        self.update_time(ON_SAMPLE, self.batch_timer, state)

    def on_forward(self, state):
        super(TimerMetric, self).on_forward(state)
        self.update_time(ON_FORWARD, self.batch_timer, state)

    def on_criterion(self, state):
        super(TimerMetric, self).on_criterion(state)
        self.update_time(ON_CRITERION, self.batch_timer, state)

    def on_backward(self, state):
        super(TimerMetric, self).on_backward(state)
        self.update_time(ON_BACKWARD, self.batch_timer, state)

    def on_step_training(self, state):
        super(TimerMetric, self).on_step_training(state)
        self.update_time(ON_STEP_TRAINING, self.batch_timer, state)

    def on_start_validation(self, state):
        super(TimerMetric, self).on_start_validation(state)
        self.update_time(ON_START_VALIDATION, self.batch_timer, state)

    def on_sample_validation(self, state):
        super(TimerMetric, self).on_sample_validation(state)
        self.update_time(ON_SAMPLE_VALIDATION, self.batch_timer, state)

    def on_forward_validation(self, state):
        super(TimerMetric, self).on_forward_validation(state)
        self.update_time(ON_FORWARD_VALIDATION, self.batch_timer, state)

    def on_criterion_validation(self, state):
        super(TimerMetric, self).on_criterion_validation(state)
        self.update_time(ON_CRITERION_VALIDATION, self.batch_timer, state)

    def on_step_validation(self, state):
        super(TimerMetric, self).on_step_validation(state)
        self.update_time(ON_STEP_VALIDATION, self.batch_timer, state)

    def on_end_training(self, state):
        super(TimerMetric, self).on_end_training(state)
        self.valid_timer.reset(state)
        self.batch_timer.reset(state)
        self.update_time(TRAIN_TIME, self.train_timer, state)

    def on_end_epoch(self, state):
        super(TimerMetric, self).on_end_epoch(state)
        self.batch_timer.reset(state)
        self.train_timer.reset(state)

    def on_end(self, state):
        super(TimerMetric, self).on_end(state)
        self.update_time(TOTAL_TIME, self.total_timer, state)
        print(str(self.time_dict))

    def on_end_validation(self, state):
        super(TimerMetric, self).on_end_validation(state)
        self.update_time(VALIDATION_TIME, self.valid_timer, state)

    def get_timings(self):
        return self.time_dict


class _TimerMetric(Metric):
    def __init__(self, name):
        super(_TimerMetric, self).__init__(name)
        self.t = time.time()

    def process(self, *args):
        super(_TimerMetric, self).process(*args)
        dt = time.time() - self.t
        self.t = time.time()
        return dt

    def reset(self, state):
        super(_TimerMetric, self).reset(state)
        self.t = time.time()
