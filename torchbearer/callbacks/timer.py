import time
from torchbearer.callbacks import Callback
import torchbearer
from torchbearer.metrics import Metric


class TimerCallback(Callback, Metric):
    def __init__(self, name, time_keys=()):
        """ Timer callback that aggregates timings for each stage of model execution
        """
        super(TimerCallback, self).__init__(name=name)
        self.t0 = time.time()
        self.time_dict = {}
        self.batch_timer = TimerMetric('t_batch')
        self.epoch_timer = TimerMetric('t_epoch')
        self.train_timer = TimerMetric('t_train')
        self.valid_timer = TimerMetric('t_valid')
        self.total_timer = TimerMetric('t_total')
        self.time_keys = time_keys

    def update_time(self, text, metric, state):
        self.time_dict[text] = metric.process(state)
        state[torchbearer.TIMINGS] = self.time_dict

    def process(self, state):
        super().process(state)
        d_out = {key: self.time_dict[key] for key in self.time_keys if key in self.time_dict}
        return d_out

    def on_start(self, state):
        self.t0 = time.time()
        self.batch_timer.reset(state)
        self.epoch_timer.reset(state)
        self.train_timer.reset(state)
        self.valid_timer.reset(state)
        self.total_timer.reset(state)

    def on_start_training(self, state):
        super().on_start_training(state)
        self.update_time('OnStartTraining', self.batch_timer, state)
        self.update_time('OnStartTraining', self.train_timer, state)

    def on_start_epoch(self, state):
        super().on_start_epoch(state)
        self.update_time('OnStartEpoch', self.epoch_timer, state)

    def on_sample(self, state):
        super().on_sample(state)
        self.update_time('OnSample', self.batch_timer, state)

    def on_forward(self, state):
        super().on_forward(state)
        self.update_time('OnForward', self.batch_timer, state)

    def on_criterion(self, state):
        super().on_criterion(state)
        self.update_time('OnCriterion', self.batch_timer, state)

    def on_backward(self, state):
        super().on_backward(state)
        self.update_time('OnBackward', self.batch_timer, state)

    def on_step_training(self, state):
        super().on_step_training(state)
        self.update_time('OnStep', self.batch_timer, state)

    def on_start_validation(self, state):
        super().on_start_validation(state)
        self.update_time('OnStartValidation', self.batch_timer, state)

    def on_sample_validation(self, state):
        super().on_sample_validation(state)
        self.update_time('OnSampleValidation', self.batch_timer, state)

    def on_forward_validation(self, state):
        super().on_forward_validation(state)
        self.update_time('OnForwardValidation', self.batch_timer, state)

    def on_criterion_validation(self, state):
        super().on_criterion_validation(state)
        self.update_time('OnCriterionValidation', self.batch_timer, state)

    def on_step_validation(self, state):
        super().on_step_validation(state)
        self.update_time('OnStepValidation', self.batch_timer, state)

    def on_end_training(self, state):
        super().on_end_training(state)
        self.valid_timer.reset(state)
        self.batch_timer.reset(state)
        self.update_time('TrainTime', self.train_timer, state)

    def on_end_epoch(self, state):
        super().on_end_epoch(state)
        self.batch_timer.reset(state)
        self.train_timer.reset(state)

    def on_end(self, state):
        super().on_end(state)
        self.update_time('Total', self.total_timer, state)
        print(self.time_dict)

    def on_end_validation(self, state):
        super().on_end_validation(state)
        self.update_time('ValidationTime', self.valid_timer, state)

    def get_timings(self):
        return self.time_dict


class TimerMetric(torchbearer.metrics.Metric):
    def __init__(self, name):
        super().__init__(name)
        self.t = time.time()

    def process(self, *args):
        super().process(*args)
        dt = time.time() - self.t
        self.t = time.time()
        return dt

    def reset(self, state):
        super().reset(state)
        self.t = time.time()
