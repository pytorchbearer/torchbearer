import time
from torchbearer.callbacks import Callback
import torchbearer


class TimerCallback(Callback):
    def __init__(self):
        """ Timer callback that aggregates timings for each stage of model execution
        """
        super().__init__()
        self.t0 = time.time()
        self.time_dict = {}

    def update_time(self, text):
        self.time_dict[text] = time.time() - self.t0
        self.t0 = time.time()

    def on_start(self, state):
        self.t0 = time.time()
        self.update_time('OnStart')

    def on_start_training(self, state):
        super().on_start_training(state)
        self.update_time('OnStartTraining')

    def on_start_epoch(self, state):
        super().on_start_epoch(state)
        self.update_time('OnStartEpoch')

    def on_sample(self, state):
        super().on_sample(state)
        self.update_time('OnSample')

    def on_forward(self, state):
        super().on_forward(state)
        self.update_time('OnForward')

    def on_criterion(self, state):
        super().on_criterion(state)
        self.update_time('OnCriterion')

    def on_backward(self, state):
        super().on_backward(state)
        self.update_time('OnBackward')

    def on_step_training(self, state):
        super().on_step_training(state)
        self.update_time('OnStep')

    def on_end_training(self, state):
        super().on_end_training(state)
        self.update_time('OnEndTraining')

    def on_end_epoch(self, state):
        super().on_end_epoch(state)
        self.update_time('OnEndEpoch')
        state[torchbearer.TIMINGS] = self.time_dict

    def on_end(self, state):
        super().on_end(state)
        self.update_time('OnEnd')
        state[torchbearer.TIMINGS] = self.time_dict

    def on_start_validation(self, state):
        super().on_start_validation(state)
        self.update_time('OnStartValidation')

    def on_sample_validation(self, state):
        super().on_sample_validation(state)
        self.update_time('OnSampleValidation')

    def on_forward_validation(self, state):
        super().on_forward_validation(state)
        self.update_time('OnForwardValidation')

    def on_criterion_validation(self, state):
        super().on_criterion_validation(state)
        self.update_time('OnCriterionValidation')

    def on_end_validation(self, state):
        super().on_end_validation(state)
        self.update_time('OnEndValidation')

    def on_step_validation(self, state):
        super().on_step_validation(state)
        self.update_time('OnStepValidation')

    def get_timings(self):
        return self.time_dict


