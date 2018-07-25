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

    def update_time(self, text, state):
        self.time_dict[text] = time.time() - self.t0
        state[torchbearer.TIMINGS] = self.time_dict
        self.t0 = time.time()

    def on_start(self, state):
        self.t0 = time.time()
        self.update_time('OnStart', state)

    def on_start_training(self, state):
        super().on_start_training(state)
        self.update_time('OnStartTraining', state)

    def on_start_epoch(self, state):
        super().on_start_epoch(state)
        self.update_time('OnStartEpoch', state)

    def on_sample(self, state):
        super().on_sample(state)
        self.update_time('OnSample', state)

    def on_forward(self, state):
        super().on_forward(state)
        self.update_time('OnForward', state)

    def on_criterion(self, state):
        super().on_criterion(state)
        self.update_time('OnCriterion', state)

    def on_backward(self, state):
        super().on_backward(state)
        self.update_time('OnBackward', state)

    def on_step_training(self, state):
        super().on_step_training(state)
        self.update_time('OnStep', state)

    def on_start_validation(self, state):
        super().on_start_validation(state)
        self.update_time('OnStartValidation', state)

    def on_sample_validation(self, state):
        super().on_sample_validation(state)
        self.update_time('OnSampleValidation', state)

    def on_forward_validation(self, state):
        super().on_forward_validation(state)
        self.update_time('OnForwardValidation', state)

    def on_criterion_validation(self, state):
        super().on_criterion_validation(state)
        self.update_time('OnCriterionValidation', state)

    def on_step_validation(self, state):
        super().on_step_validation(state)
        self.update_time('OnStepValidation', state)

    def get_timings(self):
        return self.time_dict


