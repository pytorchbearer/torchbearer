import torchbearer

from torchbearer.callbacks import Callback

import math


class TerminateOnNaN(Callback):
    """Callback that terminates training when the given metric is nan or inf.
    """

    def __init__(self, monitor='running_loss'):
        """Create a TerminateOnNaN callback which montiors the given metric and halts training if its value is nan or
        inf.

        :param monitor: The metric name to monitor
        :type monitor: str
        """
        super(TerminateOnNaN, self).__init__()
        self._monitor = monitor

    def _step_training(self, state):
        value = state[torchbearer.METRICS][self._monitor]
        if value is not None:
            if math.isnan(value) or math.isinf(value):
                print('Batch %d: Invalid ' % (state[torchbearer.BATCH]) + self._monitor + ', terminating training')
                state[torchbearer.STOP_TRAINING] = True

    def on_step_training(self, state):
        if self._monitor in state[torchbearer.METRICS]:
            self.on_step_training = lambda inner_state: self._step_training(inner_state)
            return self._step_training(state)
        else:
            self.on_step_training = lambda inner_state: ...

    def _end_epoch(self, state):
        value = state[torchbearer.METRICS][self._monitor]
        if value is not None:
            if math.isnan(value) or math.isinf(value):
                print('Epoch %d: Invalid ' % (state[torchbearer.EPOCH]) + self._monitor + ', terminating')
                state[torchbearer.STOP_TRAINING] = True

    def on_end_epoch(self, state):
        if self._monitor in state[torchbearer.METRICS]:
            self.on_end_epoch = lambda inner_state: self._end_epoch(inner_state)
            return self._end_epoch(state)
        else:
            self.on_end_epoch = lambda inner_state: ...

    def _step_validation(self, state):
        value = state[torchbearer.METRICS][self._monitor]
        if value is not None:
            if math.isnan(value) or math.isinf(value):
                print('Batch %d: Invalid ' % (state[torchbearer.BATCH]) + self._monitor + ', terminating validation')
                state[torchbearer.STOP_TRAINING] = True

    def on_step_validation(self, state):
        if self._monitor in state[torchbearer.METRICS]:
            self.on_step_validation = lambda inner_state: self._step_validation(inner_state)
            return self._step_validation(state)
        else:
            self.on_step_validation = lambda inner_state: ...
