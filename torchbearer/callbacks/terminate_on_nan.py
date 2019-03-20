from __future__ import print_function
import torchbearer

from torchbearer.callbacks import Callback

import math


class TerminateOnNaN(Callback):
    """Callback which montiors the given metric and halts training if its value is nan or inf.

    Args:
        monitor (str): The name of the metric to monitor

    State Requirements:
        - :attr:`torchbearer.state.METRICS`: Metrics should be a dict containing at least the key `monitor`
    """
    def __init__(self, monitor='running_loss'):
        super(TerminateOnNaN, self).__init__()
        self._monitor = monitor

    def _check(self, state):
        if self._monitor in state[torchbearer.METRICS]:
            value = state[torchbearer.METRICS][self._monitor]
            if value is not None:
                if math.isnan(value) or math.isinf(value):
                    print('Invalid ' + self._monitor + ', terminating')
                    state[torchbearer.STOP_TRAINING] = True

    def on_step_training(self, state):
        self._check(state)

    def on_end_epoch(self, state):
        self._check(state)

    def on_step_validation(self, state):
        self._check(state)
