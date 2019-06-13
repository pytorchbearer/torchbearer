from __future__ import print_function
import torchbearer

from torchbearer.callbacks import Callback

import math


class TerminateOnNaN(Callback):
    """Callback which montiors the given metric and halts training if its value is nan or inf.

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import TerminateOnNaN

        # Example Trial which terminates on a NaN, forced by a separate callback. Terminates on the 11th batch since
        the running loss only updates every 10 iterations.
        >>> term = TerminateOnNaN(monitor='running_loss')
        >>> @torchbearer.callbacks.on_criterion
        ... def force_terminate(state):
        ...     if state[torchbearer.BATCH] == 5:
        ...         state[torchbearer.LOSS] = state[torchbearer.LOSS] * torch.Tensor([float('NaN')])
        >>> trial = Trial(None, callbacks=[term, force_terminate], metrics=['loss'], verbose=2).for_steps(30).run(1)
        Invalid running_loss, terminating

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
