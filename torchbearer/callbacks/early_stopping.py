from __future__ import print_function
import torchbearer

from torchbearer.callbacks import Callback
from .decorators import only_if


class EarlyStopping(Callback):
    """Callback to stop training when a monitored quantity has stopped improving.

    Example: ::

        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import EarlyStopping

        # Example Trial which does early stopping if the validation accuracy drops below the max seen for 5 epochs in a row
        >>> stopping = EarlyStopping(monitor='val_acc', patience=5, mode='max')
        >>> trial = Trial(None, callbacks=[stopping], metrics=['acc'])

    Args:
        monitor (str): Name of quantity in metrics to be monitored
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute
            change of less than min_delta, will count as no improvement.
        patience (int): Number of epochs with no improvement after which training will be stopped.
        mode (str): One of {auto, min, max}. In `min` mode, training will stop when the quantity monitored has stopped
            decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing; in `auto` mode,
            the direction is automatically inferred from the name of the monitored quantity.

    State Requirements:
        - :attr:`torchbearer.state.METRICS`: Metrics should be a dict containing the given monitor key as a minimum
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, mode='auto', step_on_batch=False):

        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.mode = mode
        self.step_on_batch = step_on_batch

        if self.mode not in ['min', 'max']:
            if 'acc' in self.monitor:
                self.mode = 'max'
            else:
                self.mode = 'min'

        if self.mode == 'min':
            self.min_delta *= -1
            self.monitor_op = lambda x1, x2: x1 < x2
        elif self.mode == 'max':
            self.min_delta *= 1
            self.monitor_op = lambda x1, x2: x1 > x2

        self.wait = 0
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def state_dict(self):
        state_dict = {
            'wait': self.wait,
            'best': self.best
        }
        return state_dict

    def load_state_dict(self, state_dict):
        self.wait = state_dict['wait']
        self.best = state_dict['best']

    def step(self, state):
        current = state[torchbearer.METRICS][self.monitor]
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                state[torchbearer.STOP_TRAINING] = True

    @only_if(lambda self, _: self.step_on_batch)
    def on_step_training(self, state):
        self.step(state)

    @only_if(lambda self, _: not self.step_on_batch)
    def on_end_epoch(self, state):
        self.step(state)
