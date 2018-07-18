import torchbearer

from torchbearer.callbacks import Callback


class EarlyStopping(Callback):
    """Callback to stop training when a monitored quantity has stopped improving.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0, patience=0, verbose=0, mode='auto'):
        """Stop training when a monitored quantity has stopped improving.

        :param monitor: Quantity to be monitored
        :type monitor: str
        :param min_delta: Minimum change in the monitored quantity to qualify as an improvement, i.e. an absolute change
        of less than min_delta, will count as no improvement.
        :type min_delta: float
        :param patience: Number of epochs with no improvement after which training will be stopped.
        :type patience: int
        :param verbose: Verbosity mode, will print stopping info if verbose > 0
        :type verbose: int
        :param mode: One of {auto, min, max}. In `min` mode, training will stop when the quantity monitored has stopped
        decreasing; in `max` mode it will stop when the quantity monitored has stopped increasing; in `auto` mode, the
        direction is automatically inferred from the name of the monitored quantity.
        :type mode: str
        """
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.mode = mode

        self.wait = 0
        self.stopped_epoch = 0

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

    def on_start(self, state):
        self.wait = 0
        self.stopped_epoch = 0
        self.best = float('inf') if self.mode == 'min' else -float('inf')

    def on_end_epoch(self, state):
        current = state[torchbearer.METRICS][self.monitor]
        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = state[torchbearer.EPOCH]
                state[torchbearer.STOP_TRAINING] = True

    def on_end(self, state):
        if self.stopped_epoch > 0 and self.verbose > 0:
            print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))
