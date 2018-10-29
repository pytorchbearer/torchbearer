import torchbearer

from torchbearer.callbacks import Callback
from tqdm import tqdm
from collections import OrderedDict
from numbers import Number


def _format_num(n, precision):
    # Adapted from https://github.com/tqdm/tqdm
    f = ('{0:.' + str(precision) + 'g}').format(n).replace('+0', '+').replace('-0', '-')
    n = str(n)
    return f if len(f) < len(n) else n


def _format_metrics(metrics, precision):
    # Adapted from https://github.com/tqdm/tqdm
    postfix = OrderedDict([])
    for key in sorted(metrics.keys()):
        postfix[key] = metrics[key]

    for key in postfix.keys():
        if isinstance(postfix[key], Number):
            postfix[key] = _format_num(postfix[key], precision)
        elif not isinstance(postfix[key], str):
            postfix[key] = str(postfix[key])
    postfix = ', '.join(key + '=' + postfix[key].strip() for key in postfix.keys())
    return postfix


class ConsolePrinter(Callback):
    """The ConsolePrinter callback simply outputs the training metrics to the console.

    :param validation_label_letter: This is the letter displayed after the epoch number indicating the current phase of training
    :type validation_label_letter: String
    :param precision: Precision of the number format in significant figures
    :type precision: int
    """
    def __init__(self, validation_label_letter='v', precision=4):
        super().__init__()
        self.validation_label = validation_label_letter
        self.precision = precision

    def _step(self, state, letter, steps):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        batch_str = '{:d}/{:d} '.format(state[torchbearer.BATCH], steps)
        stats_str = _format_metrics(state[torchbearer.METRICS], self.precision)
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def _end(self, state, letter):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        stats_str = _format_metrics(state[torchbearer.METRICS], self.precision)
        print('\r' + epoch_str + stats_str)

    def on_step_training(self, state):
        self._step(state, 't', state[torchbearer.STEPS])

    def on_end_training(self, state):
        self._end(state, 't')

    def on_step_validation(self, state):
        self._step(state, self.validation_label, state[torchbearer.STEPS])

    def on_end_validation(self, state):
        self._end(state, self.validation_label)


class Tqdm(Callback):
    """The Tqdm callback outputs the progress and metrics for training and validation loops to the console using TQDM. The given key is used to label validation output.

    :param validation_label_letter: The letter to use for validation outputs.
    :type validation_label_letter: str
    :param precision: Precision of the number format in significant figures
    :type precision: int
    :param on_epoch: If True, output a single progress bar which tracks epochs
    :type on_epoch: bool
    :param tqdm_args: Any extra keyword args provided here will be passed through to the tqdm module constructor. See `github.com/tqdm/tqdm#parameters <https://github.com/tqdm/tqdm#parameters>`_ for more details.
    """
    def __init__(self, tqdm_module=tqdm, validation_label_letter='v', precision=4, on_epoch=False, **tqdm_args):
        self.tqdm_module = tqdm_module
        self._loader = None
        self.validation_label = validation_label_letter
        self.precision = precision
        self._on_epoch = on_epoch
        self.tqdm_args = tqdm_args

    def _on_start(self, state, letter):
        bar_desc = '{:d}/{:d}({:s})'.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        self._loader = self.tqdm_module(total=state[torchbearer.STEPS], desc=bar_desc, **self.tqdm_args)

    def _update(self, state):
        self._loader.update(1)
        self._loader.set_postfix_str(_format_metrics(state[torchbearer.METRICS], self.precision))

    def _close(self, state):
        self._loader.set_postfix_str(_format_metrics(state[torchbearer.METRICS], self.precision))
        self._loader.close()

    def on_start(self, state):
        if self._on_epoch:

            n = len(state[torchbearer.HISTORY])
            self._loader = self.tqdm_module(initial=n, total=state[torchbearer.MAX_EPOCHS], **self.tqdm_args)

            if n > 0:
                metrics = state[torchbearer.HISTORY][-1][1]
                state[torchbearer.METRICS] = metrics
                self._update(state)

    def on_end_epoch(self, state):
        if self._on_epoch:
            self._update(state)

    def on_end(self, state):
        if self._on_epoch:
            self._close(state)

    def on_start_training(self, state):
        """Initialise the TQDM bar for this training phase.

        :param state: The Model state
        :type state: dict
        """
        if not self._on_epoch:
            self._on_start(state, 't')

    def on_step_training(self, state):
        """Update the bar with the metrics from this step.

        :param state: The Model state
        :type state: dict
        """
        if not self._on_epoch:
            self._update(state)

    def on_end_training(self, state):
        """Update the bar with the terminal training metrics and then close.

        :param state: The Model state
        :type state: dict
        """
        if not self._on_epoch:
            self._close(state)

    def on_start_validation(self, state):
        """Initialise the TQDM bar for this validation phase.

        :param state: The Model state
        :type state: dict
        """
        if not self._on_epoch:
            self._on_start(state, self.validation_label)

    def on_step_validation(self, state):
        """Update the bar with the metrics from this step.

        :param state: The Model state
        :type state: dict
        """
        if not self._on_epoch:
            self._update(state)

    def on_end_validation(self, state):
        """Update the bar with the terminal validation metrics and then close.

        :param state: The Model state
        :type state: dict
        """
        if not self._on_epoch:
            self._close(state)
