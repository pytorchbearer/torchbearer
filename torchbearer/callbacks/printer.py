from __future__ import print_function
from collections import OrderedDict
from functools import partial

from tqdm import tqdm

import torchbearer
from torchbearer.callbacks import Callback


def _format_metrics(metrics, rounder):
    # Adapted from https://github.com/tqdm/tqdm
    postfix = OrderedDict([])
    for key in sorted(metrics.keys()):
        postfix[key] = metrics[key]

    for key in postfix.keys():
        try:
            postfix[key] = str(rounder(postfix[key]))
        except TypeError:
            try:
                postfix[key] = str(list(map(rounder, postfix[key])))
            except TypeError:
                postfix[key] = str(postfix[key])
    postfix = ', '.join(key + '=' + postfix[key].strip() for key in postfix.keys())
    return postfix


class ConsolePrinter(Callback):
    """The ConsolePrinter callback simply outputs the training metrics to the console.

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import ConsolePrinter

        # Example Trial which forgoes the usual printer for a console printer
        >>> printer = ConsolePrinter()
        >>> trial = Trial(None, callbacks=[printer], verbose=0).for_steps(1).run()
        0/1(t):

    Args:
        validation_label_letter (str): This is the letter displayed after the epoch number indicating the current phase
            of training
        precision (int): Precision of the number format in decimal places

    State Requirements:
        - :attr:`torchbearer.state.EPOCH`: The current epoch number
        - :attr:`torchbearer.state.MAX_EPOCHS`: The total number of epochs for this run
        - :attr:`torchbearer.state.BATCH`: The current batch / iteration number
        - :attr:`torchbearer.state.STEPS`: The total number of steps / batches / iterations for this epoch
        - :attr:`torchbearer.state.METRICS`: The metrics dict to print
    """
    def __init__(self, validation_label_letter='v', precision=4):
        super(ConsolePrinter, self).__init__()
        self.validation_label = validation_label_letter
        self.rounder = partial(round, ndigits=precision)

    def _step(self, state, letter, steps):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        batch_str = '{:d}/{:d} '.format(state[torchbearer.BATCH], steps)
        stats_str = _format_metrics(state[torchbearer.METRICS], self.rounder)
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def _end(self, state, letter):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        stats_str = _format_metrics(state[torchbearer.METRICS], self.rounder)
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
    """The Tqdm callback outputs the progress and metrics for training and validation loops to the console using TQDM.
    The given key is used to label validation output.

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import Tqdm

        # Example Trial which forgoes the usual printer for a customised tqdm printer.
        >>> printer = Tqdm(precision=8)
        # Note that outputs are written to stderr, not stdout as shown in this example
        >>> trial = Trial(None, callbacks=[printer], verbose=0).for_steps(1).run(1)
        0/1(t): 100%|...| 1/1 [00:00<00:00, 29.40it/s]

    Args:
        tqdm_module: The tqdm module to use. If none, defaults to tqdm or tqdm_notebook if in notebook
        validation_label_letter (str): The letter to use for validation outputs.
        precision (int): Precision of the number format in decimal places
        on_epoch (bool): If True, output a single progress bar which tracks epochs
        tqdm_args: Any extra keyword args provided here will be passed through to the tqdm module constructor.
            See `github.com/tqdm/tqdm#parameters <https://github.com/tqdm/tqdm#parameters>`_ for more details.

    State Requirements:
        - :attr:`torchbearer.state.EPOCH`: The current epoch number
        - :attr:`torchbearer.state.MAX_EPOCHS`: The total number of epochs for this run
        - :attr:`torchbearer.state.STEPS`: The total number of steps / batches / iterations for this epoch
        - :attr:`torchbearer.state.METRICS`: The metrics dict to print
        - :attr:`torchbearer.state.HISTORY`: The history of the :class:`.Trial` object
    """
    def __init__(self, tqdm_module=None, validation_label_letter='v', precision=4, on_epoch=False, **tqdm_args):
        if torchbearer.magics.is_notebook() and tqdm_module is None:
            from tqdm import tqdm_notebook
            self.tqdm_module = tqdm_notebook
        else:
            self.tqdm_module = tqdm if tqdm_module is None else tqdm_module

        self._loader = None
        self.validation_label = validation_label_letter
        self.rounder = partial(round, ndigits=precision)
        self._on_epoch = on_epoch
        self.tqdm_args = tqdm_args

    def _on_start(self, state, letter):
        bar_desc = '{:d}/{:d}({:s})'.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        self._loader = self.tqdm_module(total=state[torchbearer.STEPS], desc=bar_desc, **self.tqdm_args)

    def _update(self, state):
        self._loader.update(1)
        self._loader.set_postfix_str(_format_metrics(state[torchbearer.METRICS], self.rounder))

    def _close(self, state):
        self._loader.set_postfix_str(_format_metrics(state[torchbearer.METRICS], self.rounder))
        self._loader.close()

    def on_start(self, state):
        if self._on_epoch:

            n = len(state[torchbearer.HISTORY])
            self._loader = self.tqdm_module(initial=n, total=state[torchbearer.MAX_EPOCHS], **self.tqdm_args)

            if n > 0:
                metrics = dict(state[torchbearer.HISTORY][-1])
                del metrics[str(torchbearer.TRAIN_STEPS)]
                del metrics[str(torchbearer.VALIDATION_STEPS)]
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

        Args:
            state (dict): The :class:`.Trial` state
        """
        if not self._on_epoch:
            self._on_start(state, 't')

    def on_step_training(self, state):
        """Update the bar with the metrics from this step.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if not self._on_epoch:
            self._update(state)

    def on_end_training(self, state):
        """Update the bar with the terminal training metrics and then close.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if not self._on_epoch:
            self._close(state)

    def on_start_validation(self, state):
        """Initialise the TQDM bar for this validation phase.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if not self._on_epoch:
            self._on_start(state, self.validation_label)

    def on_step_validation(self, state):
        """Update the bar with the metrics from this step.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if not self._on_epoch:
            self._update(state)

    def on_end_validation(self, state):
        """Update the bar with the terminal validation metrics and then close.

        Args:
            state (dict): The :class:`.Trial` state
        """
        if not self._on_epoch:
            self._close(state)
