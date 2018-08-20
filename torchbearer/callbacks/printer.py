import torchbearer

from torchbearer.callbacks import Callback
from tqdm import tqdm


class ConsolePrinter(Callback):
    """The ConsolePrinter callback simply outputs the training metrics to the console.

    :param validation_label_letter: This is the letter displayed after the epoch number indicating the current phase of training
    :type validation_label_letter: String
    """
    def __init__(self, validation_label_letter='v'):
        super().__init__()
        self.validation_label = validation_label_letter

    @staticmethod
    def _step(state, letter, steps):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        batch_str = '{:d}/{:d} '.format(state[torchbearer.BATCH], steps)
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state[torchbearer.METRICS].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    @staticmethod
    def _end(state, letter):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state[torchbearer.METRICS].items()])
        print('\r' + epoch_str + stats_str)

    def on_step_training(self, state):
        ConsolePrinter._step(state, 't', state[torchbearer.TRAIN_STEPS])

    def on_end_training(self, state):
        ConsolePrinter._end(state, 't')

    def on_step_validation(self, state):
        ConsolePrinter._step(state, self.validation_label, state[torchbearer.VALIDATION_STEPS])

    def on_end_validation(self, state):
        ConsolePrinter._end(state, self.validation_label)


class Tqdm(Callback):
    """The Tqdm callback outputs the progress and metrics for training and validation loops to the console using TQDM. The given key is used to label validation output.

    :param validation_label_letter: The letter to use for validation outputs.
    :type validation_label_letter: str
    :param on_epoch: If True, output a single progress bar which tracks epochs
    :type on_epoch: bool
    :param tqdm_args: Any extra keyword args provided here will be passed through to the tqdm module constructor. See `github.com/tqdm/tqdm#parameters <https://github.com/tqdm/tqdm#parameters>`_ for more details.
    """
    def __init__(self, tqdm_module=tqdm, validation_label_letter='v', on_epoch=False, **tqdm_args):
        self.tqdm_module = tqdm_module
        self._loader = None
        self.validation_label = validation_label_letter
        self._on_epoch = on_epoch
        self.tqdm_args = tqdm_args

    def _on_start(self, state, letter, steps):
        bar_desc = '{:d}/{:d}({:s})'.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        self._loader = self.tqdm_module(total=steps, desc=bar_desc, **self.tqdm_args)

    def _update(self, state):
        self._loader.update(1)
        self._loader.set_postfix(state[torchbearer.METRICS])

    def _close(self, state):
        self._loader.set_postfix(state[torchbearer.METRICS])
        self._loader.close()

    def on_start(self, state):
        if self._on_epoch:
            self._loader = self.tqdm_module(total=state[torchbearer.MAX_EPOCHS], **self.tqdm_args)

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
            self._on_start(state, 't', state[torchbearer.TRAIN_STEPS])

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
            self._on_start(state, self.validation_label, state[torchbearer.VALIDATION_STEPS])

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
