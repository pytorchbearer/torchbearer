import torchbearer

from torchbearer.callbacks import Callback
from tqdm import tqdm


class ConsolePrinter(Callback):
    """The ConsolePrinter callback simply outputs the training metrics to the console.
    """
    def __init__(self, validation_label_letter='v'):
        super().__init__()
        self.validation_label = validation_label_letter

    def _step(self, state, letter, steps):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        batch_str = '{:d}/{:d} '.format(state[torchbearer.BATCH], steps)
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state[torchbearer.METRICS].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def _end(self, state, letter):
        epoch_str = '{:d}/{:d}({:s}): '.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state[torchbearer.METRICS].items()])
        print('\r' + epoch_str + stats_str)

    def on_step_training(self, state):
        self._step(state, 't', state[torchbearer.TRAIN_STEPS])

    def on_end_training(self, state):
        self._end(state, 't')

    def on_step_validation(self, state):
        self._step(state, self.validation_label, state[torchbearer.VALIDATION_STEPS])

    def on_end_validation(self, state):
        self._end(state, self.validation_label)


class Tqdm(Callback):
    """The Tqdm callback outputs the progress and metrics for training and validation loops to the console using TQDM.
    """

    def __init__(self, validation_label_letter='v'):
        """Create Tqdm callback which uses the given key to label validation output.

        :param validation_label_letter: The letter to use for validation outputs.
        :type validation_label_letter: str
        """
        self._loader = None
        self.validation_label = validation_label_letter

    def _on_start(self, state, letter, steps):
        bar_desc = '{:d}/{:d}({:s})'.format(state[torchbearer.EPOCH], state[torchbearer.MAX_EPOCHS], letter)
        self._loader = tqdm(total=steps, desc=bar_desc)

    def _update(self, state):
        self._loader.update(1)
        self._loader.set_postfix(state[torchbearer.METRICS])

    def _close(self, state):
        self._loader.set_postfix(state[torchbearer.METRICS])
        self._loader.close()

    def on_start_training(self, state):
        """Initialise the TQDM bar for this training phase.

        :param state: The Model state
        :type state: dict
        """
        self._on_start(state, 't', state[torchbearer.TRAIN_STEPS])

    def on_step_training(self, state):
        """Update the bar with the metrics from this step.

        :param state: The Model state
        :type state: dict
        """
        self._update(state)

    def on_end_training(self, state):
        """Update the bar with the terminal training metrics and then close.

        :param state: The Model state
        :type state: dict
        """
        self._close(state)

    def on_start_validation(self, state):
        """Initialise the TQDM bar for this validation phase.

        :param state: The Model state
        :type state: dict
        """
        self._on_start(state, self.validation_label, state[torchbearer.VALIDATION_STEPS])

    def on_step_validation(self, state):
        """Update the bar with the metrics from this step.

        :param state: The Model state
        :type state: dict
        """
        self._update(state)

    def on_end_validation(self, state):
        """Update the bar with the terminal validation metrics and then close.

        :param state: The Model state
        :type state: dict
        """
        self._close(state)
