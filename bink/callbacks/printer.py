from bink.callbacks import Callback
from tqdm import tqdm


class ConsolePrinter(Callback):
    """The ConsolePrinter callback simply outputs the training metrics to the console.
    """
    def __init__(self, validation_label_letter='v'):
        super().__init__()
        self.validation_label = validation_label_letter

    def on_step_training(self, state):
        epoch_str = '{:d}/{:d}(t): '.format(state['epoch'] + 1, state['max_epochs'])
        batch_str = '{:d}/{:d} '.format(state['t'], state['train_steps'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def on_end_training(self, state):
        epoch_str = '{:d}/{:d}(t): '.format(state['epoch'] + 1, state['max_epochs'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + stats_str)

    def on_step_validation(self, state):
        epoch_str = '{:d}/{:d}({:s}): '.format(state['epoch'] + 1, state['max_epochs'], self.validation_label)
        batch_str = '{:d}/{:d} '.format(state['t'], state['validation_steps'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def on_end_validation(self, state):
        epoch_str = '{:d}/{:d}({:s}): '.format(state['epoch'] + 1, state['max_epochs'], self.validation_label)
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + stats_str)


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

    def on_start_training(self, state):
        """Initialise the TQDM bar for this training phase.

        :param state: The Model state
        :type state: dict
        """
        bar_desc = '{:d}/{:d}(t)'.format(state['epoch'] + 1, state['max_epochs'])
        self._loader = tqdm(total=state['train_steps'], desc=bar_desc)

    def on_step_training(self, state):
        """Update the bar with the metrics from this step.

        :param state: The Model state
        :type state: dict
        """
        self._loader.update(1)
        self._loader.set_postfix(state['metrics'])

    def on_end_training(self, state):
        """Update the bar with the terminal training metrics and then close.

        :param state: The Model state
        :type state: dict
        """
        self._loader.set_postfix(state['metrics'])
        self._loader.close()

    def on_start_validation(self, state):
        """Initialise the TQDM bar for this validation phase.

        :param state: The Model state
        :type state: dict
        """
        bar_desc = '{:d}/{:d}({:s})'.format(state['epoch'] + 1, state['max_epochs'], self.validation_label)
        self._loader = tqdm(total=state['validation_steps'], desc=bar_desc)

    def on_step_validation(self, state):
        """Update the bar with the metrics from this step.

        :param state: The Model state
        :type state: dict
        """
        self._loader.update(1)
        self._loader.set_postfix(state['metrics'])

    def on_end_validation(self, state):
        """Update the bar with the terminal validation metrics and then close.

        :param state: The Model state
        :type state: dict
        """
        self._loader.set_postfix(state['metrics'])
        self._loader.close()
