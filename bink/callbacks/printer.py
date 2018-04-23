from bink.callbacks import Callback
from tqdm import tqdm


class ConsolePrinter(Callback):
    def on_step_training(self, state):
        epoch_str = '{:d}/{:d}: '.format(state['epoch'] + 1, state['max_epochs'])
        batch_str = '{:d}/{:d} '.format(state['t'], state['train_steps'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def on_end_training(self, state):
        print()
        epoch_str = '{:d}/{:d}: '.format(state['epoch'] + 1, state['max_epochs'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['final_metrics'].items()])
        print(epoch_str + stats_str)


class Tqdm(Callback):
    def __init__(self, validation_label_letter='v'):
        self._loader = None
        self.validation_label = validation_label_letter

    def on_start_training(self, state):
        bar_desc = '{:d}/{:d}(t)'.format(state['epoch'] + 1, state['max_epochs'])
        self._loader = tqdm(total=state['train_steps'], desc=bar_desc)

    def on_step_training(self, state):
        self._loader.update(1)
        self._loader.set_postfix(state['metrics'])

    def on_end_training(self, state):
        self._loader.set_postfix(state['metrics'])
        self._loader.close()

    def on_start_validation(self, state):
        bar_desc = '{:d}/{:d}({:s})'.format(state['epoch'] + 1, state['max_epochs'], self.validation_label)
        self._loader = tqdm(total=state['validation_steps'], desc=bar_desc)

    def on_step_validation(self, state):
        self._loader.update(1)
        self._loader.set_postfix(state['metrics'])

    def on_end_validation(self, state):
        self._loader.set_postfix(state['metrics'])
        self._loader.close()
