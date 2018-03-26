from bink.callbacks.callbacks import Callback
from tqdm import tqdm


class ConsolePrinter(Callback):
    def on_update_parameters(self, state):
        epoch_str = '{:d}/{:d}: '.format(state['epoch'] + 1, state['max_epochs'])
        batch_str = '{:d}/{:d} '.format(state['t'], state['train_steps'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def on_end_epoch(self, state):
        print()
        epoch_str = '{:d}/{:d}: '.format(state['epoch'] + 1, state['max_epochs'])
        stats_str = ', '.join(['{0}:{1:.03g}'.format(key, value) for (key, value) in state['final_metrics'].items()])
        print(epoch_str + stats_str)


class Tqdm(Callback):
    def __init__(self):
        self._loader = None

    def on_start_epoch(self, state):
        bar_desc = '{:d}/{:d}'.format(state['epoch'] + 1, state['max_epochs'])
        self._loader = tqdm(state['generator'], desc=bar_desc)
        state['generator'] = self._loader

    def on_update_parameters(self, state):
        self._loader.set_postfix(state['metrics'])

    def on_end_epoch(self, state):
        metrics = state['metrics'].copy()
        metrics.update(state['final_metrics'])
        self._loader.set_postfix(metrics)
        self._loader.close()
