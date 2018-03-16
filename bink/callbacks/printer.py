from bink.callbacks.callbacks import Callback
from tqdm import tqdm


class ConsolePrinter(Callback):
    def on_update_parameters(self, state):
        epoch_str = '{:d}/{:d} '.format(state['epoch'] + 1, state['max_epochs'])
        batch_str = '{:d}/{:d} '.format(state['t']+1, state['train_steps'])
        stats_str = ', '.join(['%s:%.03g' % (key, value) for (key, value) in state['metrics'].items()])
        print('\r' + epoch_str + batch_str + stats_str, end='')

    def on_end_epoch(self, state):
        epoch_str = '{:d}/{:d}: '.format(state['epoch'] + 1, state['max_epochs'])
        stats_str = ', '.join(['%s:%.03g' % (key, value) for (key, value) in state['final_metrics'].items()])
        print('\r' + epoch_str + stats_str, end='\n')


class Tqdm(ConsolePrinter):
    def __init__(self):
        self._loader = None

    def on_start_epoch(self, state):
        bar_desc = '{:d}/{:d}'.format(state['epoch'] + 1, state['max_epochs'])
        self._loader = tqdm(state['generator'], desc=bar_desc)
        state['generator'] = self._loader

    def on_update_parameters(self, state):
        self._loader.set_postfix(state['metrics'])

    def on_end_epoch(self, state):
        super(Tqdm, self).on_end_epoch(state)
        self._loader.close()
