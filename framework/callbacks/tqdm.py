from framework.callbacks.callbacks import Callback

from tqdm import tqdm


class Tqdm(Callback):
    def __init__(self):
        self._loader = None

    def on_start_epoch(self, state):
        self._loader = tqdm(state['generator'])
        state['generator'] = self._loader

    def on_update_parameters(self, state):
        self._loader.set_postfix(state['metrics'])

    def on_end_epoch(self, state):
        self._loader.close()
