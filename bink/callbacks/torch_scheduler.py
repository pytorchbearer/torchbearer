from bink.callbacks.callbacks import Callback


class TorchScheduler(Callback):
    def __init__(self, scheduler, metric=None):
        self._scheduler = scheduler
        self._metric = metric

    def on_start_epoch(self, state):
        if self._metric is None:
            self._scheduler.step()

    def on_end_epoch(self, state):
        if self._metric is not None:
            self._scheduler.step(state['final_metrics'][self._metric])
