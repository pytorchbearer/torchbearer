import sys
import os

import torchbearer
from torchbearer.callbacks import Callback


class no_print:
    def __enter__(self):
        self.stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, *exc):
        sys.stdout = self.stdout
        return False


class LiveLossPlot(Callback):
    """
    Callback to write metrics to `LiveLossPlot <https://github.com/stared/livelossplot>`_, a library for visualisation in notebooks

    :param on_batch: If True, batch metrics will be logged. Else batch metrics will not be logged
    :param batch_step_size: The number of batches between logging metrics
    :param on_epoch: If True, epoch metrics will be logged every epoch. Else epoch metrics will not be logged
    :param draw_once: If True, draw the plot only at the end of training. Else draw every time metrics are logged
    :param kwargs: Keyword arguments for livelossplot.PlotLosses
    """
    def __init__(self, on_batch=False, batch_step_size=10, on_epoch=True, draw_once=False, **kwargs):

        super().__init__()
        self._kwargs = kwargs

        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.draw_once = draw_once
        self.batch_step_size = batch_step_size

        if on_batch:
            self.on_step_training = self._on_step_training

        if on_epoch:
            self.on_end_epoch = self._on_end_epoch

    def on_start(self, state):
        from livelossplot import PlotLosses
        self.plt = PlotLosses(**self._kwargs)
        self.batch_plt = PlotLosses(**self._kwargs)

    def _on_step_training(self, state):
        self.batch_plt.update(state[torchbearer.METRICS])
        if state[torchbearer.BATCH] % self.batch_step_size == 0 and not self.draw_once:
            with no_print():
                self.batch_plt.draw()

    def _on_end_epoch(self, state):
        self.plt.update(state[torchbearer.METRICS])
        if not self.draw_once:
            with no_print():
                self.plt.draw()

    def on_end(self, state):
        if self.draw_once:
            with no_print():
                self.batch_plt.draw()
                self.plt.draw()
