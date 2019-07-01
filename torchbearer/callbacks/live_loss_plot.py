import sys
import os

import torchbearer
from torchbearer.callbacks import Callback


class no_print:
    def __init__(self):
        pass

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

    Example: ::

        >>> import torch.nn
        >>> from torchbearer import Trial
        >>> from torchbearer.callbacks import LiveLossPlot

        # Example Trial which clips all model gradients norms at 2 under the L1 norm.
        >>> model = torch.nn.Linear(1,1)
        >>> live_loss_plot = LiveLossPlot()
        >>> trial = Trial(model, callbacks=[live_loss_plot], metrics=['acc'])

    Args:
        on_batch (bool): If True, batch metrics will be logged. Else batch metrics will not be logged
        batch_step_size (int): The number of batches between logging metrics
        on_epoch (bool): If True, epoch metrics will be logged every epoch. Else epoch metrics will not be logged
        draw_once (bool): If True, draw the plot only at the end of training. Else draw every time metrics are logged
        kwargs: Keyword arguments for livelossplot.PlotLosses

    State Requirements:
        - :attr:`torchbearer.state.METRICS`: Metrics should be a dict containing the metrics to be plotted
        - :attr:`torchbearer.state.BATCH`: Batch should be the current batch or iteration number in the epoch
    """
    def __init__(self, on_batch=False, batch_step_size=10, on_epoch=True, draw_once=False, **kwargs):
        super(LiveLossPlot, self).__init__()
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
        self.batch_plt.update({k: state[torchbearer.METRICS][k] for k in state[torchbearer.METRICS]})
        if state[torchbearer.BATCH] % self.batch_step_size == 0 and not self.draw_once:
            with no_print():
                self.batch_plt.draw()

    def _on_end_epoch(self, state):
        self.plt.update({k: state[torchbearer.METRICS][k] for k in state[torchbearer.METRICS]})
        if not self.draw_once:
            with no_print():
                self.plt.draw()

    def on_end(self, state):
        if self.draw_once:
            with no_print():
                self.batch_plt.draw()
                self.plt.draw()
