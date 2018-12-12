import torchbearer as tb
import torchbearer.callbacks as c


class LiveLossPlot(c.Callback):
    def __init__(self, on_batch=False, batch_step_size=10, on_epoch=True, draw_once=False, **kwargs):
        super().__init__()
        from livelossplot import PlotLosses
        self.plt = PlotLosses(**kwargs)
        self.batch_plt = PlotLosses(**kwargs)

        self.on_batch = on_batch
        self.on_epoch = on_epoch
        self.draw_once = draw_once
        self.batch_step_size = batch_step_size

        if on_batch:
            self.on_step_training = self._on_step_training

        if on_epoch:
            self.on_end_epoch = self._on_end_epoch

    def _on_step_training(self, state):
        self.batch_plt.update(state[tb.METRICS])
        if state[tb.BATCH] % self.batch_step_size == 0 and not self.draw_once:
            self.batch_plt.draw()

    def _on_end_epoch(self, state):
        self.plt.update(state[tb.METRICS])
        if not self.draw_once:
            self.plt.draw()

    def on_end(self, state):
        if self.draw_once:
            self.plt.draw()
