from unittest import TestCase

from mock import patch, MagicMock

import torchbearer
from torchbearer.callbacks import LiveLossPlot


class TestLiveLossPlot(TestCase):
    @patch('livelossplot.PlotLosses')
    def test_on_start(self, llp_mock):
        llp = LiveLossPlot(True, 1, True, False)
        llp.on_start({})
        self.assertTrue(llp_mock.call_count == 2)

    def test_on_batch(self):
        llp = LiveLossPlot(True, 1, False, False)
        llp.batch_plt = MagicMock()
        llp.plt = MagicMock()
        state = {torchbearer.BATCH: 1, torchbearer.METRICS: {'test': 1}}
        llp.on_step_training(state)
        llp.on_step_training(state)

        self.assertTrue(llp.batch_plt.update.call_count == 2)
        self.assertTrue(llp.plt.update.call_count == 0)

    def test_on_batch_steps(self):
        llp = LiveLossPlot(True, 2, False, False)
        llp.batch_plt = MagicMock()
        llp.plt = MagicMock()
        state = {torchbearer.BATCH: 1, torchbearer.METRICS: {'test': 1}}
        llp.on_step_training(state)
        state = {torchbearer.BATCH: 2, torchbearer.METRICS: {'test': 1}}
        llp.on_step_training(state)
        state = {torchbearer.BATCH: 3, torchbearer.METRICS: {'test': 1}}
        llp.on_step_training(state)
        state = {torchbearer.BATCH: 4, torchbearer.METRICS: {'test': 1}}
        llp.on_step_training(state)

        self.assertTrue(llp.batch_plt.draw.call_count == 2)
        self.assertTrue(llp.plt.draw.call_count == 0)

    def test_not_on_batch(self):
        llp = LiveLossPlot(False, 10, False, False)
        llp.batch_plt = MagicMock()
        llp.plt = MagicMock()
        state = {torchbearer.BATCH: 1, torchbearer.METRICS: {'test': 1}}
        llp.on_step_training(state)
        llp.on_step_training(state)

        self.assertTrue(llp.batch_plt.update.call_count == 0)

    def test_on_epoch(self):
        llp = LiveLossPlot(False, 10, True, False)
        llp.batch_plt = MagicMock()
        llp.plt = MagicMock()
        state = {torchbearer.BATCH: 1, torchbearer.METRICS: {'test': 1}}
        llp.on_end_epoch(state)
        llp.on_end_epoch(state)

        self.assertTrue(llp.batch_plt.update.call_count == 0)
        self.assertTrue(llp.plt.update.call_count == 2)

    def test_draw_once(self):
        llp = LiveLossPlot(True, 1, True, True)
        llp.batch_plt = MagicMock()
        llp.plt = MagicMock()
        state = {torchbearer.BATCH: 1, torchbearer.METRICS: {'test': 1}}
        llp.on_end_epoch(state)
        llp.on_end_epoch(state)
        llp.on_end(state)

        self.assertTrue(llp.plt.draw.call_count == 1)
        self.assertTrue(llp.batch_plt.draw.call_count == 1)
