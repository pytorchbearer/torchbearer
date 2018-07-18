import torchbearer
from torchbearer.callbacks import Callback
import torch


class AggregatePredictions(Callback):
    def __init__(self):
        super(AggregatePredictions, self).__init__()
        self.predictions_list = []

    def on_step_validation(self, state):
        super().on_step_validation(state)
        self.predictions_list.append(state[torchbearer.Y_PRED])

    def on_end_validation(self, state):
        state[torchbearer.FINAL_PREDICTIONS] = torch.cat(self.predictions_list, 0)
