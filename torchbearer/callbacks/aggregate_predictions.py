import torchbearer
from torchbearer.callbacks import Callback
import torch
import warnings


class AggregatePredictions(Callback):
    def __init__(self):
        super(AggregatePredictions, self).__init__()
        self.predictions_list = []

    def on_step_validation(self, state):
        super(AggregatePredictions, self).on_step_validation(state)
        self.predictions_list.append(state[torchbearer.Y_PRED])

    def on_end_validation(self, state):
        try:
            if len(self.predictions_list) == 1 and type(self.predictions_list[0]) is torch.Tensor:
                state[torchbearer.FINAL_PREDICTIONS] = self.predictions_list[0]
            else:
                state[torchbearer.FINAL_PREDICTIONS] = torch.cat(self.predictions_list, 0)
        except:
            warnings.warn('Failed to format predictions as tensor, returning as list.')
            state[torchbearer.FINAL_PREDICTIONS] = self.predictions_list

    def on_end_epoch(self, state):
        super(AggregatePredictions, self).on_end_epoch(state)
        self.predictions_list = []
