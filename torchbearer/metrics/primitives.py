import torchbearer
from torchbearer import metrics

import torch


class CategoricalAccuracy(metrics.BatchLambda):
    """Categorical accuracy metric. Uses torch.max to determine predictions and compares to targets.
    """
    def __init__(self):
        super().__init__('acc', self._categorical)

    def _categorical(self, y_pred, y_true):
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()


categorical_accuracy_primitive = CategoricalAccuracy()


class Loss(metrics.Metric):
    """Simply returns the 'loss' value from the model state.
    """
    def __init__(self):
        super().__init__('loss')

    def process(self, state):
        return state[torchbearer.LOSS]


loss_primitive = Loss()


class Epoch(metrics.Metric):
    """Returns the 'epoch' from the model state.
    """
    def __init__(self):
        super().__init__('epoch')

    def process_final(self, state):
        return self._process(state)

    def process(self, state):
        return self._process(state)

    def _process(self, state):
        return state[torchbearer.EPOCH]


epoch_primitive = Epoch()
