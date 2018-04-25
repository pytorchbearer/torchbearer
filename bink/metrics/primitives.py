from bink import metrics

import torch


class CategoricalAccuracy(metrics.BatchLambda):
    def __init__(self):
        super().__init__('acc', self._categorical)

    def _categorical(self, y_true, y_pred):
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()

categorical_accuracy_primitive = CategoricalAccuracy()


class Loss(metrics.Metric):
    def __init__(self):
        super().__init__('loss')

    def process(self, state):
        return state['loss']


loss_primitive = Loss()


class Epoch(metrics.Metric):
    def __init__(self):
        super().__init__('epoch')

    def process_final(self, state):
        return self._process(state)

    def process(self, state):
        return self._process(state)

    def _process(self, state):
        return state['epoch']


epoch_primitive = Epoch()
