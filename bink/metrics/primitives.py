from bink import metrics

import torch


def _categorical(y_true, y_pred):
    _, y_pred = torch.max(y_pred, 1)
    return (y_pred == y_true).float()


categorical_primitive = metrics.BatchLambda('acc', _categorical)


class Loss(metrics.BasicMetric):
    def __init__(self):
        super().__init__('loss')

    def train(self, state):
        return self._process(state)

    def validate(self, state):
        return self._process(state)

    def _process(self, state):
        return state['loss']


loss_primitive = Loss()


class Epoch(metrics.Metric):
    def __init__(self):
        super().__init__()
        self._name = 'epoch'

    def final_train_dict(self, state):
        return {self._name: self._process(state)}

    def final_validate_dict(self, state):
        return {self._name: self._process(state)}

    def train_dict(self, state):
        return {self._name: self._process(state)}

    def validate_dict(self, state):
        return {self._name: self._process(state)}

    def _process(self, state):
        return state['epoch']


epoch_primitive = Epoch()
