from framework.metrics.metrics import BasicMetric
from framework.metrics.wrappers import Lambda

import torch


def _categorical(y_true, y_pred):
    _, y_pred = torch.max(y_pred, 1)
    return (y_pred == y_true).float()


categorical = accuracy = Lambda('acc', _categorical)


class Loss(BasicMetric):
    def __init__(self):
        super().__init__('loss')

    def train(self, state):
        return self._process(state)

    def validate(self, state):
        return self._process(state)

    def _process(self, state):
        return state['loss']


loss = Loss()


class Epoch(BasicMetric):
    def __init__(self):
        super().__init__('epoch')

    def train(self, state):
        return self._process(state)

    def validate(self, state):
        return self._process(state)

    def _process(self, state):
        return state['epoch']


epoch = Epoch()
