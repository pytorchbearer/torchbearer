from bink import metrics

import torch


class CategoricalAccuracy(metrics.BatchLambda):
    def __init__(self):
        def _categorical(y_true, y_pred):
            _, y_pred = torch.max(y_pred, 1)
            return (y_pred == y_true).float()
        super().__init__('acc', _categorical)


categorical_accuracy_primitive = CategoricalAccuracy()


class Loss(metrics.Metric):
    def __init__(self):
        super().__init__('loss')

    def evaluate(self, state):
        return state['loss']


loss_primitive = Loss()


class Epoch(metrics.Metric):
    def __init__(self):
        super().__init__('epoch')

    def evaluate_final(self, state):
        return self._process(state)

    def evaluate(self, state):
        return self._process(state)

    def evaluate_final_dict(self, state):
        return {self.name: self.evaluate_final(state)}

    def evaluate_dict(self, state):
        return {self.name: self.evaluate(state)}

    def _process(self, state):
        return state['epoch']


epoch_primitive = Epoch()
