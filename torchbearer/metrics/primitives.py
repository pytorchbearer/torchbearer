"""
    .. autoclass:: BinaryAccuracy()
    .. autoclass:: CategoricalAccuracy(ignore_index=-100)
    .. autoclass:: TopKCategoricalAccuracy(k=5, ignore_index=-100)
    .. autoclass:: MeanSquaredError()
    .. autoclass:: Loss()
    .. autoclass:: Epoch()
"""

import torchbearer
from torchbearer import metrics

import torch


@metrics.default_for_key('binary_acc')
@metrics.running_mean
@metrics.mean
class BinaryAccuracy(metrics.Metric):
    """Binary accuracy metric. Uses torch.eq to compare predictions to targets. Decorated with a mean and running_mean.
    Default for key: 'binary_acc'.

    :param pred_key: The key in state which holds the predicted values
    :type pred_key: torchbearer.StateKey
    :param target_key: The key in state which holds the target values
    :type target_key: torchbearer.StateKey
    :param threshold: value between 0 and 1 to use as a threshold when binarizing predictions and targets
    :type threshold: float
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE, threshold=0.5):
        super().__init__('binary_acc')
        self.pred_key = pred_key
        self.target_key = target_key

        self.threshold = threshold

    def process(self, *args):
        state = args[0]
        y_pred = (state[self.pred_key].float() > self.threshold).long()
        y_true = (state[self.target_key].float() > self.threshold).long()

        return torch.eq(y_pred, y_true).view(-1).float()


@metrics.default_for_key('cat_acc')
@metrics.running_mean
@metrics.std
@metrics.mean
class CategoricalAccuracy(metrics.Metric):
    """Categorical accuracy metric. Uses torch.max to determine predictions and compares to targets. Decorated with a
    mean, running_mean and std. Default for key: 'cat_acc'

    :param pred_key: The key in state which holds the predicted values
    :type pred_key: torchbearer.StateKey
    :param target_key: The key in state which holds the target values
    :type target_key: torchbearer.StateKey
    :param ignore_index: Specifies a target value that is ignored and does not contribute to the metric output. See `<https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_
    :type ignore_index: int
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE, ignore_index=-100):
        super().__init__('acc')
        self.pred_key = pred_key
        self.target_key = target_key

        self.ignore_index = ignore_index

    def process(self, *args):
        state = args[0]
        y_pred = state[self.pred_key]
        y_true = state[self.target_key]
        mask = y_true.eq(self.ignore_index).eq(0)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()


@metrics.default_for_key('top_10_acc', k=10)
@metrics.default_for_key('top_5_acc')
@metrics.running_mean
@metrics.std
@metrics.mean
class TopKCategoricalAccuracy(metrics.Metric):
    """Top K Categorical accuracy metric. Uses torch.topk to determine the top k predictions and compares to targets.
    Decorated with a mean, running_mean and std. Default for keys: 'top_5_acc', 'top_10_acc'.

    :param pred_key: The key in state which holds the predicted values
    :type pred_key: torchbearer.StateKey
    :param target_key: The key in state which holds the target values
    :type target_key: torchbearer.StateKey
    :param ignore_index: Specifies a target value that is ignored and does not contribute to the metric output. See `<https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_
    :type ignore_index: int
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE, k=5, ignore_index=-100):
        super().__init__('top_' + str(k) + '_acc')
        self.pred_key = pred_key
        self.target_key = target_key

        self.k = k
        self.ignore_index = ignore_index

    def process(self, *args):
        state = args[0]
        y_pred = state[self.pred_key]
        y_true = state[self.target_key]
        mask = y_true.eq(self.ignore_index).eq(0)
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        sorted_indices = torch.topk(y_pred, self.k, dim=1)[1]
        expanded_y = y_true.view(-1, 1).expand(-1, self.k)
        return torch.sum(torch.eq(sorted_indices, expanded_y), dim=1).float()


@metrics.default_for_key('mse')
@metrics.running_mean
@metrics.mean
class MeanSquaredError(metrics.Metric):
    """Mean squared error metric. Computes the pixelwise squared error which is then averaged with decorators.
    Decorated with a mean and running_mean. Default for key: 'mse'.

    :param pred_key: The key in state which holds the predicted values
    :type pred_key: torchbearer.StateKey
    :param target_key: The key in state which holds the target values
    :type target_key: torchbearer.StateKey
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE):
        super().__init__('mse')
        self.pred_key = pred_key
        self.target_key = target_key

    def process(self, *args):
        state = args[0]
        y_pred = state[self.pred_key]
        y_true = state[self.target_key]
        return torch.pow(y_pred - y_true.view_as(y_pred), 2)


@metrics.default_for_key('loss')
@metrics.running_mean
@metrics.std
@metrics.mean
class Loss(metrics.Metric):
    """Simply returns the 'loss' value from the model state. Decorated with a mean, running_mean and std. Default for
    key: 'loss'.
    """

    def __init__(self):
        super().__init__('loss')

    def process(self, *args):
        state = args[0]
        return state[torchbearer.LOSS]


@metrics.default_for_key('epoch')
@metrics.to_dict
class Epoch(metrics.Metric):
    """Returns the 'epoch' from the model state. Default for key: 'epoch'.
    """

    def __init__(self):
        super().__init__('epoch')

    def process_final(self, *args):
        state = args[0]
        return self._process(state)

    def process(self, *args):
        state = args[0]
        return self._process(state)

    def _process(self, state):
        return state[torchbearer.EPOCH]
