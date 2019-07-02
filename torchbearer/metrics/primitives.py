"""
    .. autoclass:: BinaryAccuracy()
    .. autoclass:: CategoricalAccuracy(ignore_index=-100)
    .. autoclass:: TopKCategoricalAccuracy(k=5, ignore_index=-100)
    .. autoclass:: MeanSquaredError()
    .. autoclass:: Loss()
    .. autoclass:: Epoch()
"""
import torchbearer
from torchbearer import Metric
from .decorators import default_for_key, running_mean, mean, std, to_dict

import torch

old_super = super


def super(_, obj):
    return old_super(obj.__class__, obj)


@default_for_key('binary_accuracy')
@default_for_key('binary_acc')
@running_mean
@mean
class BinaryAccuracy(Metric):
    """Binary accuracy metric. Uses torch.eq to compare predictions to targets. Decorated with a mean and running_mean.
    Default for key: 'binary_acc'.

    Args:
        pred_key (StateKey): The key in state which holds the predicted values
        target_key (StateKey): The key in state which holds the target values
        threshold (float): value between 0 and 1 to use as a threshold when binarizing predictions and targets
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE, threshold=0.5):
        super(BinaryAccuracy, self).__init__('binary_acc')
        self.pred_key = pred_key
        self.target_key = target_key

        self.threshold = threshold

    def process(self, *args):
        state = args[0]
        y_pred = (state[self.pred_key].float() > self.threshold).long()
        y_true = (state[self.target_key].float() > self.threshold).long()

        return torch.eq(y_pred, y_true).view(-1).float()


@default_for_key('cat_accuracy')
@default_for_key('cat_acc')
@running_mean
@mean
class CategoricalAccuracy(Metric):
    """Categorical accuracy metric. Uses torch.max to determine predictions and compares to targets. Decorated with a
    mean, running_mean and std. Default for key: 'cat_acc'

    Args:
        pred_key (StateKey): The key in state which holds the predicted values
        target_key (StateKey): The key in state which holds the target values
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the metric output.
            See `<https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE, ignore_index=-100):
        super(Metric, self).__init__('acc')
        self.pred_key = pred_key
        self.target_key = target_key

        self.ignore_index = ignore_index

    def process(self, *args):
        state = args[0]
        y_pred = state[self.pred_key]
        y_true = state[self.target_key]

        if len(y_true.shape) == 2:
            _, y_true = torch.max(y_true, 1)

        mask = y_true.eq(self.ignore_index).eq(0)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()


@default_for_key('top_10_accuracy', k=10)
@default_for_key('top_5_accuracy')
@default_for_key('top_10_acc', k=10)
@default_for_key('top_5_acc')
@running_mean
@mean
class TopKCategoricalAccuracy(Metric):
    """Top K Categorical accuracy metric. Uses torch.topk to determine the top k predictions and compares to targets.
    Decorated with a mean, running_mean and std. Default for keys: 'top_5_acc', 'top_10_acc'.

    Args:
        pred_key (StateKey): The key in state which holds the predicted values
        target_key (StateKey): The key in state which holds the target values
        ignore_index (int): Specifies a target value that is ignored and does not contribute to the metric output.
            See `<https://pytorch.org/docs/stable/nn.html#crossentropyloss>`_
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE, k=5, ignore_index=-100):
        super(TopKCategoricalAccuracy, self).__init__('top_' + str(k) + '_acc')
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


@default_for_key('mse')
@running_mean
@mean
class MeanSquaredError(Metric):
    """Mean squared error metric. Computes the pixelwise squared error which is then averaged with decorators.
    Decorated with a mean and running_mean. Default for key: 'mse'.

    Args:
        pred_key (StateKey): The key in state which holds the predicted values
        target_key (StateKey): The key in state which holds the target values
    """

    def __init__(self, pred_key=torchbearer.Y_PRED, target_key=torchbearer.Y_TRUE):
        super(MeanSquaredError, self).__init__('mse')
        self.pred_key = pred_key
        self.target_key = target_key

    def process(self, *args):
        state = args[0]
        y_pred = state[self.pred_key]
        y_true = state[self.target_key]
        return torch.pow(y_pred - y_true.view_as(y_pred), 2).data


@default_for_key('loss')
@running_mean
@mean
class Loss(Metric):
    """Simply returns the 'loss' value from the model state. Decorated with a mean, running_mean and std. Default for
    key: 'loss'.

    State Requirements:
        - :attr:`torchbearer.state.LOSS`: This key should map to the loss for the current batch
    """

    def __init__(self):
        super(Loss, self).__init__('loss')

    def process(self, *args):
        state = args[0]
        return state[torchbearer.LOSS]


@default_for_key('epoch')
@to_dict
class Epoch(Metric):
    """Returns the 'epoch' from the model state. Default for key: 'epoch'.

    State Requirements:
        - :attr:`torchbearer.state.EPOCH`: This key should map to the number of the current epoch
    """

    def __init__(self):
        super(Epoch, self).__init__('epoch')

    def process_final(self, *args):
        state = args[0]
        return self._process(state)

    def process(self, *args):
        state = args[0]
        return self._process(state)

    def _process(self, state):
        return state[torchbearer.EPOCH]
