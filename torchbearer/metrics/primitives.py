"""
Base metrics are the base classes which represent the metrics supplied with torchbearer. The all use the
:func:`.default_for_key` decorator so that they can be accessed in the call to :class:`.torchbearer.Model` via the
following strings:

- '`acc`' or '`accuracy`': The :class:`.CategoricalAccuracy` metric
- '`loss`': The :class:`.Loss` metric
- '`epoch`': The :class:`.Epoch` metric
- '`roc_auc`' or '`roc_auc_score`': The :class:`.RocAucScore` metric
"""
import torchbearer
from torchbearer import metrics

import torch


class CategoricalAccuracy(metrics.Metric):
    """Categorical accuracy metric. Uses torch.max to determine predictions and compares to targets.

    :param ignore_index: Specifies a target value that is ignored and does not contribute to the metric output. See `https://pytorch.org/docs/stable/nn.html#crossentropyloss`_
    :type ignore_index: int
    """

    def __init__(self, ignore_index=-100):
        super(CategoricalAccuracy, self).__init__('acc')

        self.ignore_index = ignore_index

    def process(self, *args):
        state = args[0]
        y_pred = state[torchbearer.Y_PRED]
        y_true = state[torchbearer.Y_TRUE]
        mask = y_true.eq(self.ignore_index).eq(0)
        y_pred = y_pred[mask]
        y_true = y_true[mask]
        _, y_pred = torch.max(y_pred, 1)
        return (y_pred == y_true).float()


@metrics.default_for_key('acc')
@metrics.default_for_key('accuracy')
@metrics.running_mean
@metrics.std
@metrics.mean
class CategoricalAccuracyFactory(metrics.MetricFactory):
    """Categorical accuracy metric factory. Essentially a :class:`.CategoricalAccuracy` with running mean, mean and std.

    :param ignore_index: Specifies a target value that is ignored and does not contribute to the metric output. See `https://pytorch.org/docs/stable/nn.html#crossentropyloss`_
    :type ignore_index: int
    """

    def __init__(self, ignore_index=-100):
        self.ignore_index = ignore_index

    def build(self):
        return CategoricalAccuracy(ignore_index=self.ignore_index)


class Loss(metrics.Metric):
    """Simply returns the 'loss' value from the model state.
    """

    def __init__(self):
        super().__init__('loss')

    def process(self, *args):
        state = args[0]
        return state[torchbearer.LOSS]


@metrics.default_for_key('loss')
@metrics.running_mean
@metrics.std
@metrics.mean
class LossFactory(metrics.MetricFactory):
    def build(self):
        return Loss()


class Epoch(metrics.Metric):
    """Returns the 'epoch' from the model state.
    """

    def __init__(self):
        super().__init__('epoch')

    def process_final(self, *args):
        state = args[0]
        return Epoch._process(state)

    def process(self, *args):
        state = args[0]
        return Epoch._process(state)

    @staticmethod
    def _process(state):
        return state[torchbearer.EPOCH]


@metrics.default_for_key('epoch')
@metrics.to_dict
class EpochFactory(metrics.MetricFactory):
    def build(self):
        return Epoch()
