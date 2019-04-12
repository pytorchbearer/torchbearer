"""
Metric wrappers are classes which wrap instances of :class:`.Metric` or, in the case of :class:`EpochLambda` and
:class:`BatchLambda`, functions. Typically, these should **not** be used directly (although this is entirely possible),
but via the :mod:`decorator API<.metrics.decorators>`.
"""
import torchbearer
from torchbearer.bases import Metric
from .metrics import AdvancedMetric

import torch


class ToDict(AdvancedMetric):
    """The :class:`ToDict` class is an :class:`.AdvancedMetric` which will put output from the inner :class:`.Metric` in
    a dict (mapping metric name to value) before returning. When in `eval` mode, 'val\_' will be prepended to the metric
    name.

    Example: ::

        >>> from torchbearer import metrics

        >>> @metrics.lambda_metric('my_metric')
        ... def my_metric(y_pred, y_true):
        ...     return y_pred + y_true
        ...
        >>> metric = metrics.ToDict(my_metric().build())
        >>> metric.process({'y_pred': 4, 'y_true': 5})
        {'my_metric': 9}
        >>> metric.eval()
        >>> metric.process({'y_pred': 4, 'y_true': 5})
        {'val_my_metric': 9}

    Args:
        metric (Metric): The :class:`.Metric` instance to *wrap*.
    """

    def __init__(self, metric):
        super(ToDict, self).__init__(metric.name)

        self.eval_flag = 'val'
        self.metric = metric

    def process_train(self, *args):
        val = self.metric.process(*args)
        if val is not None:
            return {self.metric.name: val}

    def process_validate(self, *args):
        val = self.metric.process(*args)
        if val is not None:
            return {self.eval_flag + '_' + self.metric.name: val}

    def process_final_train(self, *args):
        val = self.metric.process_final(*args)
        if val is not None:
            return {self.metric.name: val}

    def process_final_validate(self, *args):
        val = self.metric.process_final(*args)
        if val is not None:
            return {self.eval_flag + '_' + self.metric.name: val}

    def eval(self, data_key=None):
        super(ToDict, self).eval(data_key=data_key)
        if data_key == torchbearer.TEST_DATA:
            self.eval_flag = 'test'
        elif data_key == torchbearer.TRAIN_DATA:
            self.eval_flag = 'train'
        else:
            self.eval_flag = 'val'
        self.metric.eval(data_key=data_key)

    def train(self):
        super(ToDict, self).train()
        self.metric.train()

    def reset(self, state):
        super(ToDict, self).reset(state)
        self.metric.reset(state)


class BatchLambda(Metric):
    """A metric which returns the output of the given function on each batch.

    Args:
        name (str): The name of the metric.
        metric_function (func): A metric function('y_pred', 'y_true') to wrap.
    """

    def __init__(self, name, metric_function):
        super(BatchLambda, self).__init__(name)
        self._metric_function = metric_function

    def process(self, *args):
        """Return the output of the wrapped function.

        Args:
            args: The :class:`.torchbearer.Trial` state.

        Returns:
            The value of the metric function('y_pred', 'y_true').
        """
        state = args[0]
        return self._metric_function(state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE])


class EpochLambda(AdvancedMetric):
    """A metric wrapper which computes the given function for concatenated values of 'y_true' and 'y_pred' each epoch.
    Can be used as a running metric which computes the function for batches of outputs with a given step size during
    training.

    Args:
        name (str): The name of the metric.
        metric_function (func): The function('y_pred', 'y_true') to use as the metric.
        running (bool): True if this should act as a running metric.
        step_size (int): Step size to use between calls if running=True.
    """

    def __init__(self, name, metric_function, running=True, step_size=50):
        super(EpochLambda, self).__init__(name)
        self._step = metric_function
        self._final = metric_function
        self._step_size = step_size
        self._result = 0.0

        if not running:
            self._step = lambda y_pred, y_true: None

    def process_train(self, *args):
        """Concatenate the 'y_true' and 'y_pred' from the state along the 0 dimension, this must be the batch dimension.
        If this is a running metric, evaluates the function every number of steps.

        Args:
            args: The :class:`.torchbearer.Trial` state.

        Returns:
            The current running result.
        """
        state = args[0]
        if (self._y_pred is None) or (self._y_true is None):
            self._y_true = state[torchbearer.Y_TRUE]
            self._y_pred = state[torchbearer.Y_PRED]
        else:
            self._y_true = torch.cat((self._y_true, state[torchbearer.Y_TRUE]), dim=0)
            self._y_pred = torch.cat((self._y_pred, state[torchbearer.Y_PRED]), dim=0)
        if state[torchbearer.BATCH] % self._step_size == 0:
            self._result = self._step(self._y_pred, self._y_true)
        return self._result

    def process_final_train(self, *args):
        """Evaluate the function with the aggregated outputs.

        Returns:
            The result of the function.
        """
        return self._final(self._y_pred, self._y_true)

    def process_validate(self, *args):
        """During validation, just concatenate 'y_true' and y_pred'.

        Args:
            args: The :class:`.torchbearer.Trial` state.
        """
        state = args[0]
        if (self._y_pred is None) or (self._y_true is None):
            self._y_true = state[torchbearer.Y_TRUE]
            self._y_pred = state[torchbearer.Y_PRED]
        else:
            self._y_true = torch.cat((self._y_true, state[torchbearer.Y_TRUE]), dim=0)
            self._y_pred = torch.cat((self._y_pred, state[torchbearer.Y_PRED]), dim=0)

    def process_final_validate(self, *args):
        """Evaluate the function with the aggregated outputs.

        Returns:
            The result of the function.
        """
        return self._final(self._y_pred, self._y_true)

    def reset(self, state):
        """Reset the 'y_true' and 'y_pred' caches.

        Args:
            state (dict): The :class:`.torchbearer.Trial` state.
        """
        super(EpochLambda, self).reset(state)
        self._y_true = None
        self._y_pred = None
