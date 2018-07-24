"""
Metric wrappers are classes which wrap instances of :class:`.Metric` or, in the case of :class:`EpochLambda` and
:class:`BatchLambda`, functions. Typically, these should **not** be used directly (although this is entirely possible),
but via the :mod:`decorator API<.metrics.decorators>`.
"""
import torchbearer
from torchbearer import metrics

import torch


class ToDict(metrics.AdvancedMetric):
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

    :param metric: The :class:`.Metric` instance to *wrap*.
    :type metric: metrics.Metric
    """

    def __init__(self, metric):
        super(ToDict, self).__init__(metric.name)

        self.metric = metric

    def process_train(self, *args):
        val = self.metric.process(*args)
        if val is not None:
            return {self.metric.name: val}

    def process_validate(self, *args):
        val = self.metric.process(*args)
        if val is not None:
            return {'val_' + self.metric.name: val}

    def process_final_train(self, *args):
        val = self.metric.process_final(*args)
        if val is not None:
            return {self.metric.name: val}

    def process_final_validate(self, *args):
        val = self.metric.process_final(*args)
        if val is not None:
            return {'val_' + self.metric.name: val}

    def eval(self):
        super().eval()
        self.metric.eval()

    def train(self):
        super().train()
        self.metric.train()

    def reset(self, state):
        super().reset(state)
        self.metric.reset(state)


class BatchLambda(metrics.Metric):
    """A metric which returns the output of the given function on each batch.

    :param name: The name of the metric.
    :type name: str
    :param metric_function: A metric function('y_pred', 'y_true') to wrap.
    """

    def __init__(self, name, metric_function):
        super(BatchLambda, self).__init__(name)
        self._metric_function = metric_function

    def process(self, state):
        """Return the output of the wrapped function.

        :param state: The :class:`.torchbearer.Model` state.
        :type state: dict
        :return: The value of the metric function('y_pred', 'y_true').

        """
        return self._metric_function(state[torchbearer.Y_PRED], state[torchbearer.Y_TRUE])


class EpochLambda(metrics.AdvancedMetric):
    """A metric wrapper which computes the given function for concatenated values of 'y_true' and 'y_pred' each epoch.
    Can be used as a running metric which computes the function for batches of outputs with a given step size during
    training.

    :param name: The name of the metric.
    :type name: str
    :param metric_function: The function('y_pred', 'y_true') to use as the metric.
    :param running: True if this should act as a running metric.
    :type running: bool
    :param step_size: Step size to use between calls if running=True.
    :type step_size: int
    """

    def __init__(self, name, metric_function, running=True, step_size=50):
        super(EpochLambda, self).__init__(name)
        self._step = metric_function
        self._final = metric_function
        self._step_size = step_size
        self._result = 0.0

        if not running:
            self._step = lambda y_pred, y_true: ...

    def process_train(self, state):
        """Concatenate the 'y_true' and 'y_pred' from the state along the 0 dimension. If this is a running metric,
        evaluates the function every number of steps.

        :param state: The :class:`.torchbearer.Model` state.
        :type state: dict
        :return: The current running result.

        """
        self._y_true = torch.cat((self._y_true, state[torchbearer.Y_TRUE]), dim=0)
        self._y_pred = torch.cat((self._y_pred, state[torchbearer.Y_PRED].float()), dim=0)
        if state[torchbearer.BATCH] % self._step_size == 0:
            self._result = self._step(self._y_pred, self._y_true)
        return self._result

    def process_final_train(self, state):
        """Evaluate the function with the aggregated outputs.

        :param state: The :class:`.torchbearer.Model` state.
        :type state: dict
        :return: The result of the function.

        """
        return self._final(self._y_pred, self._y_true)

    def process_validate(self, state):
        """During validation, just concatenate 'y_true' and y_pred'.

        :param state: The :class:`.torchbearer.Model` state.
        :type state: dict

        """
        self._y_true = torch.cat((self._y_true, state[torchbearer.Y_TRUE]), dim=0)
        self._y_pred = torch.cat((self._y_pred, state[torchbearer.Y_PRED].to(self._y_pred.dtype)), dim=0)

    def process_final_validate(self, state):
        """Evaluate the function with the aggregated outputs.

        :param state: The :class:`.torchbearer.Model` state.
        :type state: dict
        :return: The result of the function.

        """
        return self._final(self._y_pred, self._y_true)

    def reset(self, state):
        """Reset the 'y_true' and 'y_pred' caches.

        :param state: The :class:`.torchbearer.Model` state.
        :type state: dict

        """
        super().reset(state)
        self._y_true = torch.zeros(0).long()
        self._y_pred = torch.zeros(0, 0)

        self._y_true = self._y_true.to(state[torchbearer.DEVICE])
        self._y_pred = self._y_pred.to(state[torchbearer.DEVICE], state[torchbearer.DATA_TYPE])
