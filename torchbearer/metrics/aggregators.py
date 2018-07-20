from torchbearer import metrics
from abc import ABCMeta, abstractmethod
from collections import deque

import torch


class RunningMetric(metrics.AdvancedMetric):
    """A metric which aggregates batches of results and presents a method to periodically process these into a value.

    .. note::

        This class only provides output during training.

    """
    __metaclass__ = ABCMeta

    def __init__(self, name, batch_size=50, step_size=10):
        """Initialise the deque of results.

        :param name: The name of the metric. Will be prepended with 'running_'.
        :type name: str
        :param batch_size: The size of the deque to store of previous results.
        :type batch_size: int
        :param step_size: The number of iterations between aggregations.
        :type step_size: int

        """
        super().__init__(name)
        self._batch_size = batch_size
        self._step_size = step_size
        self._cache = deque()
        self._result = {}

    @abstractmethod
    def _process_train(self, *args):
        """Process the metric for a single train step.

        :param state: The current model state.
        :type state: dict
        :return: The metric value.

        """
        ...

    @abstractmethod
    def _step(self, cache):
        """Aggregate the cache to produce a single metric value.

        :param cache: The current stored metric cache.
        :type cache: list
        :return: The new metric value.

        """
        ...

    def process_train(self, *args):
        """Add the current metric value to the cache and call '_step' is needed.

        :param state: The current model state.
        :type state: dict
        :return: The current metric value.

        """
        self._cache.append(self._process_train(*args))
        if len(self._cache) > self._batch_size:
            self._cache.popleft()
        if self._i % self._step_size == 0:
            self._result = self._step(list(self._cache))
        self._i += 1
        return self._result

    def reset(self, state):
        """Reset the step counter. Does not clear the cache.

        :param state: The current model state.
        :type state: dict

        """
        self._i = 0


class RunningMean(RunningMetric):
    """A running metric wrapper which outputs the mean of a sequence of observations.
    """

    def __init__(self, name, batch_size=50, step_size=10):
        """Wrap the given metric in initialise the parent :class:`RunningMetric`.

        :param metric: The metric to wrap.
        :type metric: Metric
        :param batch_size: The size of the deque to store of previous results.
        :type batch_size: int
        :param step_size: The number of iterations between aggregations.
        :type step_size: int
        """
        super().__init__(name, batch_size=batch_size, step_size=step_size)

    def _process_train(self, data):
        return data.mean().item()

    def _step(self, cache):
        return sum(cache) / float(len(cache))


class Std(metrics.Metric):
    """Metric wrapper which calculates the standard deviation of process outputs between calls to reset.
    """

    def __init__(self, name):
        super(Std, self).__init__(name)

    def process(self, data):
        """Process the wrapped metric and compute values required for the std.

        :param state: The model state.
        :type state: dict

        """
        self._sum += data.sum().item()
        self._sum_sq += data.pow(2).sum().item()

        if data.size() == torch.Size([]):
            self._count += 1
        else:
            self._count += data.size(0)

    def process_final(self, data):
        """Compute and return the final standard deviation.

        :param state: The model state.
        :type state: dict
        :return: The standard deviation of each observation since the last reset call.

        """
        mean = self._sum / self._count
        mean = mean ** 2
        return ((self._sum_sq / self._count) - mean) ** 0.5

    def reset(self, state):
        """Reset the statistics to compute the next deviation.

        :param state: The model state.
        :type state: dict

        """
        super().reset(state)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


class Mean(metrics.Metric):
    """Metric wrapper which calculates the mean value of a series of observations between reset calls.
    """

    def __init__(self, name):
        super(Mean, self).__init__(name)

    def process(self, data):
        """Compute the metric value and add it to the rolling sum.

        :param state: The model state.
        :type state: dict

        """
        self._sum += data.sum().item()

        if data.size() == torch.Size([]):
            self._count += 1
        else:
            self._count += data.size(0)

    def process_final(self, data):
        """Compute and return the mean of all metric values since the last call to reset.

        :param state: The model state.
        :type state: dict
        :return: The mean of the metric values since the last call to reset.

        """
        return self._sum / self._count

    def reset(self, state):
        """Reset the running count and total.

        :param state: The model state.
        :type state: dict

        """
        super().reset(state)
        self._sum = 0.0
        self._count = 0
