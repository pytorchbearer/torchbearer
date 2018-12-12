"""
Aggregators are a special kind of :class:`.Metric` which takes as input, the output from a previous metric or metrics.
As a result, via a :class:`.MetricTree`, a series of aggregators can collect statistics such as Mean or Standard
Deviation without needing to compute the underlying metric multiple times. This can, however, make the aggregators
complex to use. It is therefore typically better to use the :mod:`decorator API<.metrics.decorators>`.
"""
from torchbearer import metrics
from abc import ABCMeta, abstractmethod
from collections import deque

import torch


class RunningMetric(metrics.AdvancedMetric):
    """A metric which aggregates batches of results and presents a method to periodically process these into a value.

    .. note::

       Running metrics only provide output during training.

    :param name: The name of the metric.
    :type name: str
    :param batch_size: The size of the deque to store of previous results.
    :type batch_size: int
    :param step_size: The number of iterations between aggregations.
    :type step_size: int
    """
    __metaclass__ = ABCMeta

    def __init__(self, name, batch_size=50, step_size=10):
        super().__init__(name)
        self._batch_size = batch_size
        self._step_size = step_size
        self._cache = deque()
        self._result = {}

    @abstractmethod
    def _process_train(self, *args):
        """Process the metric for a single train step.

        :param args: The output of some :class:`.Metric`
        :return: The metric value.

        """
        pass

    @abstractmethod
    def _step(self, cache):
        """Aggregate the cache to produce a single metric value.

        :param cache: The current stored metric cache.
        :type cache: list
        :return: The new metric value.

        """
        pass

    def process_train(self, *args):
        """Add the current metric value to the cache and call '_step' is needed.

        :param args: The output of some :class:`.Metric`
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
    """A :class:`RunningMetric` which outputs the running mean of its input tensors over the course of an epoch.

    :param name: The name of this running mean.
    :type name: str
    :param batch_size: The size of the deque to store of previous results.
    :type batch_size: int
    :param step_size: The number of iterations between aggregations.
    :type step_size: int
    """

    def __init__(self, name, batch_size=50, step_size=10):
        super().__init__(name, batch_size=batch_size, step_size=step_size)

    def _process_train(self, *args):
        data = args[0]
        return data.mean().item()

    def _step(self, cache):
        return sum(cache) / float(len(cache))


class Std(metrics.Metric):
    """Metric aggregator which calculates the standard deviation of process outputs between calls to reset.

    :param name: The name of this metric.
    :type name: str
    """

    def __init__(self, name):
        super(Std, self).__init__(name)

    def process(self, *args):
        """Compute values required for the std from the input. The input should be a torch Tensor. The sum and sum of
        squares will be computed for all elements in the input.

        :param args:  The output of some previous call to :meth:`.Metric.process`.
        :type args: torch.Tensor

        """
        data = args[0]
        self._sum += data.sum().item()
        self._sum_sq += data.pow(2).sum().item()
        self._count += float(torch.numel(data))

    def process_final(self, *args):
        """Compute and return the final standard deviation.

        :return: The standard deviation of each observation since the last reset call.

        """
        mean = self._sum / self._count
        mean = mean ** 2
        variance = (self._sum_sq / self._count) - mean
        if variance < 0:
            return 0
        else: 
            return variance ** 0.5

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
    """Metric aggregator which calculates the mean of process outputs between calls to reset.

    :param name: The name of this metric.
    :type name: str
    """

    def __init__(self, name):
        super(Mean, self).__init__(name)

    def process(self, *args):
        """Add the input to the rolling sum. Input must be a torch tensor.

        :param args:  The output of some previous call to :meth:`.Metric.process`.
        :type args: torch.Tensor

        """
        data = args[0]
        self._sum += data.sum().item()
        self._count += float(torch.numel(data))

    def process_final(self, *args):
        """Compute and return the mean of all metric values since the last call to reset.

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
