"""
Aggregators are a special kind of :class:`.Metric` which takes as input, the output from a previous metric or metrics.
As a result, via a :class:`.MetricTree`, a series of aggregators can collect statistics such as Mean or Standard
Deviation without needing to compute the underlying metric multiple times. This can, however, make the aggregators
complex to use. It is therefore typically better to use the :mod:`decorator API<.metrics.decorators>`.
"""
from collections import deque

import torch

from torchbearer import metrics


class RunningMetric(metrics.AdvancedMetric):
    """A metric which aggregates batches of results and presents a method to periodically process these into a value.

    .. note::

       Running metrics only provide output during training.

    Args:
        name (str): The name of the metric.
        batch_size (int): The size of the deque to store of previous results.
        step_size (int): The number of iterations between aggregations.
    """
    def __init__(self, name, batch_size=50, step_size=10):
        super().__init__(name)
        self._batch_size = batch_size
        self._step_size = step_size
        self._cache = deque()
        self._result = {}

    def _process_train(self, *args):
        """Process the metric for a single train step.

        Args:
            args: The output of some :class:`.Metric`

        Returns:
            The metric value.
        """
        raise NotImplementedError

    def _step(self, cache):
        """Aggregate the cache to produce a single metric value.

        Args:
            cache (list): The current stored metric cache.

        Returns:
            The new metric value.
        """
        raise NotImplementedError

    def process_train(self, *args):
        """Add the current metric value to the cache and call '_step' is needed.

        Args:
            args: The output of some :class:`.Metric`

        Returns:
            The current metric value.
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

        Args:
            state (dict): The current model state.
        """
        self._i = 0


class RunningMean(RunningMetric):
    """A :class:`RunningMetric` which outputs the running mean of its input tensors over the course of an epoch.

    Args:
        name (str): The name of this running mean.
        batch_size (int): The size of the deque to store of previous results.
        step_size (int): The number of iterations between aggregations.
        dim (int, tuple): The dimension(s) on which to perform the mean. If left as None, this will mean over the whole
            Tensor
    """

    def __init__(self, name, batch_size=50, step_size=10, dim=None):
        super().__init__(name, batch_size=batch_size, step_size=step_size)
        self._kwargs = {'dim': dim} if dim is not None else {}

    def _process_train(self, *args):
        data = args[0]
        res = data.mean(**self._kwargs)
        return res

    def _step(self, cache):
        res = sum(cache) / float(len(cache))
        return res.item() if res.numel() <= 1 else res.tolist()


class Std(metrics.Metric):
    """Metric aggregator which calculates the standard deviation of process outputs between calls to reset.

    Args:
        name (str): The name of this metric.
    """

    def __init__(self, name):
        super(Std, self).__init__(name)

    def process(self, *args):
        """Compute values required for the std from the input. The input should be a torch Tensor. The sum and sum of
        squares will be computed for all elements in the input.

        Args:
            args (`torch.Tensor`):  The output of some previous call to :meth:`.Metric.process`.
        """
        data = args[0]
        self._sum += data.sum().item()
        self._sum_sq += data.pow(2).sum().item()
        self._count += float(torch.numel(data))

    def process_final(self, *args):
        """Compute and return the final standard deviation.

        Returns:
            The standard deviation of each observation since the last reset call.
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

        Args:
            state (dict): The model state.
        """
        super().reset(state)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


class Mean(metrics.Metric):
    """Metric aggregator which calculates the mean of process outputs between calls to reset.

    Args:
        name (str): The name of this metric.
        dim (int, tuple): The dimension(s) on which to perform the mean. If left as None, this will mean over the whole
            Tensor
    """

    def __init__(self, name, dim=None):
        super(Mean, self).__init__(name)
        self._kwargs = {'dim': dim} if dim is not None else {}
        self._sum = 0.0
        self._count = 0

    def process(self, *args):
        """Add the input to the rolling sum. Input must be a torch tensor.

        Args:
            args:  The output of some previous call to :meth:`.Metric.process`.
        """
        data = args[0]
        tot = data.sum(**self._kwargs)
        self._sum += tot
        self._count += data.numel() / tot.numel()

    def process_final(self, *args):
        """Compute and return the mean of all metric values since the last call to reset.

        Returns:
            The mean of the metric values since the last call to reset.
        """
        res = self._sum / self._count
        return res.item() if res.numel() <= 1 else res.tolist()

    def reset(self, state):
        """Reset the running count and total.

        Args:
            state (dict): The model state.
        """
        super().reset(state)
        self._sum = 0.0
        self._count = 0
