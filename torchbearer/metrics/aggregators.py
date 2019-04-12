"""
Aggregators are a special kind of :class:`.Metric` which takes as input, the output from a previous metric or metrics.
As a result, via a :class:`.MetricTree`, a series of aggregators can collect statistics such as Mean or Standard
Deviation without needing to compute the underlying metric multiple times. This can, however, make the aggregators
complex to use. It is therefore typically better to use the :mod:`decorator API<.metrics.decorators>`.
"""
from collections import deque

import torch

from torchbearer.bases import Metric
from .metrics import AdvancedMetric


class RunningMetric(AdvancedMetric):
    """A metric which aggregates batches of results and presents a method to periodically process these into a value.

    .. note::

       Running metrics only provide output during training.

    Args:
        name (str): The name of the metric.
        batch_size (int): The size of the deque to store of previous results.
        step_size (int): The number of iterations between aggregations.
    """
    def __init__(self, name, batch_size=50, step_size=10):
        super(RunningMetric, self).__init__(name)
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
        super(RunningMean, self).__init__(name, batch_size=batch_size, step_size=step_size)
        self._kwargs = {'dim': dim} if dim is not None else {}

    def _process_train(self, *args):
        data = args[0]
        res = data.mean(**self._kwargs)
        return res

    def _step(self, cache):
        res = sum(cache) / float(len(cache))
        return res.item() if res.numel() <= 1 else res.tolist()


class Var(Metric):
    """Metric aggregator which calculates the **sample** variance of process outputs between calls to reset.
    Optionally calculate the population variance if ``unbiased = False``.

    Args:
        name (str): The name of this metric.
        unbiased (bool): If True (default), calculates the sample variance, else, the population variance
        dim (int, tuple): The dimension(s) on which to compute the std. If left as None, this will operate over the
            whole Tensor
    """

    def __init__(self, name, unbiased=True, dim=None):
        super(Var, self).__init__(name)
        self._unbiased = unbiased
        self._kwargs = {'dim': dim} if dim is not None else {}
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0

    def process(self, *args):
        """Compute values required for the variance from the input. The input should be a torch Tensor. The sum and sum
        of squares will be computed over the provided dimension.

        Args:
            args (torch.Tensor):  The output of some previous call to :meth:`.Metric.process`.
        """
        data = args[0]
        tot = data.sum(**self._kwargs)
        self._sum += tot
        self._sum_sq += data.pow(2).sum(**self._kwargs)
        self._count += data.numel() / tot.numel()

    def _process_final(self):
        mean = self._sum / self._count
        mean = mean.pow(2)
        variance = ((self._sum_sq / self._count) - mean).clamp(min=0)
        if self._unbiased:
            variance = variance * (self._count / (self._count - 1.0))
        return variance

    def process_final(self, *args):
        """Compute and return the final variance.

        Returns:
            The variance of each observation since the last reset call.
        """
        res = self._process_final()
        return res.item() if res.numel() <= 1 else res.tolist()

    def reset(self, state):
        """Reset the statistics to compute the next variance.

        Args:
            state (dict): The model state.
        """
        super(Var, self).reset(state)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


class Std(Var):
    """Metric aggregator which calculates the **sample** standard deviation of process outputs between calls to reset.
    Optionally calculate the population std if ``unbiased = False``.

    Args:
        name (str): The name of this metric.
        unbiased (bool): If True (default), calculates the sample standard deviation, else, the population standard
            deviation
        dim (int, tuple): The dimension(s) on which to compute the std. If left as None, this will operate over the
            whole Tensor
    """

    def __init__(self, name, unbiased=True, dim=None):
        super(Std, self).__init__(name, unbiased=unbiased, dim=dim)

    def process_final(self, *args):
        """Compute and return the final standard deviation.

        Returns:
            The standard deviation of each observation since the last reset call.
        """
        res = super(Std, self)._process_final().sqrt()
        return res.item() if res.numel() <= 1 else res.tolist()


class Mean(Metric):
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
        super(Mean, self).reset(state)
        self._sum = 0.0
        self._count = 0
