from torchbearer import metrics
from abc import ABCMeta, abstractmethod
from collections import deque


def running_mean(metric):
    """Utility function to wrap the given metric in a :class:`RunningMean` and return the result.

    :param metric: The metric to wrap.
    :type metric: Metric
    :return: RunningMean -- A running mean for the given metric.

    """
    return RunningMean(metric)


def running_statistics(metric):
    """Utility function to wrap the given metric in a set of running statistics and return the result.

    :param metric: The metric to wrap.
    :type metric: Metric
    :return: MetricList -- A list of running statistics.

    """
    return metrics.MetricList([running_mean(metric)])


running_stats = running_statistics


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
        super().__init__('running_' + name)
        self._batch_size = batch_size
        self._step_size = step_size
        self._cache = deque()
        self._result = {}

    @abstractmethod
    def _process_train(self, state):
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

    def process_train(self, state):
        """Add the current metric value to the cache and call '_step' is needed.

        :param state: The current model state.
        :type state: dict
        :return: The current metric value.

        """
        self._cache.append(self._process_train(state))
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

    def __init__(self, metric, batch_size=50, step_size=10):
        """Wrap the given metric in initialise the parent :class:`RunningMetric`.

        :param metric: The metric to wrap.
        :type metric: Metric
        :param batch_size: The size of the deque to store of previous results.
        :type batch_size: int
        :param step_size: The number of iterations between aggregations.
        :type step_size: int
        """
        super().__init__(metric.name, batch_size=batch_size, step_size=step_size)
        self._metric = metric

    def _process_train(self, state):
        return self._metric.process(state).mean().item()

    def _step(self, cache):
        return sum(cache) / float(len(cache))
