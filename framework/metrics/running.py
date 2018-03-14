from framework.metrics.metrics import BasicMetric, MetricList
from abc import ABCMeta, abstractmethod
from collections import deque


def mean(metric):
    return Mean(metric)


def statistics(metric):
    return MetricList([mean(metric)])


stats = statistics


class RunningMetric(BasicMetric):
    __metaclass__ = ABCMeta

    def __init__(self, name, batch_size=50, step_size=10):
        super().__init__('running_' + name)
        self._batch_size = batch_size
        self._step_size = step_size
        self._cache = deque()
        self._result = {}
        self.reset()

    @abstractmethod
    def _train(self, state): ...

    @abstractmethod
    def _step(self, cache): ...

    def train(self, state):
        self._cache.append(self._train(state))
        if len(self._cache) > self._batch_size:
            self._cache.popleft()
        if self._i % self._step_size == 0:
            self._result = self._step(self._cache)
        self._i += 1
        return self._result

    def reset(self):
        self._i = 0


class Mean(RunningMetric):
    def __init__(self, metric, batch_size=50, step_size=10):
        super().__init__(metric.name + '_mean', batch_size=batch_size, step_size=step_size)
        self._metric = metric

    def _train(self, state):
        return self._metric.train(state).mean()

    def _step(self, cache):
        return sum(cache) / float(len(cache))
