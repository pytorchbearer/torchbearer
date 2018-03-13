from framework.metrics.metrics import BasicMetric

import numpy as np


def std(metric):
    return AggregateMetric(metric, np.std, extension='_std')


def mean(metric):
    return AggregateMetric(metric, np.mean, extension='_mean')


def max(metric):
    return ComparisonMetric(metric, lambda a, b: a > b, extension='_max')


def min(metric):
    return ComparisonMetric(metric, lambda a, b: a < b, extension='_min')


class AggregateMetric(BasicMetric):
    def __init__(self, metric, function, extension=''):
        super().__init__(metric.name + extension)
        self._metric = metric
        self._function = function
        self.reset()

    def train(self, state):
        self._list_train += self._metric.train(state)

    def validate(self, state):
        self._list_val += self._metric.validate(state)

    def final_train(self, state):
        return self._function(self._list_train)

    def final_validate(self, state):
        return self._function(self._list_val)

    def reset(self):
        self._list_train = []
        self._list_val = []


class ComparisonMetric(BasicMetric):
    def __init__(self, metric, comparator, extension=''):
        super().__init__(metric.name + extension)
        self._metric = metric
        self._comparator = comparator
        self.reset()

    def train(self, state):
        result = self._metric.train(state)
        self._ext_train = result if self._ext_train is None or self._comparator(result, self._ext_train) else self._ext_train

    def validate(self, state):
        result = self._metric.validate(state)
        self._ext_val = result if self._ext_val is None or self._comparator(result, self._ext_val) else self._ext_val

    def final_train(self, state):
        return self._ext_train

    def final_validate(self, state):
        return self._ext_val

    def reset(self):
        self._ext_train = None
        self._ext_val = None
