from framework.metrics.metrics import BasicMetric

import numpy as np


def std(metric):
    return Std(metric)


def mean(metric):
    return Mean(metric)


class Std(BasicMetric):
    def __init__(self, metric):
        super().__init__(metric.name + '_std')
        self._metric = metric
        self.reset()

    def train(self, state):
        result = self._metric.train(state).float()
        self._sum_train += result.sum()
        self._sum_sq_train += result.pow(2).sum()
        self._count_train += result.size(0)

    def validate(self, state):
        result = self._metric.validate(state).float()
        self._sum_val += result.sum()
        self._sum_sq_val += result.pow(2).sum()
        self._count_val += result.size(0)

    def final_train(self, state):
        mean = self._sum_train / self._count_train
        mean = mean ** 2
        return (self._sum_sq_train / self._count_train) - mean

    def final_validate(self, state):
        mean = self._sum_val / self._count_val
        mean = mean ** 2
        return (self._sum_sq_val / self._count_val) - mean

    def reset(self):
        self._sum_train = 0.0
        self._sum_sq_train = 0.0
        self._count_train = 0
        self._sum_val = 0.0
        self._sum_sq_val = 0.0
        self._count_val = 0


class Mean(BasicMetric):
    def __init__(self, metric):
        super().__init__(metric.name + '_mean')
        self._metric = metric
        self.reset()

    def train(self, state):
        result = self._metric.train(state)
        self._sum_train += result.sum()
        self._count_train += result.size(0)

    def validate(self, state):
        result = self._metric.validate(state)
        self._sum_val += result.sum()
        self._count_val += result.size(0)

    def final_train(self, state):
        return self._sum_train / self._count_train

    def final_validate(self, state):
        return self._sum_val / self._count_val

    def reset(self):
        self._sum_train = 0.0
        self._count_train = 0
        self._sum_val = 0.0
        self._count_val = 0
