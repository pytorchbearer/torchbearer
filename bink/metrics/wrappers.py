from bink import metrics

import torch


def std(metric):
    return Std(metric)


def mean(metric):
    return Mean(metric)


def statistics(metric):
    return metrics.MetricList([mean(metric), std(metric)])


stats = statistics


class Std(metrics.BasicMetric):
    def __init__(self, metric):
        super().__init__(metric.name + '_std')
        self._metric = metric

    def train(self, state):
        result = self._metric.train(state)
        self._sum += result.sum()
        self._sum_sq += result.pow(2).sum()
        self._count += result.size(0)

    def validate(self, state):
        result = self._metric.validate(state)
        self._sum += result.sum()
        self._sum_sq += result.pow(2).sum()
        self._count += result.size(0)

    def final_train(self, state):
        mean = self._sum / self._count
        mean = mean ** 2
        return ((self._sum_sq / self._count) - mean) ** 0.5

    def final_validate(self, state):
        mean = self._sum / self._count
        mean = mean ** 2
        return ((self._sum_sq / self._count) - mean) ** 0.5

    def reset(self, state):
        self._metric.reset(state)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


class Mean(metrics.BasicMetric):
    def __init__(self, metric):
        super().__init__(metric.name)
        self._metric = metric

    def train(self, state):
        result = self._metric.train(state)
        self._sum += result.sum()
        self._count += result.size(0)

    def validate(self, state):
        result = self._metric.validate(state)
        self._sum += result.sum()
        self._count += result.size(0)

    def final_train(self, state):
        return self._sum / self._count

    def final_validate(self, state):
        return self._sum / self._count

    def reset(self, state):
        self._metric.reset(state)
        self._sum = 0.0
        self._count = 0


class BatchLambda(metrics.BasicMetric):
    def __init__(self, name, metric_function):
        super().__init__(name)
        self._metric_function = metric_function

    def train(self, state):
        return self._process(state)

    def validate(self, state):
        return self._process(state)

    def _process(self, state):
        return self._metric_function(state['y_true'], state['y_pred'])


class EpochLambda(metrics.BasicMetric):
    def __init__(self, name, metric_function, running=True, step_size=50):
        super().__init__(name)
        self._step = lambda y_true, y_pred: metric_function(y_true, y_pred)
        self._final = lambda y_true, y_pred: metric_function(y_true, y_pred)
        self._step_size = step_size
        self._result = 0.0

        if not running:
            self._step = lambda y_true, y_pred: ...

    def _train(self, state):
        pass

    def train(self, state):
        self._y_true = torch.cat((self._y_true, state['y_true']), dim=0)
        self._y_pred = torch.cat((self._y_pred, state['y_pred'].float()), dim=0)
        if state['t'] % self._step_size == 0:
            self._result = self._step(self._y_true, self._y_pred)
        return self._result

    def final_train(self, state):
        return self._final(self._y_true, self._y_pred)

    def validate(self, state):
        self._y_true = torch.cat((self._y_true, state['y_true']), dim=0)
        self._y_pred = torch.cat((self._y_pred, state['y_pred'].float()), dim=0)

    def final_validate(self, state):
        return self._final(self._y_true, self._y_pred)

    def reset(self, state):
        self._y_true = torch.zeros(0).long()
        self._y_pred = torch.zeros(0, 0)

        if 'use_cuda' in state and state['use_cuda']:
            self._y_true = self._y_true.cuda()
            self._y_pred = self._y_pred.cuda()
