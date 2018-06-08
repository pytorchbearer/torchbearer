from bink import metrics

import torch


def std(metric):
    return Std(metric)


def mean(metric):
    return Mean(metric)


def statistics(metric):
    return metrics.MetricList([mean(metric), std(metric)])


stats = statistics


class Wrapper(metrics.Metric):
    def __init__(self, metric, postfix):
        super().__init__(metric.name + postfix)
        self._metric = metric

    def eval(self):
        super().eval()
        self._metric.eval()

    def train(self):
        super().train()
        self._metric.train()

    def reset(self, state):
        super().reset(state)
        self._metric.reset(state)


class Std(Wrapper):
    def __init__(self, metric):
        super().__init__(metric, '_std')

    def process(self, state):
        result = self._metric.process(state)
        self._sum += result.sum().item()
        self._sum_sq += result.pow(2).sum().item()

        if result.size() == torch.Size([]):
            self._count += 1
        else:
            self._count += result.size(0)

    def process_final(self, state):
        mean = self._sum / self._count
        mean = mean ** 2
        return ((self._sum_sq / self._count) - mean) ** 0.5

    def reset(self, state):
        super().reset(state)
        self._sum = 0.0
        self._sum_sq = 0.0
        self._count = 0


class Mean(Wrapper):
    def __init__(self, metric):
        super().__init__(metric, '')

    def process(self, state):
        result = self._metric.process(state)
        self._sum += result.sum().item()

        if result.size() == torch.Size([]):
            self._count += 1
        else:
            self._count += result.size(0)

    def process_final(self, state):
        return self._sum / self._count

    def reset(self, state):
        super().reset(state)
        self._sum = 0.0
        self._count = 0


class BatchLambda(metrics.Metric):
    def __init__(self, name, metric_function):
        super().__init__(name)
        self._metric_function = metric_function

    def process(self, state):
        return self._metric_function(state['y_pred'], state['y_true'])


class EpochLambda(metrics.AdvancedMetric):
    def __init__(self, name, metric_function, running=True, step_size=50):
        super().__init__(name)
        self._step = metric_function
        self._final = metric_function
        self._step_size = step_size
        self._result = 0.0

        if not running:
            self._step = lambda y_true, y_pred: ...

    def process_train(self, state):
        self._y_true = torch.cat((self._y_true, state['y_true']), dim=0)
        self._y_pred = torch.cat((self._y_pred, state['y_pred'].float()), dim=0)
        if state['t'] % self._step_size == 0:
            self._result = self._step(self._y_true, self._y_pred)
        return self._result

    def process_final_train(self, state):
        return self._final(self._y_true, self._y_pred)

    def process_validate(self, state):
        self._y_true = torch.cat((self._y_true, state['y_true']), dim=0)
        self._y_pred = torch.cat((self._y_pred, state['y_pred'].to(self._y_pred.dtype)), dim=0)

    def process_final_validate(self, state):
        return self._final(self._y_true, self._y_pred)

    def reset(self, state):
        super().reset(state)
        self._y_true = torch.zeros(0).long()
        self._y_pred = torch.zeros(0, 0)

        self._y_true = self._y_true.to(state['device'])
        self._y_pred = self._y_pred.to(state['device'], state['dtype'])
