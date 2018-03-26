class Metric(object):
    def __init__(self, name):
        super().__init__()
        self.training = True
        self.prefix = 'train_'
        self.name = name

    def evaluate(self, state): ...

    def evaluate_final(self, state): ...

    def evaluate_dict(self, state):
        result = self.evaluate(state)
        return {self.prefix + self.name: result} if result is not None else {}

    def evaluate_final_dict(self, state):
        result = self.evaluate_final(state)
        return {self.prefix + self.name: result} if result is not None else {}

    def eval(self):
        self.training = False
        self.prefix = 'val_'

    def train(self):
        self.training = True
        self.prefix = 'train_'

    def reset(self, state): ...


class MetricList(Metric):
    def __init__(self, metric_list):
        super().__init__('metric_list')
        import bink.metrics.defaults as defaults

        for i in range(len(metric_list)):
            metric = metric_list[i]
            if str(metric) == metric:
                metric_list[i] = getattr(defaults, metric)()

        self._metric_list = metric_list

    def _for_list(self, function):
        result = {}
        for metric in self._metric_list:
            out = function(metric)
            if out is not None:
                result.update(out)
        return result

    def evaluate_dict(self, state):
        return self._for_list(lambda metric: metric.evaluate_dict(state))

    def evaluate_final_dict(self, state):
        return self._for_list(lambda metric: metric.evaluate_final_dict(state))

    def train(self):
        self._for_list(lambda metric: metric.train())

    def eval(self):
        self._for_list(lambda metric: metric.eval())

    def reset(self, state):
        self._for_list(lambda metric: metric.reset(state))


class AdvancedMetric(Metric):
    def __init__(self, name):
        super().__init__(name)
        self._process = self.process_train
        self._process_final = self.final_train

    def process_train(self, state): ...

    def process_validate(self, state): ...

    def final_train(self, state): ...

    def final_validate(self, state): ...

    def evaluate(self, state):
        return self._process(state)

    def evaluate_final(self, state):
        return self._process_final(state)

    def eval(self):
        super().eval()
        self._process = self.process_validate
        self._process_final = self.final_validate

    def train(self):
        super().train()
        self._process = self.process_train
        self._process_final = self.final_train
