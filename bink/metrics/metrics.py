class Metric(object):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def process(self, state): ...

    def process_final(self, state): ...

    def eval(self): ...

    def train(self): ...

    def reset(self, state): ...


class MetricList(Metric):
    def __init__(self, metric_list):
        super().__init__('metric_list')
        import bink.metrics.defaults as defaults

        self.prefix = ''
        self.metric_list = []

        for i in range(len(metric_list)):
            metric = metric_list[i]

            if str(metric) == metric:
                metric = getattr(defaults, metric)()

            if type(metric) is MetricList:
                self.metric_list = self.metric_list + metric.metric_list
            else:
                self.metric_list.append(metric)

    def _for_list(self, function):
        result = {}
        for metric in self.metric_list:
            out = function(metric)
            if out is not None:
                result.update(out)
        return result

    def _to_dict(self, name, result):
        return {self.prefix + name: result} if result is not None else None

    def process(self, state):
        return self._for_list(lambda metric: self._to_dict(metric.name, metric.process(state)))

    def process_final(self, state):
        return self._for_list(lambda metric: self._to_dict(metric.name, metric.process_final(state)))

    def train(self):
        self.prefix = ''
        self._for_list(lambda metric: metric.train())

    def eval(self):
        self.prefix = 'val_'
        self._for_list(lambda metric: metric.eval())

    def reset(self, state):
        self._for_list(lambda metric: metric.reset(state))


class AdvancedMetric(Metric):
    def __init__(self, name):
        super().__init__(name)
        self._process = self.process_train
        self._process_final = self.process_final_train

    def process_train(self, state): ...

    def process_validate(self, state): ...

    def process_final_train(self, state): ...

    def process_final_validate(self, state): ...

    def process(self, state):
        return self._process(state)

    def process_final(self, state):
        return self._process_final(state)

    def eval(self):
        self._process = self.process_validate
        self._process_final = self.process_final_validate

    def train(self):
        self._process = self.process_train
        self._process_final = self.process_final_train
