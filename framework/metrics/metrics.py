class Metric(object):
    def __init__(self):
        super().__init__()

    def train_dict(self, state):
        return {}

    def validate_dict(self, state):
        return {}

    def final_train_dict(self, state):
        return {}

    def final_validate_dict(self, state):
        return {}

    def reset(self):
        pass


class MetricList(Metric):
    def __init__(self, list):
        super().__init__()
        self._list = list

    def _for_list(self, function):
        result = {}
        for metric in self._list:
            out = function(metric)
            if out:
                result.update(out)
        return result

    def train_dict(self, state):
        return self._for_list(lambda metric: metric.train_dict(state))

    def validate_dict(self, state):
        return self._for_list(lambda metric: metric.validate_dict(state))

    def final_train_dict(self, state):
        return self._for_list(lambda metric: metric.final_train_dict(state))

    def final_validate_dict(self, state):
        return self._for_list(lambda metric: metric.final_validate_dict(state))

    def reset(self):
        self._for_list(lambda metric: metric.reset())


class BasicMetric(Metric):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def train(self, state):
        pass

    def validate(self, state):
        pass

    def final_train(self, state):
        pass

    def final_validate(self, state):
        pass

    def train_dict(self, state):
        result = self.train(state)
        return {'train_' + self.name: result} if result else {}

    def validate_dict(self, state):
        result = self.validate(state)
        return {'val_' + self.name: result} if result else {}

    def final_train_dict(self, state):
        result = self.final_train(state)
        return {'train_' + self.name: result} if result else {}

    def final_validate_dict(self, state):
        result = self.final_validate(state)
        return {'val_' + self.name: result} if result else {}

    def reset(self):
        pass


class LossMetric(BasicMetric):
    def __init__(self):
        super().__init__('loss')

    def process_train(self, state):
        return self._process(state)

    def process_val(self, state):
        return self._process(state)

    def _process(self, state):
        return state['loss']


class LambdaMetric(BasicMetric):
    def __init__(self, name, metric_function):
        super().__init__(name)
        self._metric_function = metric_function

    def process_train(self, state):
        return self._process(state)

    def process_val(self, state):
        return self._process(state)

    def _process(self, state):
        return self._metric_function(state['y_true'], state['y_pred'])


class Average(BasicMetric):
    def __init__(self, metric):
        super().__init__(metric.name + '_avg')
        self._metric = metric
        self.reset()

    def train(self, state):
        result = self._metric.train(state)
        self._total_train += result * state['y_pred'].shape[0]
        self._count_train += state['y_pred'].shape[0]
        return result

    def validate(self, state):
        result = self._metric.validate(state)
        self._total_val += result * state['y_pred'].shape[0]
        self._count_val += state['y_pred'].shape[0]
        return result

    def final_train(self, state):
        return self._total_train / self._count_train

    def final_validate(self, state):
        return self._total_val / self._count_val

    def reset(self):
        self._total_train = 0.0
        self._count_train = 0
        self._total_val = 0.0
        self._count_val = 0
