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


class Loss(BasicMetric):
    def __init__(self):
        super().__init__('loss')

    def train(self, state):
        return self._process(state)

    def validate(self, state):
        return self._process(state)

    def _process(self, state):
        return state['loss']


class Lambda(BasicMetric):
    def __init__(self, name, metric_function):
        super().__init__(name)
        self._metric_function = metric_function

    def train(self, state):
        return self._process(state)

    def validate(self, state):
        return self._process(state)

    def _process(self, state):
        return self._metric_function(state['y_true'], state['y_pred'])
